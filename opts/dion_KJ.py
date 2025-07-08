import math
import torch
import os
from torch.optim.optimizer import Optimizer, ParamsT
from typing import Any, Dict, Tuple, Optional


def extract_PQ(
    u: torch.Tensor,  # shape (n, m)
    Q_init: torch.Tensor,  # shape (m, rank)
    method: str = "qr", 
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns a rank-`rank` low-rank approximation of `u` by extracting
    P and Q such that (approximately) u ≈ P @ Q^T.

    The SVD method computes a truncated SVD. The power iteration method
    uses iterative QR decompositions.
    """
    if method not in ("svd", "qr", "cqr", "rcqr", "flash-qr", "power"):
        raise ValueError(f"Unknown method: {method}")

    if method == "svd":
        rank = Q_init.size(1)
        U, S, V = torch.linalg.svd(u, full_matrices=False)
        P = U[:, :rank]  # (n, rank)
        S_r = S[:rank]
        Q = V.T[:, :rank] * S_r.unsqueeze(0)  # (m, rank)
        return P, Q
    
    else:
        if Q_init is None:
            raise ValueError(
                "For power iter methods, please supply a Q_init of shape (m, rank)."
            )

        # Power iteration method
        Q = Q_init
        Q = u.T @ u @ Q  
        Q = orthogonalize(Q, method=method)
        P = u @ Q  # shape (n, rank) 
    
        # """
        # If input is all zero, P and Q will be nan or all zero.
        # We want to return the conditional expressions:

        #     if is_all_zero:
        #         P = torch.zeros_like(P)
        #         Q = Q_init
        #     else:
        #         P = P
        #         Q = Q

        # Here this is implemented without data-dependent control flow.
        # """
        # is_all_zero = (u == 0).all()
        # not_all_zero = ~is_all_zero
        # P = P.nan_to_num() * not_all_zero
        # Q = Q.nan_to_num() * not_all_zero + Q_init * is_all_zero

        return P, Q

def orthogonalize(
    P: torch.Tensor,
    method: str,
    oversample: float = 1.25,  # only used for method "rcqr"
) -> torch.Tensor:
    """
    Orthogonalize the input matrix using the specified method.
        - "qr": Householder QR (torch.linalg.qr)
        - "cqr": Cholesky QR
        - "rcqr": Randomized Cholesky QR
        - "flash-qr" : Block-fused randomised QR (no full R materialised)
    """
    m, n = P.shape

    # Always use standard QR if matrix is square or wide
    if method == "qr" or m <= n:
        Q, _ = torch.linalg.qr(P)
        return Q

    # Cholesky QR (may not be numerically stable)
    elif method == "cqr":
        R, _ = torch.linalg.cholesky_ex(P.T @ P, upper=True)
        Q = torch.linalg.solve_triangular(R, P, upper=True, left=False)
        return Q

    # Randomized Cholesky QR
    elif method == "rcqr":
        # Compute size k and round up to next multiple of 128
        k = math.ceil(oversample * n / 128.0) * 128

        # Generate random sketching matrix of shape (k, m)
        std = math.sqrt(1.0 / k)
        S = torch.empty((k, m), device=P.device, dtype=P.dtype).normal_(std=std)

        # Calculate right triangular matrix R using standard QR, and solve for Q
        _, R = torch.linalg.qr(S @ P, mode="r")
        Q = torch.linalg.solve_triangular(R, P, upper=True, left=False)

        # Apply another iteration of Cholesky QR to better orthogonalize Q
        R, _ = torch.linalg.cholesky_ex(Q.T @ Q, upper=True)
        Q = torch.linalg.solve_triangular(R, Q, upper=True, left=False)
        return Q

    else:
        raise ValueError(f"Unknown orthogonalization method: {method}")




@torch.compile(dynamic=True)
def dion_update(
    X: torch.Tensor,          # Model weights (modified in place)
    M: torch.Tensor,          # Momentum buffer (modified in place)
    Q: torch.Tensor,          # Q matrix for power iteration
    lr: torch.Tensor,         # Learning rate (scalar tensor)
    mu: torch.Tensor,         # Momentum factor (scalar tensor)
    weight_decay: torch.Tensor,   # Weight decay (scalar tensor)
    approx_method: str,       # Approximation method for low-rank factorization
    epsilon: float,
    step: int,                 
    warmup_steps: int,         
    recompute_period: int,    
) -> torch.Tensor:
    """
    Dion optimizer update.
    Until step 10: do a factorisation every step.
    Afterwards: do a factorisation every 4th step.
    """
    # Apply weight decay
    X.mul_(1 - lr * weight_decay)

    # Accumulate gradient into momentum
    M.add_(X.grad)

    # Decide whether to recompute the low-rank factors
    recompute = (step <= warmup_steps) or (step % recompute_period == 0)
    
            
    if recompute: 
        # ---- power-iteration path ------------------------------------
        R, Qnew = extract_PQ(
            u=M,
            Q_init=Q,
            method=approx_method, 
        )
        Q.copy_(Qnew)  # Update Q with the new orthonormal basis

        # Error-feedback
        M.addmm_(R, Q.T, alpha=-(1 - mu))

        # Column-normalise R – becomes the next Q
        P =  R / (R.norm(dim=0, keepdim=True) + epsilon)

        # Scale learning-rate like the paper
        fan_out, fan_in = X.size()
        scale = (fan_out / fan_in) ** 0.5
 
        return scale * P @ Q.T
    else:
        # ---- cheap path: reuse previous Q ---------------------------------
        P = M @ Q
        M.addmm_(P, Q.T, alpha=-(1 - mu))

        Pn = P / (P.norm(dim=0, keepdim=True) + epsilon)

        fan_out, fan_in = X.size()
        scale = (fan_out / fan_in) ** 0.5 

        return scale * Pn @ Q.T  
 

@torch.compile(dynamic=True)
def adamw_update(
    X: torch.Tensor,  # Model weights (modified in place)
    M: torch.Tensor,  # Momentum buffer (modified in place)
    V: torch.Tensor,  # Variance buffer (modified in place)
    lr: torch.Tensor,  # Learning rate (scalar tensor)
    beta1: torch.Tensor,  # Beta 1 (scalar tensor)
    beta2: torch.Tensor,  # Beta 2 (scalar tensor)
    weight_decay: torch.Tensor,  # Weight decay (scalar tensor)
    step: int,
    epsilon: float,
):
    """
    AdamW optimizer algorithm.
    """
    # Apply weight decay
    X.mul_(1 - lr * weight_decay)

    # Update momentum and variance
    # M = beta1 * M + (1 - beta1) * X.grad
    M.lerp_(X.grad, 1 - beta1)
    # V = beta2 * V + (1 - beta2) * X.grad * X.grad
    V.mul_(beta2).addcmul_(X.grad, X.grad, value=1 - beta2)

    # Bias correction
    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step
    bias_correction2_sqrt = bias_correction2.sqrt()

    # The goal is to compute the following in-place:
    # M = M / bias_correction1
    # V = V / bias_correction2
    # X = X - lr * M / (sqrt(V) + epsilon)

    # sqrt(V / bias_correction2) = sqrt(V) / sqrt(bias_correction2)
    denom = V.sqrt().div_(bias_correction2_sqrt).add_(epsilon)

    # Adjust learning rate to include bias correction 1
    adj_lr = lr / bias_correction1

    # Weight update
    # X = X - adj_lr * M / denom
    X.addcdiv_(M, denom, value=-adj_lr)

@torch.compile(dynamic=True)
def lion_update(
    X: torch.Tensor,  # Model weights (modified in place)
    M: torch.Tensor,  # Momentum buffer (modified in place)
    lr: torch.Tensor,  # Learning rate (scalar tensor)
    beta1: torch.Tensor,  # Beta 1 (scalar tensor)
    beta2: torch.Tensor,  # Beta 2 (scalar tensor)
    weight_decay: torch.Tensor,  # Weight decay (scalar tensor)
):
    """
    Lion optimizer algorithm. Sign update should guarantee RMS norm equal to 1.
    """
    # Apply weight decay
    X.mul_(1 - lr * weight_decay)

    # Compute sign update
    # U = sign(beta1 * M + (1 - beta1) * X.grad)
    U = M.lerp(X.grad, 1 - beta1).sign_()

    # Update momentum with new gradient
    # M = beta2 * M + (1 - beta2) * X.grad
    M.lerp_(X.grad, 1 - beta2)

    # Weight update
    # X = X - lr * U
    X.add_(U, alpha=-lr)


class DionKJ(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float,
        mu: float = 0.95,  # For Dion
        betas: Tuple[float, float] = (0.9, 0.95),  # For AdamW and Lion
        weight_decay: float = 0.01,
        rank: int = 8, 
        epsilon: float = 1e-8,
        approx_method: str = "qr",
        warmup_steps: int = 10,         
        recompute_period: int = 8,     
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if mu < 0.0:
            raise ValueError(f"Invalid momentum factor (mu): {mu}")
        if len(betas) != 2 or betas[0] < 0.0 or betas[1] < 0.0:
            raise ValueError(f"Invalid betas: {betas}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if rank <= 0:
            raise ValueError(f"Invalid rank value: {rank}") 
        if approx_method not in ("qr", "cqr", "rcqr", "svd", "flash-qr"):
            raise ValueError(f"Unknown approximation method: {approx_method}")

        defaults = dict(
            lr=lr,
            mu=mu,
            beta1=betas[0],
            beta2=betas[1],
            weight_decay=weight_decay,
            algorithm="dion",
            vector_dim=-1,
            step=0,
        )
        super().__init__(params, defaults)

        self.rank = rank 
        self.epsilon = torch.tensor(epsilon)
        self.approx_method = approx_method
        self.warmup_steps = warmup_steps
        self.recompute_period = recompute_period

        print(
            f"Dion optimizer initialized with:\n"
            f"  - Learning rate: {lr}\n"
            f"  - Momentum factor (mu): {mu}\n"
            f"  - Weight decay: {weight_decay}\n"
            f"  - Rank: {rank}\n" 
            f"  - Epsilon: {epsilon}\n"
            f"  - Rank-r approximation method: {approx_method}\n"
            f"  - Warm-up steps:       {warmup_steps}\n"
            f"  - Recompute every N:   {recompute_period}\n"
        )

        # Check that all Dion parameters are 2D tensors
        for group in self.param_groups:
            if group["algorithm"] == "dion":
                for p in group["params"]:
                    if p.dim() != 2:
                        raise ValueError(
                            f"Expected matrix parameters to be 2D tensor, got {p.dim()}D."
                        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))

        for group in self.param_groups:
            algo = group["algorithm"]
            group["step"] += 1
            step = group["step"]

            # Wrap hyperparameters in tensors for torch.compile
            lr = torch.tensor(group["lr"])
            mu = torch.tensor(group["mu"])
            beta1 = torch.tensor(group["beta1"])
            beta2 = torch.tensor(group["beta2"])
            weight_decay = torch.tensor(group["weight_decay"])

            if algo == "dion":

                total_params = sum(p.numel() for p in group["params"])
                device = group["params"][0].device
                # Create a flat buffer to hold the update direction for the entire parameter group.
                updates_flat = torch.zeros(total_params, device=device, dtype=group["params"][0].dtype)
                curr_idx = 0

                for i, param in enumerate(group["params"]):
                    numel = param.numel()

                    # Only the owner rank (defined by modulo over param index) 
                    # computes the update and keeps M and Q in its state.
                    if i % world_size == rank:

                        if param.grad is None:
                            raise ValueError("Gradient is None.")
                        state = self.state[param]
                        if not state:
                            self._init_opt_state_dion(param, state)

                        update = dion_update(
                                    X=param,
                                    M=state["momentum"],
                                    Q=state["Q"],
                                    lr=lr,
                                    mu=mu,
                                    weight_decay=weight_decay,
                                    approx_method=self.approx_method, 
                                    epsilon=self.epsilon,
                                    step=step,                        
                                    warmup_steps=self.warmup_steps,   
                                    recompute_period=self.recompute_period,
                                )

                        updates_flat[curr_idx : curr_idx + numel] = update.flatten()
                    
                    # Non-owner replicas leave this portion as zeros.
                    curr_idx += numel


                # All-reduce to gather computed update directions from all ranks.
                # All-reduce is equivalent to all-gather here.
                torch.distributed.all_reduce(updates_flat, op=torch.distributed.ReduceOp.SUM)

                curr_idx = 0
                for param in group["params"]:
                    numel = param.numel()
                    update_tensor = updates_flat[curr_idx: curr_idx + numel].view_as(param)
                    param.add_(update_tensor, alpha=-lr)
                    curr_idx += numel

            elif algo == "adamw":
                for param in group["params"]:
                    if param.grad is None:
                        raise ValueError("Gradient is None.")

                    # Get optimizer state for this parameter
                    state = self.state[param]
                    if not state:
                        self._init_opt_state_adam(param, state)

                    # Apply update
                    adamw_update(
                        X=param,
                        M=state["momentum"],
                        V=state["variance"],
                        lr=lr,
                        beta1=beta1,
                        beta2=beta2,
                        weight_decay=weight_decay,
                        step=step,
                        epsilon=self.epsilon,
                    )

            elif algo == "lion":
                for param in group["params"]:
                    if param.grad is None:
                        raise ValueError("Gradient is None.")

                    # Get optimizer state for this parameter
                    state = self.state[param]
                    if not state:
                        self._init_opt_state_lion(param, state)

                    # Apply update
                    lion_update(
                        X=param,
                        M=state["momentum"],
                        lr=lr,
                        beta1=beta1,
                        beta2=beta2,
                        weight_decay=weight_decay,
                    )

            else:
                raise ValueError(f"Unknown algorithm: {algo}")

        return loss

    def _init_opt_state_dion(self, param: torch.Tensor, state: Dict[str, Any]):
        # Initialize momentum and Q for a 2D matrix parameter
        state["momentum"] = torch.zeros_like(param)
        r = min(self.rank, min(param.shape))
        state["Q"] = torch.randn(
            (param.size(1), r), device=param.device, dtype=param.dtype
        )

    def _init_opt_state_adam(self, param: torch.Tensor, state: Dict[str, Any]):
        state["momentum"] = torch.zeros_like(param)
        state["variance"] = torch.zeros_like(param)

    def _init_opt_state_lion(self, param: torch.Tensor, state: Dict[str, Any]):
        state["momentum"] = torch.zeros_like(param)