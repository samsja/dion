import math
import torch
from torch.optim.optimizer import Optimizer, ParamsT
from typing import Any, Dict, Tuple, Optional


def extract_PQ(
    u: torch.Tensor,  # shape (n, m)
    Q_init: torch.Tensor,  # shape (m, rank)
    method: str = "qr",
    power_iters: int = 1,  # ignored for SVD method
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns a rank-`rank` low-rank approximation of `u` by extracting
    P and Q such that (approximately) u ≈ P @ Q^T.

    The SVD method computes a truncated SVD. The power iteration method
    uses iterative QR decompositions.
    """
    if method not in ("svd", "qr", "cqr", "rcqr", "flash-qr"):
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
        for _ in range(power_iters):
            P = u @ Q  # shape (n, rank)
            P = orthogonalize(P, method=method)
            Q = u.T @ P  # shape (m, rank)

        """
        If input is all zero, P and Q will be nan or all zero.
        We want to return the conditional expressions:

            if is_all_zero:
                P = torch.zeros_like(P)
                Q = Q_init
            else:
                P = P
                Q = Q

        Here this is implemented without data-dependent control flow.
        """
        is_all_zero = (u == 0).all()
        not_all_zero = ~is_all_zero
        P = P.nan_to_num() * not_all_zero
        Q = Q.nan_to_num() * not_all_zero + Q_init * is_all_zero

        return P, Q


def flash_orthogonalize(
    A: torch.Tensor,
    block_size: int = 16,
    oversample: float = 1.25,  # keep same default as rcqr
) -> torch.Tensor:
    """
    Block-fused QR that never materialises the full R.
      A : (m, n) matrix to orthogonalise   (modified in-place)
      block_size : number of columns processed together
    The sketch S is generated on-the-fly so the call signature stays
    compatible with `orthogonalize`.
    """
    m, n = A.shape
    k = math.ceil(oversample * n / 128.0) * 128
    assert n % block_size == 0, "`n` must be divisible by block_size"

    # Gaussian sketch S  ~  N(0, 1/k)
    std = math.sqrt(1.0 / k)
    S = torch.empty((k, m), dtype=A.dtype, device=A.device).normal_(std=std)
    SA = S @ A  # (k, n)   random projection

    # --- the original kernel ------------------------------------------------
    Q = SA.clone()
    num_blocks = n // block_size
    for blk in range(num_blocks):
        c0, c1 = blk * block_size, (blk + 1) * block_size
        Q_blk, A_blk = Q[:, c0:c1], A[:, c0:c1]

        # subtract projections of previous blocks
        for pblk in range(blk):
            p0, p1 = pblk * block_size, (pblk + 1) * block_size
            R = Q[:, p0:p1].T @ Q_blk
            Q_blk.sub_(Q[:, p0:p1] @ R)
            A_blk.sub_(A[:, p0:p1] @ R)

        # classical Gram–Schmidt inside the block
        for j in range(block_size):
            if j:  # proj on previous vecs
                dots = Q_blk[:, :j].T @ Q_blk[:, j]
                Q_blk[:, j].sub_(Q_blk[:, :j] @ dots)
                A_blk[:, j].sub_(A_blk[:, :j] @ dots)
            nrm = torch.linalg.norm(Q_blk[:, j])
            Q_blk[:, j].div_(nrm)
            A_blk[:, j].div_(nrm)
    # ------------------------------------------------------------------------
    return A  # orthogonalised copy (same shape as input)


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
    if method == "qr":
        Q, _ = torch.linalg.qr(P)
        return Q

    # Cholesky QR (may not be numerically stable)
    elif method == "cqr":
        R, info = torch.linalg.cholesky_ex(P.T @ P, upper=True)
        if info.item() == 0:
            Q = torch.linalg.solve_triangular(R, P, upper=True, left=False)
            return Q
        else:
            # If CQR fails do standard QR. Note that if you are in the right
            # regime this will happen rarely. 
            Q, _ = torch.linalg.qr(P)
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

    # Flash QR
    elif method == "flash-qr":
        # Always use sketch + fused kernel, whether m >= n or not.
        return flash_orthogonalize(P, block_size=16, oversample=oversample)

    else:
        raise ValueError(f"Unknown orthogonalization method: {method}")


import pdb
@torch.compile(dynamic=True)
def dion_update(
    X: torch.Tensor,  # Model weights (modified in place)
    M: torch.Tensor,  # Momentum buffer (modified in place)
    Q: torch.Tensor,  # Q matrix for power iteration
    lr: torch.Tensor,  # Learning rate (scalar tensor)
    mu: torch.Tensor,  # Momentum factor (scalar tensor)
    weight_decay: torch.Tensor,  # Weight decay (scalar tensor)
    approx_method: str,  # Approximation method for low-rank factorization
    power_iters: int,  # Number of power iterations
    epsilon: float,
) -> torch.Tensor:
    """
    Dion optimizer algorithm.
    """
    # Apply weight decay
    X.mul_(1 - lr * weight_decay)

    # Add new gradient to momentum
    M.add_(X.grad)

    # Compute low-rank approximation of M = P @ Q^T
    P, R = extract_PQ(
        u=M,
        Q_init=Q,
        method=approx_method,
        power_iters=power_iters,
    )

    # Error feedback
    # M = M - (1 - mu) * (P @ R.T)
    M.addmm_(P, R.T, alpha=-(1 - mu))

    # Column normalize R to get new Q
    
    # Normalize R for orthonormality  
    Q = R / (R.norm(dim=0, keepdim=True) + epsilon)
    


    # Compute update scale factor
    fan_out = X.size(0)
    fan_in = X.size(1)
    scaled_lr = ((fan_out / fan_in) ** 0.5) * lr

    # Apply weight update
    # X = X - scaled_lr * (P @ Q.T)
    X.addmm_(P, Q.T, alpha=-scaled_lr)

    # Return new Q for next iteration
    return Q


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
def clion_update(
    X: torch.Tensor,  # Model weights (modified in place)
    M: torch.Tensor,  # Momentum buffer (modified in place)
    lr: torch.Tensor,  # Learning rate (scalar tensor)
    beta1: torch.Tensor,  # Beta 1 (scalar tensor)
    beta2: torch.Tensor,  # Beta 2 (scalar tensor)
    weight_decay: torch.Tensor,  # Weight decay (scalar tensor)
    vector_dim: Optional[int],  # Dimension to normalize vectors
    epsilon: float,
):
    """
    C-Lion optimizer algorithm based on paper: https://arxiv.org/pdf/2411.16085
    Modified version to maintain constant RMS norm for weight update.

    The RMS norm is calculated over vector_dim. Setting to -1 (last dim)
    should work for both 1D bias vectors and 2D embedding matrices.
    If vector_dim is None, the entire tensor is used for normalization.
    """
    # Apply weight decay
    X.mul_(1 - lr * weight_decay)

    # Compute sign update
    # U = sign(beta1 * M + (1 - beta1) * X.grad)
    U = M.lerp(X.grad, 1 - beta1).sign_()

    # Update momentum with new gradient
    # M = beta2 * M + (1 - beta2) * X.grad
    M.lerp_(X.grad, 1 - beta2)

    # Compute boolean mask where update and gradient have the same sign
    # X.grad will be no longer needed, so we can modify in place
    # mask = sign(X.grad) * U >= 0
    mask = X.grad.sign_().mul_(U).greater_equal_(0)

    # Adjust by sqrt to maintain the same RMS norm
    if vector_dim is None:
        vec_len = mask.numel()
        vec_sum = mask.sum()
    else:
        vec_len = mask.size(vector_dim)
        vec_sum = mask.sum(dim=vector_dim, keepdim=True)
    adj_lr = lr * (vec_len / (vec_sum + epsilon)).sqrt()

    # Weight update
    # X = X - adj_lr * mask * U
    X.addcmul_(U, mask, value=-adj_lr)


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


class Dion(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float,
        mu: float = 0.95,  # For Dion
        betas: Tuple[float, float] = (0.9, 0.95),  # For AdamW and Lion
        weight_decay: float = 0.01,
        rank: int = 8,
        power_iters: int = 1,
        epsilon: float = 1e-8,
        approx_method: str = "qr",
        total_steps: int = 3000,
        qr_warmup: float = 0.05,
        efficient: bool = False
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
        if power_iters <= 0:
            raise ValueError(f"Invalid power iterations: {power_iters}")
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
        self.power_iters = power_iters
        self.epsilon = torch.tensor(epsilon)
        self.approx_method = approx_method
        self.total_steps = total_steps
        self.qr_warmup = qr_warmup
        self.efficient = efficient 

        print(
            f"Dion optimizer initialized with:\n"
            f"  - Learning rate: {lr}\n"
            f"  - Momentum factor (mu): {mu}\n"
            f"  - Weight decay: {weight_decay}\n"
            f"  - Rank: {rank}\n"
            f"  - Power iterations: {power_iters}\n"
            f"  - Epsilon: {epsilon}\n"
            f"  - Rank-r approximation method: {approx_method}\n"
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
                for param in group["params"]:
                    if param.grad is None:
                        raise ValueError("Gradient is None.")

                    # Get optimizer state for this parameter
                    state = self.state[param]
                    if not state:
                        self._init_opt_state_dion(param, state)

                    approx_method = self.approx_method
                    if self.efficient and step >= (self.total_steps * self.qr_warmup):
                        # Use Cholesky QR with fallback after landscape becomes well-conditioned
                        approx_method = "cqr"

                    # Apply update
                    Q_new = dion_update(
                        X=param,
                        M=state["momentum"],
                        Q=state["Q"],
                        lr=lr,
                        mu=mu,
                        weight_decay=weight_decay,
                        approx_method=approx_method,
                        power_iters=self.power_iters,
                        epsilon=self.epsilon,
                    )
                    state["Q"] = Q_new

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

            elif algo == "clion":
                vector_dim = group["vector_dim"]
                for param in group["params"]:
                    if param.grad is None:
                        raise ValueError("Gradient is None.")

                    # Get optimizer state for this parameter
                    state = self.state[param]
                    if not state:
                        self._init_opt_state_lion(param, state)

                    # Apply update
                    clion_update(
                        X=param,
                        M=state["momentum"],
                        lr=lr,
                        beta1=beta1,
                        beta2=beta2,
                        weight_decay=weight_decay,
                        vector_dim=vector_dim,
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
