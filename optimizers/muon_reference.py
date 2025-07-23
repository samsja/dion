import math
import os
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.optim.optimizer import Optimizer


@torch.compile
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz iteration to approximate the orthogonalization of G.
    """
    assert len(G.shape) == 2, "Expected 2D tensor"
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X /= X.norm() + eps
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


# Muon version based on code from Moonlight
# https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py
class MuonMoonlight(Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
    """

    def __init__(
        self,
        muon_params=None,
        lr=1e-3,
        weight_decay=0.1,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adjust_lr="adam",
        adamw_params=None,
        lion_params=None,
        betas=(0.95, 0.95),
        epsilon=1e-8,
    ):

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adjust_lr=adjust_lr,
            betas=betas,
            epsilon=epsilon,
        )

        muon_params = list(muon_params) if muon_params is not None else []
        adamw_params = list(adamw_params) if adamw_params is not None else []
        lion_params = list(lion_params) if lion_params is not None else []
        params = muon_params + adamw_params + lion_params
        super().__init__(params, defaults)

        # Sort parameters into those for which we will use Muon, and those for which we will not
        for param_or_param_group in muon_params:
            if isinstance(param_or_param_group, dict):
                for p in param_or_param_group["params"]:
                    assert p.ndim == 2, p.ndim
                    self.state[p]["algorithm"] = "muon"
            else:
                assert param_or_param_group.ndim == 2, param_or_param_group.ndim
                self.state[param_or_param_group]["algorithm"] = "muon"
        for param_or_param_group in adamw_params:
            if isinstance(param_or_param_group, dict):
                for p in param_or_param_group["params"]:
                    self.state[p]["algorithm"] = "adamw"
            else:
                self.state[param_or_param_group]["algorithm"] = "adamw"
        for param_or_param_group in lion_params:
            if isinstance(param_or_param_group, dict):
                for p in param_or_param_group["params"]:
                    self.state[p]["algorithm"] = "lion"
            else:
                self.state[param_or_param_group]["algorithm"] = "lion"

    def adjust_lr_to_match_adam(self, lr, param_shape):
        A, B = param_shape[:2]
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as describted in the paper
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def adjust_lr_spectral_norm(self, lr, param_shape):
        # Adjust from spectral norm 1 to RMS operator norm 1
        # https://arxiv.org/abs/2310.17813
        fan_out, fan_in = param_shape[:2]
        adjusted_lr = lr * math.sqrt(fan_out / fan_in)
        return adjusted_lr

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            ############################
            #           Muon           #
            ############################

            muon_params = [
                p for p in group["params"] if self.state[p]["algorithm"] == "muon"
            ]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            eps = group["epsilon"]

            # generate weight updates in distributed fashion
            for p in muon_params:
                # sanity check
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None

                # calc update
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                if isinstance(g, DTensor):
                    # all-gather into full unsharded matrix
                    g_local = g.full_tensor()

                    # calculate muon update
                    u_local = zeropower_via_newtonschulz5(
                        g_local, steps=group["ns_steps"], eps=eps
                    )

                    # convert back to DTensor and re-shard to original placements
                    u = DTensor.from_local(
                        u_local,
                        device_mesh=g.device_mesh,
                        placements=None,  # fully replicated
                        run_check=False,
                    ).redistribute(placements=g.placements)

                else:
                    u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"], eps=eps)

                # scale update
                if group["adjust_lr"] == "spectral_norm":
                    adjusted_lr = self.adjust_lr_spectral_norm(lr, p.shape)
                elif group["adjust_lr"] == "adam":
                    adjusted_lr = self.adjust_lr_to_match_adam(lr, p.shape)
                else:
                    raise ValueError(f"Unknown adjust_lr value: {group['adjust_lr']}")

                # apply weight decay
                p.mul_(1 - lr * weight_decay)

                # apply update
                p.add_(u, alpha=-adjusted_lr)

            ############################
            #       AdamW backup       #
            ############################

            adamw_params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["epsilon"]
            weight_decay = group["weight_decay"]

            for p in adamw_params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.mul_(1 - lr * weight_decay)
                p.add_(g, alpha=-lr / scale)

            ############################
            #       Lion backup        #
            ############################

            lion_params = [
                p for p in group["params"] if self.state[p]["algorithm"] == "lion"
            ]
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]

            for p in lion_params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]

                update = buf.lerp(g, 1 - beta1).sign_()
                buf.lerp_(g, 1 - beta2)
                p.mul_(1 - lr * weight_decay)
                p.add_(update, alpha=-lr)

        return loss


# Muon version based on Keller Jordan repo
# https://github.com/KellerJordan/modded-nanogpt
class MuonKellerJordan(Optimizer):
    """
    Muon optimizer - runs standard SGD with momentum and then orthogonalizes each 2D update.
    """

    def __init__(
        self,
        params,
        lr=0.02,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

        # Make sure no parameters are DTensor
        for group in self.param_groups:
            for p in group["params"]:
                if isinstance(p, DTensor):
                    raise NotImplementedError("DTensor parameters not supported.")

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            total_params = sum(p.numel() for p in group["params"])
            updates_flat = torch.zeros(
                total_params, device="cuda", dtype=torch.bfloat16
            )
            curr_idx = 0
            for i, p in enumerate(group["params"]):
                if i % int(os.environ["WORLD_SIZE"]) == int(os.environ["RANK"]):
                    g = p.grad
                    assert g is not None, "Gradient is None"
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if group["nesterov"]:
                        g = g.add(buf, alpha=momentum)
                    else:
                        g = buf
                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                    g *= (g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr_idx : curr_idx + p.numel()] = g.flatten()
                curr_idx += p.numel()
            dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            curr_idx = 0
            for p in group["params"]:
                g = updates_flat[curr_idx : curr_idx + p.numel()].view_as(p).type_as(p)
                p.add_(g, alpha=-lr)
                curr_idx += p.numel()
