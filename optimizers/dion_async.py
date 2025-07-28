import math
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from dataclasses import dataclass
from itertools import chain
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DeviceMesh, DTensor, Placement, Replicate, Shard
from torch.distributed.tensor import randn as dtensor_randn
from torch.optim.optimizer import Optimizer, ParamsT
from typing import Any, Dict, Generator, List, Tuple, Optional, Union

from .opt_utils import (
    AsyncTask,
    AsyncRuntime,
    to_local,
    dtensor_from_local,
    create_param_batches,
    pad_batch,
)
from .scalar_opts import (
    adamw_update_foreach,
    lion_update_foreach,
)

try:
    from torch.distributed.tensor.placement_types import _StridedShard
except ImportError:
    _StridedShard = None


@dataclass
class DionParamConfig:
    """
    Per-parameter configuration for Dion optimizer.
    """

    # Dimensions of the tensor that is sharded
    outer_shard_tensor_dim: Optional[int] = None
    inner_shard_tensor_dim: Optional[int] = None

    # Dimensions of the device mesh that the tensor is sharded over
    outer_shard_mesh_dim: Optional[int] = None
    inner_shard_mesh_dim: Optional[int] = None

    # Use transposed version of the algorithm
    is_transposed: bool = False

    # Whether to all-reduce compressed P and R instead of full gradient
    # This should always be False for 1D tensors
    compressed_all_reduce = False

    # Sharding configurations for the Q matrix
    Q_sharded_placements: Optional[Tuple[Placement]] = None
    Q_inner_unsharded_placements: Optional[Tuple[Placement]] = None


@dataclass
class DionMixedPrecisionConfig:
    """
    Configuration for mixed precision in Dion optimizer.
    None means that optimizer states will use the same dtype as each parameter.
    """

    # Momentum state for all algorithms
    momentum_dtype: Optional[torch.dtype] = None
    # Dion Q matrix
    Q_dtype: Optional[torch.dtype] = None
    # Adam variance state
    variance_dtype: Optional[torch.dtype] = None
    # TODO look into separate dtypes for communication operations


class Dion(Optimizer):
    """
    Distributed Dion Optimizer.
    https://arxiv.org/abs/2504.05295

    Args:
        params: Parameters for the optimizer.
        replicate_mesh: DeviceMesh or ProcessGroup for replicated data parallelism.
            Use DeviceMesh for hybrid sharded FSDP and ProcessGroup for DistributedDataParallel.
        outer_shard_mesh: Parameter sharding DeviceMesh, replicated during orthogonalization.
            This is the FS dimension in the paper.
        inner_shard_mesh: Parameter sharding DeviceMesh, sharded during orthogonalization.
            This is the TP dimension in the paper.
        replicate_mesh_grad_sync: If True, optimizer handles data-parallel gradient sync.
            If False, the optimizer expects gradients to be already synchronized.
        rank_fraction: r/d fraction for low-rank approximation. Used to compute the low-rank dimension.
            This may be specified per param-group to have different rank fractions.
        rank_multiple_of: Round up the low-rank dimension to a multiple of this number.
            This may be useful to ensure even sharding.
        lr: Base learning rate. For Dion, this will be scaled based on the matrix dimensions.
            For non-Dion algorithms, this is the actual learning rate and no additional scaling is done.

    Note: We assume parameters are all DTensor or all regular Tensors. All sharded tensors are assumed
    to be uniformly sharded - that is, each device along the sharding axis has identical size shards.
    The only distributed scenarios supported are:
        - DTensor + DeviceMesh: sharding with FSDP2 fully_shard() and/or TP parallelize_module().
        - regular Tensor + ProcessGroup: No sharding allowed. DDP may be used.
    FSDP1 (FullyShardedDataParallel wrapper class) is not supported.
    """

    def __init__(
        self,
        params: ParamsT,
        replicate_mesh: Optional[Union[DeviceMesh, ProcessGroup]] = None,
        outer_shard_mesh: Optional[DeviceMesh] = None,
        inner_shard_mesh: Optional[DeviceMesh] = None,
        replicate_mesh_grad_sync: bool = True,
        rank_fraction: float = 1.0,
        rank_multiple_of: int = 1,
        lr: float = 0.01,
        mu: float = 0.95,  # Momentum for Dion
        betas: Tuple[float, float] = (0.9, 0.95),  # Betas for AdamW and Lion
        weight_decay: float = 0.01,
        epsilon: float = 1e-8,
        power_iters: int = 1,  # Number of power iterations for low-rank approximation
        oversample: float = 1.25,  # For QR random sketch matrix
        mixed_precision_config: Optional[DionMixedPrecisionConfig] = None,
    ):
        # Check hyperparameters
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if mu < 0.0:
            raise ValueError(f"Invalid momentum factor (mu): {mu}")
        if len(betas) != 2 or betas[0] < 0.0 or betas[1] < 0.0:
            raise ValueError(f"Invalid betas: {betas}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if rank_fraction <= 0 or rank_fraction > 1:
            raise ValueError(f"Invalid rank fraction: {rank_fraction}")
        if rank_multiple_of <= 0:
            raise ValueError(f"Invalid rank multiple of: {rank_multiple_of}")
        if power_iters != 1:
            raise ValueError("Async Dion only supports power_iters=1")

        # Check device mesh
        if outer_shard_mesh is not None:
            if not isinstance(outer_shard_mesh, DeviceMesh):
                raise ValueError(
                    f"Outer shard mesh must be a DeviceMesh, but got {type(outer_shard_mesh)}."
                )
            if outer_shard_mesh.ndim != 1:
                raise ValueError(
                    f"Outer shard mesh must be 1D, but got {outer_shard_mesh.ndim}D. Try using a 1D sub-mesh."
                )
            if outer_shard_mesh == replicate_mesh:
                raise ValueError(
                    "Outer shard mesh must be different from replicate mesh."
                )
        if inner_shard_mesh is not None:
            if not isinstance(inner_shard_mesh, DeviceMesh):
                raise ValueError(
                    f"Inner shard mesh must be a DeviceMesh, but got {type(inner_shard_mesh)}."
                )
            if inner_shard_mesh.ndim != 1:
                raise ValueError(
                    f"Inner shard mesh must be 1D, but got {inner_shard_mesh.ndim}D. Try using a 1D sub-mesh."
                )
            if inner_shard_mesh == replicate_mesh:
                raise ValueError(
                    "Inner shard mesh must be different from replicate mesh."
                )
            if inner_shard_mesh == outer_shard_mesh:
                raise ValueError("Outer and inner shard meshes must be different.")

        # Default arguments for each param group
        defaults = dict(
            rank_fraction=rank_fraction,
            rank_multiple_of=rank_multiple_of,
            lr=lr,
            mu=mu,
            beta1=betas[0],
            beta2=betas[1],
            weight_decay=weight_decay,
            epsilon=epsilon,
            algorithm="dion",
            step=0,
        )
        super().__init__(params, defaults)
        self._oversample = oversample

        # This is intentionally not in self.state so it doesn't get checkpointed
        # State here may change upon resharding a checkpoint, so we recompute it
        self._param_config: Dict[Tensor, DionParamConfig] = {}

        self._replicate_mesh = replicate_mesh
        self._outer_shard_mesh = outer_shard_mesh
        self._inner_shard_mesh = inner_shard_mesh
        self._replicate_mesh_grad_sync = replicate_mesh_grad_sync

        # Get world size for the replicate mesh
        if isinstance(replicate_mesh, DeviceMesh):
            self._replicate_world_size = replicate_mesh.size()
        elif isinstance(replicate_mesh, ProcessGroup):
            self._replicate_world_size = dist.get_world_size(replicate_mesh)
        elif replicate_mesh is None:
            self._replicate_world_size = 1
        else:
            raise ValueError(f"Invalid replicate mesh type: {type(replicate_mesh)}.")

        # Get global ranks for outer and inner shard meshes
        if self._outer_shard_mesh is not None and self._outer_shard_mesh.size() > 1:
            self._outer_shard_ranks = dist.get_process_group_ranks(
                self._outer_shard_mesh.get_group()
            )
        else:
            self._outer_shard_ranks = None
        if self._inner_shard_mesh is not None and self._inner_shard_mesh.size() > 1:
            self._inner_shard_ranks = dist.get_process_group_ranks(
                self._inner_shard_mesh.get_group()
            )
        else:
            self._inner_shard_ranks = None

        # Mixed precision
        if mixed_precision_config is None:
            mixed_precision_config = DionMixedPrecisionConfig()
        self._mixed_precision_config = mixed_precision_config
        # TODO check what happens when loading state dict with different precision

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        dion_groups = []
        lion_groups = []
        adamw_groups = []

        for group in self.param_groups:
            # Increment step
            group["step"] += 1

            # Split parameter groups by algorithm
            algo = group["algorithm"]
            if algo == "dion":
                dion_groups.append(group)
            elif algo == "lion":
                lion_groups.append(group)
            elif algo == "adamw":
                adamw_groups.append(group)
            else:
                raise ValueError(f"Unknown algorithm: {algo}")

        # Create async tasks for each algorithm
        dion_tasks = self._create_dion_tasks(dion_groups)
        lion_tasks = self._create_lion_tasks(lion_groups)
        adamw_tasks = self._create_adamw_tasks(adamw_groups)

        all_tasks = chain(dion_tasks, lion_tasks, adamw_tasks)
        runtime = AsyncRuntime(all_tasks, max_concurrent_tasks=3)
        runtime.run()

        return loss

    def _create_dion_tasks(
        self,
        param_groups: List[dict],
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to create batches of Dion matrices and generate
        AsyncTask objects so we can process multiple batches concurrently.
        """
        # Select the appropriate Dion implementation based on sharding
        if self._inner_shard_mesh is not None:
            dion_update_func = dion_update_fsdp_tp
            batch_size = self._inner_shard_mesh.size()
            use_dtensor = True
        elif self._outer_shard_mesh is not None:
            dion_update_func = dion_update_fsdp
            batch_size = self._outer_shard_mesh.size()
            use_dtensor = True
        else:
            dion_update_func = dion_update_ddp
            batch_size = self._replicate_world_size
            use_dtensor = False  # no sharded matrices in this case

        # If replicate_mesh_grad_sync is True, the optimizer needs to all-reduce gradients
        # If replicate_mesh_grad_sync is False, gradients are already synchronized

        for group in param_groups:
            group_params = [p for p in group["params"] if p.grad is not None]
            if not group_params:
                continue

            # Wrap hyperparameters in tensors for torch.compile
            lr = torch.tensor(group["lr"])
            mu = torch.tensor(group["mu"])
            weight_decay = torch.tensor(group["weight_decay"])
            epsilon = torch.tensor(group["epsilon"])

            # Create batches of parameters
            for params in create_param_batches(group_params, batch_size=batch_size):
                gradients = [p.grad for p in params]
                states = [self._get_or_initialize_state(p, group) for p in params]
                momentums = [s["momentum"] for s in states]
                Qs = [s["Q"] for s in states]
                param_config = self._get_dion_param_config(params[0])

                if not use_dtensor:
                    params = to_local(params)
                    gradients = to_local(gradients)
                    momentums = to_local(momentums)
                    Qs = to_local(Qs)

                yield AsyncTask(
                    dion_update_func(
                        X=pad_batch(params, batch_size),
                        G=pad_batch(gradients, batch_size),
                        M=pad_batch(momentums, batch_size),
                        Q=pad_batch(Qs, batch_size),
                        lr=lr,
                        mu=mu,
                        weight_decay=weight_decay,
                        epsilon=epsilon,
                        param_config=param_config,
                        replicate_mesh=self._replicate_mesh,
                        replicate_mesh_grad_sync=self._replicate_mesh_grad_sync,
                        oversample=self._oversample,
                    )
                )

    def _create_lion_tasks(
        self,
        param_groups: List[dict],
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to generate AsyncTask objects for Lion updates.
        """
        for group in param_groups:
            # Get parameters and optimizer states
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            gradients = [p.grad for p in params]
            states = [self._get_or_initialize_state(p, group) for p in params]
            momentums = [s["momentum"] for s in states]

            # Wrap hyperparameters in tensors for torch.compile
            lr = torch.tensor(group["lr"])
            beta1 = torch.tensor(group["beta1"])
            beta2 = torch.tensor(group["beta2"])
            weight_decay = torch.tensor(group["weight_decay"])

            # If replicate_mesh_grad_sync is True, the optimizer needs to all-reduce gradients
            # If replicate_mesh_grad_sync is False, gradients are already synchronized
            all_reduce_mesh = (
                self._replicate_mesh if self._replicate_mesh_grad_sync else None
            )

            yield AsyncTask(
                lion_update_allreduce_grad(
                    X=to_local(params),
                    G=to_local(gradients),
                    M=to_local(momentums),
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=weight_decay,
                    replicate_mesh=all_reduce_mesh,
                )
            )

    def _create_adamw_tasks(
        self,
        param_groups: List[dict],
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to generate AsyncTask objects for AdamW updates.
        """
        for group in param_groups:
            # Get parameters and optimizer states
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            gradients = [p.grad for p in params]
            states = [self._get_or_initialize_state(p, group) for p in params]
            momentums = [s["momentum"] for s in states]
            variances = [s["variance"] for s in states]

            # Wrap hyperparameters in tensors for torch.compile
            lr = torch.tensor(group["lr"])
            beta1 = torch.tensor(group["beta1"])
            beta2 = torch.tensor(group["beta2"])
            weight_decay = torch.tensor(group["weight_decay"])
            epsilon = torch.tensor(group["epsilon"])
            step = torch.tensor(group["step"])

            # If replicate_mesh_grad_sync is True, the optimizer needs to all-reduce gradients
            # If replicate_mesh_grad_sync is False, gradients are already synchronized
            all_reduce_mesh = (
                self._replicate_mesh if self._replicate_mesh_grad_sync else None
            )

            yield AsyncTask(
                adamw_update_allreduce_grad(
                    X=to_local(params),
                    G=to_local(gradients),
                    M=to_local(momentums),
                    V=to_local(variances),
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=weight_decay,
                    step=step,
                    epsilon=epsilon,
                    replicate_mesh=all_reduce_mesh,
                )
            )

    def _get_or_initialize_state(self, param: Tensor, group: dict) -> dict:
        """
        Get optimizer state for the given parameter tensor,
        or lazy-initialize it if it doesn't exist.
        """
        state = self.state[param]
        algo = group["algorithm"]
        if not state:
            if algo == "dion":
                self._init_opt_state_dion(
                    param,
                    state,
                    rank_fraction=group["rank_fraction"],
                    rank_multiple_of=group["rank_multiple_of"],
                )
            elif algo == "adamw":
                self._init_opt_state_adam(param, state)
            elif algo == "lion" or algo == "clion":
                self._init_opt_state_momentum(param, state)
            else:
                raise ValueError(f"Unknown algorithm: {algo}")
        return state

    def _get_dion_param_config(self, x: Tensor) -> DionParamConfig:
        """
        Get the Dion-specific parameter configuration for a given tensor.
        If the configuration is not already initialized, it will be created.
        Lazy initialization is necessary because PyTorch allows new parameters
        to be added to the optimizer after it has been created.
        """
        if x in self._param_config:
            return self._param_config[x]

        if x.ndim > 2:
            raise NotImplementedError(
                f"Tensors with more than 2 dimensions are not supported. Got {x.ndim}D tensor."
            )

        # Check for allowed DeviceMesh and DTensor combinations
        # We only allow DTensor + DeviceMesh or regular Tensor + ProcessGroup
        using_device_mesh = (
            isinstance(self._replicate_mesh, DeviceMesh)
            or isinstance(self._outer_shard_mesh, DeviceMesh)
            or isinstance(self._inner_shard_mesh, DeviceMesh)
        )
        using_process_group = isinstance(self._replicate_mesh, ProcessGroup)
        if using_device_mesh and not isinstance(x, DTensor):
            raise ValueError("When using DeviceMesh, all parameters must be DTensor.")
        if using_process_group and isinstance(x, DTensor):
            raise ValueError(
                "When using DTensor parameters, the data parallel group must be specified by a DeviceMesh instead of ProcessGroup."
            )

        # State is initialized for both matrix and scalar parameters
        config = DionParamConfig()

        # By default, we transpose matrices so that dim0 >= dim1
        # This can change depending on sharding
        if x.ndim == 2:
            m, n = x.shape
            config.is_transposed = m < n

        # Detect sharding dimensions for DTensor
        if isinstance(x, DTensor) and x.ndim == 2:
            device_mesh = x.device_mesh
            placements = x.placements
            assert len(placements) == device_mesh.ndim

            dim_map = [None for _ in range(x.ndim)]

            for mesh_dim, placement in enumerate(placements):
                # StridedShard not allowed
                if _StridedShard is not None and isinstance(placement, _StridedShard):
                    raise NotImplementedError(
                        f"StridedShard is not supported. Ensure that FSDP and TP shard different dimensions of each matrix."
                    )

                # Skip non-sharded device mesh dimensions
                if not placement.is_shard():
                    continue
                tensor_dim = placement.dim

                # Check for double sharding on same tensor dimension
                if dim_map[tensor_dim] is not None:
                    raise ValueError(
                        f"Got double-sharded DTensor for tensor dimension {placement.dim}."
                    )
                dim_map[tensor_dim] = mesh_dim

                # Get global ranks corresponding to this mesh dimension
                mesh_dim_ranks = dist.get_process_group_ranks(
                    device_mesh.get_group(mesh_dim)
                )

                # Check if it matches the outer or inner shard ranks
                outer_sharded, inner_sharded = False, False
                if mesh_dim_ranks == self._outer_shard_ranks:
                    config.outer_shard_tensor_dim = tensor_dim
                    config.outer_shard_mesh_dim = mesh_dim
                    outer_sharded = True
                if mesh_dim_ranks == self._inner_shard_ranks:
                    config.inner_shard_tensor_dim = tensor_dim
                    config.inner_shard_mesh_dim = mesh_dim
                    inner_sharded = True

                # Check for double sharding on same mesh dimension
                if outer_sharded and inner_sharded:
                    raise ValueError(
                        "Cannot have outer and inner sharding over the same process group."
                    )

                # Check for sharding on unrecognized mesh dimension
                # Ignore edge case for single GPU "sharding" = Replicate()
                # Make sure to check that size(mesh_dim) > 1
                if (
                    device_mesh.size(mesh_dim) > 1
                    and not outer_sharded
                    and not inner_sharded
                ):
                    raise ValueError(
                        f"Got DTensor sharded on unrecognized {mesh_dim=}, which does not match outer_shard_mesh or inner_shard_mesh."
                    )

            # Set transpose so that orthogonalization happens over the inner sharding dimension
            # Standard Dion orthogonalizes over tensor dimension 0
            if config.inner_shard_tensor_dim == 0 or config.outer_shard_tensor_dim == 1:
                config.is_transposed = False
            # Transposed Dion orthogonalizes over tensor dimension 1
            if config.outer_shard_tensor_dim == 0 or config.inner_shard_tensor_dim == 1:
                config.is_transposed = True

        self._param_config[x] = config
        return config

    def _init_opt_state_momentum(self, param: Tensor, state: Dict[str, Any]):
        # Create the momentum buffer
        # If param is DTensor, this will also be a DTensor
        state["momentum"] = torch.zeros_like(
            param, dtype=self._mixed_precision_config.momentum_dtype
        )

    def _init_opt_state_adam(self, param: Tensor, state: Dict[str, Any]):
        self._init_opt_state_momentum(param, state)
        state["variance"] = torch.zeros_like(
            param, dtype=self._mixed_precision_config.variance_dtype
        )

    def _init_opt_state_dion(
        self,
        param: Tensor,
        state: Dict[str, Any],
        rank_fraction: float,
        rank_multiple_of: int,
    ):
        """
        Initialize the optimizer state for Dion.
        This includes the momentum buffer and the Q matrix.

        The low-rank factor `r` is computed as `rank_fraction` * min(m, n),
        and rounded up to the next multiple of `rank_multiple_of`.
        """
        if param.ndim != 2:
            raise ValueError(
                f"Expected Dion parameters to be 2D matrix, but got {param.ndim}D. "
                f"For scalar parameters, set 'algorithm' to 'lion' or 'adamw' when creating param group."
            )

        param_config = self._get_dion_param_config(param)
        self._init_opt_state_momentum(param, state)

        # Compute the low-rank factor r
        m, n = param.shape
        r = rank_fraction * min(m, n)
        r = rank_multiple_of * math.ceil(r / rank_multiple_of)
        r = min(r, m, n)
        Q_shape = (m, r) if param_config.is_transposed else (n, r)

        # Set compressed_all_reduce based on if it saves communication cost
        # Otherwise we will all-reduce the gradient matrix instead
        if rank_fraction < 1 and (m + n) * r < m * n:
            param_config.compressed_all_reduce = True

        # Get dtype for Q
        if self._mixed_precision_config.Q_dtype is not None:
            Q_dtype = self._mixed_precision_config.Q_dtype
        else:
            Q_dtype = param.dtype

        if isinstance(param, DTensor):
            # Directly construct Q as DTensor
            # Shard(0) on outer sharding mesh and Shard(1) on inner sharding mesh
            placements = [Replicate() for _ in range(param.device_mesh.ndim)]
            if param_config.outer_shard_mesh_dim is not None:
                placements[param_config.outer_shard_mesh_dim] = Shard(0)
            if param_config.inner_shard_mesh_dim is not None:
                placements[param_config.inner_shard_mesh_dim] = Shard(1)
            param_config.Q_sharded_placements = tuple(placements)

            # Q is unsharded along the inner sharding dimension only
            if param_config.inner_shard_mesh_dim is not None:
                placements[param_config.inner_shard_mesh_dim] = Replicate()
                param_config.Q_inner_unsharded_placements = tuple(placements)
            else:
                # No inner sharding, so placements are the same as Q_sharded_placements
                param_config.Q_inner_unsharded_placements = None

            # DTensor RNG should automatically produce identical results across DP replicas
            Q = dtensor_randn(
                Q_shape,
                device_mesh=param.device_mesh,
                dtype=Q_dtype,
                placements=param_config.Q_sharded_placements,
            )

        else:
            # Make sure all DP ranks have the same Q
            Q = torch.randn(Q_shape, device=param.device, dtype=Q_dtype)
            self._replicate_mesh_broadcast(Q)

        state["Q"] = Q

    def _replicate_mesh_broadcast(self, tensor: Tensor):
        """
        Broadcast a tensor from rank 0 over the replicated data-parallel world.
        Tensor is modified in place.
        """
        if self._replicate_mesh is None:
            # No data parallelism used, do nothing
            pass
        elif isinstance(self._replicate_mesh, DeviceMesh):
            for group in self._replicate_mesh.get_all_groups():
                dist.broadcast(tensor, group=group, group_src=0)
        elif isinstance(self._replicate_mesh, ProcessGroup):
            dist.broadcast(tensor, group=self._replicate_mesh, group_src=0)
        else:
            raise ValueError(
                "Data parallel mesh must be either a DeviceMesh or ProcessGroup."
            )


def dion_update_ddp(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    M: List[Tensor],  # Momentum buffer (modified in place)
    Q: List[Tensor],  # Q matrix for power iteration (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    mu: Tensor,  # Momentum factor (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    epsilon: float,
    param_config: DionParamConfig,  # shared for all params in batch
    replicate_mesh: Union[DeviceMesh, ProcessGroup, None] = None,
    replicate_mesh_grad_sync: bool = True,
    oversample: float = 1.25,
) -> Generator[None, None, None]:
    """
    Dion optimizer algorithm. Async batched implementation for DDP.
    This function does not support sharded matrices.

    Batch size should equal the DDP world size. Each device will
    orthogonalize one full matrix in the batch.
    """
    assert param_config.outer_shard_mesh_dim is None
    assert param_config.outer_shard_tensor_dim is None
    assert param_config.inner_shard_mesh_dim is None
    assert param_config.inner_shard_tensor_dim is None

    # Get rank and world size
    if isinstance(replicate_mesh, DeviceMesh):
        world_size = replicate_mesh.size()
        device_rank = replicate_mesh.get_rank()
    elif isinstance(replicate_mesh, ProcessGroup):
        world_size = dist.get_world_size(replicate_mesh)
        device_rank = dist.get_rank(replicate_mesh)
    else:
        world_size = 1
        device_rank = 0
    assert (
        len(X) == world_size
    ), f"Batch size {len(X)} must match DDP world size {world_size}."

    # Special case for when compressed_all_reduce is False
    # Here we all-reduce the gradient G instead of compressed P and R matrices
    # This is more efficient when rank_fraction == 1
    if (
        replicate_mesh_grad_sync
        and not param_config.compressed_all_reduce
        and replicate_mesh is not None
    ):
        G = all_reduce_gradient(G, replicate_mesh, return_dtensor=False)
        yield

    # Match dtype of Q and M
    # Stack Q into 3D tensor of shape (batch_size, n, r)
    M_dtype = M[0].dtype
    Q_init_dtype = Q[0].dtype
    Q_batch = torch.stack([q.to(M_dtype) for q in Q])

    # Add new gradient to momentum
    # Stack into 3D tensor of shape (batch_size, m, n) (when non-transposed)
    torch._foreach_add_(M, G)
    if param_config.is_transposed:
        M_batch = torch.stack([m.T for m in M])
    else:
        M_batch = torch.stack(M)

    # Compute low-rank approximation of M = P @ Q^T
    # M, Q, P, R should all have the same dtype
    # P_batch shape is (batch_size, m, r)
    # Q_batch shape is (batch_size, n, r)
    P_batch = M_batch @ Q_batch

    if (
        replicate_mesh_grad_sync
        and param_config.compressed_all_reduce
        and replicate_mesh is not None
    ):
        # Synchronize P across all DDP ranks by reduce-scatter
        # Each rank will orthogonalize one full matrix in the batch
        P_single = funcol.reduce_scatter_tensor(
            P_batch,
            reduceOp="avg",
            scatter_dim=0,  # dim 0 = batch
            group=replicate_mesh,
        )
        yield
    else:
        # Gradients are already synchronized, P_batch is identical across DDP world
        # We can just take one matrix of the batch
        P_single = P_batch[device_rank : device_rank + 1]

    # Orthogonalize P_single using QR decomposition
    # Standard QR is faster if matrix is square or wide
    _, m, r = P_single.shape
    if m <= r:
        P_single, _ = torch.linalg.qr(P_single.to(dtype=torch.float32))
        P_single = P_single.to(dtype=M_dtype).contiguous()

    # Randomized Cholesky QR
    else:
        # Orthogonalize P_single using random sketch QR
        S_single = generate_random_sketch_matrix(P_single, oversample)
        SP_single = S_single @ P_single
        _, R_single = torch.linalg.qr(SP_single.to(dtype=torch.float32), mode="r")
        P_single = torch.linalg.solve_triangular(
            R_single, P_single.to(dtype=torch.float32), upper=True, left=False
        )

        # Apply Cholesky QR to better orthogonalize
        PP_single = P_single.mT @ P_single
        R_single, _ = torch.linalg.cholesky_ex(PP_single, upper=True)
        P_single = torch.linalg.solve_triangular(
            R_single, P_single, upper=True, left=False
        )
        P_single = P_single.to(dtype=M_dtype).contiguous()

    # All gather P_batch from the per-device single matrices
    if replicate_mesh is not None:
        P_batch = funcol.all_gather_tensor(
            P_single,
            gather_dim=0,  # dim 0 = batch
            group=replicate_mesh,
        )
        yield
    else:
        assert world_size == 1
        P_batch = P_single  # batch size is 1

    # M_batch shape is (batch_size, m, n)
    # P_batch shape is (batch_size, m, r)
    # R_batch shape is (batch_size, n, r)
    # Note that this is a different R than the triangular matrix from orthogonalization
    R_batch = M_batch.mT @ P_batch

    # Synchronize R across all DDP ranks by all-reduce
    if (
        replicate_mesh_grad_sync
        and param_config.compressed_all_reduce
        and replicate_mesh is not None
    ):
        R_batch = funcol.all_reduce(
            R_batch,
            reduceOp="avg",
            group=replicate_mesh,
        )
        yield

    # NaN check
    P_batch, R_batch = fix_all_zero_or_nan(P_batch, R_batch, Q_batch, M_batch)

    # Error feedback update
    # M = M - (1 - mu) * (P @ R.T)
    foreach_baddbmm_(
        M, P_batch, R_batch, alpha=-(1 - mu), transpose=param_config.is_transposed
    )

    # Column normalize R to get new Q
    Q_batch = R_batch / (R_batch.norm(dim=-2, keepdim=True) + epsilon)

    # Apply weight decay
    torch._foreach_mul_(X, 1 - lr * weight_decay)

    # Compute update scale factor
    fan_out = X[0].size(0)
    fan_in = X[0].size(1)
    scaled_lr = ((fan_out / fan_in) ** 0.5) * lr

    # Apply weight update
    # X = X - scaled_lr * P @ Q.T
    foreach_baddbmm_(
        X, P_batch, Q_batch, alpha=-scaled_lr, transpose=param_config.is_transposed
    )

    # Update Q in place
    Q_new = Q_batch.to(Q_init_dtype).unbind(dim=0)
    torch._foreach_copy_(Q, Q_new)


def dion_update_fsdp(
    X: List[DTensor],  # Model weights (modified in place)
    G: List[DTensor],  # Gradient
    M: List[DTensor],  # Momentum buffer (modified in place)
    Q: List[DTensor],  # Q matrix for power iteration (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    mu: Tensor,  # Momentum factor (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    epsilon: float,
    param_config: DionParamConfig,  # shared for all params in batch
    replicate_mesh: Optional[DeviceMesh] = None,
    replicate_mesh_grad_sync: bool = True,
    oversample: float = 1.25,
) -> Generator[None, None, None]:
    """
    Dion optimizer algorithm. Async batched implementation for FSDP2 sharding.
    This function only supports sharding over the outer shard mesh.

    Batch size should equal the outer shard mesh size. Each device along the
    outer shard mesh dimension will orthogonalize one full matrix in the batch.
    """
    assert param_config.inner_shard_mesh_dim is None
    assert param_config.inner_shard_tensor_dim is None

    # Special case for when compressed_all_reduce is False
    # Here we all-reduce the gradient G instead of compressed P and R matrices
    # This is more efficient when rank_fraction == 1
    if (
        replicate_mesh_grad_sync
        and not param_config.compressed_all_reduce
        and replicate_mesh is not None
    ):
        G_local = all_reduce_gradient(G, replicate_mesh, return_dtensor=False)
        yield
        G = dtensor_from_local(G_local, ref=G[0])

    # Match dtype of Q and M
    # Stack Q into 3D tensor of shape (batch_size, n/outer, r)
    M_dtype = M[0].dtype
    Q_init_dtype = Q[0].dtype
    Q_batch = torch.stack([q.to(M_dtype) for q in Q])

    # Add new gradient to momentum
    # Stack into 3D tensor of shape (batch_size, m, n/outer) (when non-transposed)
    torch._foreach_add_(M, G)
    if param_config.is_transposed:
        M_batch = torch.stack([m.T for m in M])
    else:
        M_batch = torch.stack(M)

    # Compute low-rank approximation of M = P @ Q^T
    # M, Q, P, R should all have the same dtype
    # P_batch shape is (batch_size, m, r)
    # Q_batch shape is (batch_size, n/outer, r)
    P_batch = M_batch @ Q_batch

    # Reduce scatter P to get a single full matrix of the batch
    P_batch_placements = [
        Replicate() if p.is_partial() else p for p in P_batch.placements
    ]
    P_single_placements = [
        Shard(0) if p.is_partial() else p for p in P_batch.placements
    ]
    # If compressed_all_reduce is True, also average over replicate mesh
    compressed_all_reduce = (
        replicate_mesh_grad_sync and param_config.compressed_all_reduce
    )
    P_local = reduce_scatter_partial_tensor(
        P_batch,
        partial_mesh_dim=param_config.outer_shard_mesh_dim,
        scatter_dim=0,  # dim 0 = batch
        replicate_mesh=replicate_mesh if compressed_all_reduce else None,
        return_dtensor=False,
    )
    yield
    P_single = DTensor.from_local(
        P_local,
        device_mesh=P_batch.device_mesh,
        placements=P_single_placements,
    )

    # Orthogonalize P_single using QR decomposition
    # Standard QR is faster if matrix is square or wide
    _, m, r = P_single.shape  # this is unsharded tensor shape
    if m <= r:
        P_single = qr_dtensor(P_single, return_R=False, out_dtype=M_dtype)

    # Randomized Cholesky QR
    else:
        # Orthogonalize P_single using random sketch QR
        S_single = generate_random_sketch_matrix(P_single, oversample)
        SP_single = S_single @ P_single
        R_single = qr_dtensor(SP_single, return_R=True, out_dtype=torch.float32)
        P_single = solve_triangular_dtensor(P_single, R_single, out_dtype=torch.float32)

        # Apply Cholesky QR to better orthogonalize
        PP_single = P_single.mT @ P_single
        R_single = cholesky_dtensor(PP_single, upper=True, out_dtype=torch.float32)
        P_single = solve_triangular_dtensor(P_single, R_single, out_dtype=M_dtype)

    # All gather P_batch from the per-device single matrices
    P_batch = P_single.redistribute(placements=P_batch_placements, async_op=True)
    yield

    # M_batch shape is (batch_size, m, n/outer)
    # P_batch shape is (batch_size, m, r)
    # R_batch shape is (batch_size, n/outer, r)
    # Note that this is a different R than the triangular matrix from orthogonalization
    R_batch = M_batch.mT @ P_batch

    # All reduce R to average over replicate mesh
    if compressed_all_reduce and replicate_mesh is not None:
        # The contracting dimension of R = M.mT @ P isn't sharded
        # There should not be any partial placements
        assert not any(p.is_partial() for p in R_batch.placements)
        R_local = all_reduce_partial_tensor(
            R_batch,
            partial_mesh_dim=None,
            replicate_mesh=replicate_mesh,
            return_dtensor=False,
        )
        yield
        R_batch = dtensor_from_local(R_local, ref=R_batch)

    # NaN check
    P_batch, R_batch = fix_all_zero_or_nan(P_batch, R_batch, Q_batch, M_batch)

    # Error feedback update
    # M = M - (1 - mu) * (P @ R.T)
    foreach_baddbmm_(
        M, P_batch, R_batch, alpha=-(1 - mu), transpose=param_config.is_transposed
    )

    # Column normalize R to get new Q
    if param_config.outer_shard_mesh_dim is not None:
        assert isinstance(R_batch, DTensor)
        assert R_batch.placements[param_config.outer_shard_mesh_dim].is_shard()

        # Compute per-shard squared norm and sum across shards
        R_local = R_batch.to_local()
        R_norm_sq = R_local.square().sum(dim=-2, keepdim=True)
        R_norm_sq = funcol.all_reduce(
            R_norm_sq,
            reduceOp="sum",
            group=(R_batch.device_mesh, param_config.outer_shard_mesh_dim),
        )
        yield

        # Normalize R and convert back to DTensor
        Q_batch = dtensor_from_local(
            R_local / (R_norm_sq.sqrt() + epsilon), ref=R_batch
        )

    else:
        Q_batch = R_batch / (R_batch.norm(dim=-2, keepdim=True) + epsilon)

    # Apply weight decay
    torch._foreach_mul_(X, 1 - lr * weight_decay)

    # Compute update scale factor
    fan_out = X[0].size(0)
    fan_in = X[0].size(1)
    scaled_lr = ((fan_out / fan_in) ** 0.5) * lr

    # Apply weight update
    # X = X - scaled_lr * P @ Q.T
    foreach_baddbmm_(
        X, P_batch, Q_batch, alpha=-scaled_lr, transpose=param_config.is_transposed
    )

    # Update Q in place
    Q_new = Q_batch.to(Q_init_dtype).unbind(dim=0)
    torch._foreach_copy_(Q, Q_new)


def dion_update_fsdp_tp(
    X: List[DTensor],  # Model weights (modified in place)
    G: List[DTensor],  # Gradient
    M: List[DTensor],  # Momentum buffer (modified in place)
    Q: List[DTensor],  # Q matrix for power iteration (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    mu: Tensor,  # Momentum factor (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    epsilon: float,
    param_config: DionParamConfig,  # shared for all params in batch
    replicate_mesh: Optional[DeviceMesh] = None,
    replicate_mesh_grad_sync: bool = True,
    oversample: float = 1.25,
) -> Generator[None, None, None]:
    """
    Dion optimizer algorithm. Async batched implementation for combined FSDP2 + TP.
    This function supports sharding over both outer and inner shard meshes.

    Batch size should equal the inner shard mesh size. The full matrix will not be
    unsharded for orthogonalization. Each device along the inner shard mesh dimension
    will compute low-rank QR and Cholesky decompositions for one matrix in the batch.
    """
    # Special case for when compressed_all_reduce is False
    # Here we all-reduce the gradient G instead of compressed P and R matrices
    # This is more efficient when rank_fraction == 1
    if (
        replicate_mesh_grad_sync
        and not param_config.compressed_all_reduce
        and replicate_mesh is not None
    ):
        G_local = all_reduce_gradient(G, replicate_mesh, return_dtensor=False)
        yield
        G = dtensor_from_local(G_local, ref=G[0])

    # Unshard Q along the inner sharding dimension
    if param_config.Q_inner_unsharded_placements is not None:
        # Sharded Q has shape (n/outer, r/inner)
        # Unsharded Q has shape (n/outer, r)
        Q = [
            q.redistribute(
                placements=param_config.Q_inner_unsharded_placements,
                async_op=True,
            )
            for q in Q
        ]
        yield

    # Match dtype of Q and M
    # Stack Q into 3D tensor of shape (batch_size, n/outer, r)
    M_dtype = M[0].dtype
    Q_init_dtype = Q[0].dtype
    Q_batch = torch.stack([q.to(M_dtype) for q in Q])

    # Add new gradient to momentum and stack into 3D batch
    # M_batch shape is (batch_size, m/inner, n/outer) (when non-transposed)
    torch._foreach_add_(M, G)
    if param_config.is_transposed:
        M_batch = torch.stack([m.T for m in M])
    else:
        M_batch = torch.stack(M)

    # Compute low-rank approximation of M = P @ Q^T
    # M, Q, P, R should all have the same dtype
    # P_batch shape is (batch_size, m/inner, r)
    # Q_batch shape is (batch_size, n/outer, r)
    P_batch = M_batch @ Q_batch

    # All reduce P to sum over shard mesh
    # If compressed_all_reduce is True, also average over replicate mesh
    compressed_all_reduce = (
        replicate_mesh_grad_sync and param_config.compressed_all_reduce
    )
    P_placements = [Replicate() if p.is_partial() else p for p in P_batch.placements]
    P_local = all_reduce_partial_tensor(
        P_batch,
        partial_mesh_dim=param_config.outer_shard_mesh_dim,
        replicate_mesh=replicate_mesh if compressed_all_reduce else None,
        return_dtensor=False,
    )
    yield
    P_batch = DTensor.from_local(
        P_local,
        device_mesh=P_batch.device_mesh,
        placements=P_placements,
    )

    # Orthogonalize P_batch
    # Each GPU along inner shard mesh gets one full matrix of the batch
    # Shard along dim 0 = batch
    _, m, r = P_batch.shape  # this is unsharded tensor shape
    batch_sharded_placements = [Replicate() for _ in P_placements]
    if param_config.inner_shard_mesh_dim is not None:
        batch_sharded_placements[param_config.inner_shard_mesh_dim] = Shard(0)

    # Standard QR is faster if matrix is square or wide
    if m <= r:
        P_single = P_batch.redistribute(
            placements=batch_sharded_placements,
            async_op=True,
        )  # this should do all-to-all
        yield

        # Compute Q matrix of QR decomposition
        P_single = qr_dtensor(P_single, return_R=False, out_dtype=M_dtype)

        P_batch = P_single.redistribute(
            placements=P_placements,
            async_op=True,
        )  # this should do all-to-all
        yield

    # Randomized Cholesky QR
    else:
        # Generate the random sketch matrix S
        # P_batch shape is (batch_size, m/inner, r)
        # S_batch shape is (batch_size, k, m/inner)
        S_batch = generate_random_sketch_matrix(
            P_batch, oversample, shard_mesh_dim=param_config.inner_shard_mesh_dim
        )

        # SP_batch shape is (batch_size, k, r)
        # SP_single shape is (1, k, r) after redistribute
        SP_batch: DTensor = S_batch @ P_batch
        SP_single = SP_batch.redistribute(
            placements=batch_sharded_placements,
            async_op=True,
        )  # this should do reduce-scatter
        yield

        # Compute R matrix using QR decomposition
        R_single = qr_dtensor(SP_single, return_R=True, out_dtype=torch.float32)

        R_batch = R_single.redistribute(
            placements=[Replicate() for _ in P_placements],
            async_op=True,
        )  # this should do all-gather
        yield

        # Solve for orthogonalized P_batch
        P_batch = solve_triangular_dtensor(P_batch, R_batch, out_dtype=torch.float32)

        # Apply Cholesky QR to better orthogonalize P_batch
        # PP_batch shape is (batch_size, r, r)
        PP_batch: DTensor = P_batch.mT @ P_batch
        PP_single = PP_batch.redistribute(
            placements=batch_sharded_placements,
            async_op=True,
        )  # this should do reduce-scatter
        yield

        # Compute R matrix using Cholesky decomposition
        R_single = cholesky_dtensor(PP_single, upper=True, out_dtype=torch.float32)

        R_batch = R_single.redistribute(
            placements=[Replicate() for _ in P_placements],
            async_op=True,
        )  # this should do all-gather
        yield

        # Solve for orthogonalized P_batch
        P_batch = solve_triangular_dtensor(P_batch, R_batch, out_dtype=M_dtype)

    # M_batch shape is (batch_size, m/inner, n/outer)
    # P_batch shape is (batch_size, m/inner, r)
    # R_batch shape is (batch_size, n/outer, r)
    # Note that this is a different R than the triangular matrix from orthogonalization
    R_batch = M_batch.mT @ P_batch

    # All reduce R to sum over shard mesh
    # If compressed_all_reduce is True, also average over replicate mesh
    R_placements = [Replicate() if p.is_partial() else p for p in R_batch.placements]
    R_local = all_reduce_partial_tensor(
        R_batch,
        partial_mesh_dim=param_config.inner_shard_mesh_dim,
        replicate_mesh=replicate_mesh if compressed_all_reduce else None,
        return_dtensor=False,
    )
    yield
    R_batch = DTensor.from_local(
        R_local,
        device_mesh=R_batch.device_mesh,
        placements=R_placements,
    )

    # NaN check
    P_batch, R_batch = fix_all_zero_or_nan(P_batch, R_batch, Q_batch, M_batch)

    # Error feedback update
    # M = M - (1 - mu) * (P @ R.T)
    foreach_baddbmm_(
        M, P_batch, R_batch, alpha=-(1 - mu), transpose=param_config.is_transposed
    )

    # Column normalize R to get new Q
    if param_config.outer_shard_mesh_dim is not None:
        assert isinstance(R_batch, DTensor)
        assert R_batch.placements[param_config.outer_shard_mesh_dim].is_shard()

        # Compute per-shard squared norm and sum across shards
        R_local = R_batch.to_local()
        R_norm_sq = R_local.square().sum(dim=-2, keepdim=True)
        R_norm_sq = funcol.all_reduce(
            R_norm_sq,
            reduceOp="sum",
            group=(R_batch.device_mesh, param_config.outer_shard_mesh_dim),
        )
        yield

        # Normalize R and convert back to DTensor
        Q_batch = dtensor_from_local(
            R_local / (R_norm_sq.sqrt() + epsilon), ref=R_batch
        )

    else:
        Q_batch = R_batch / (R_batch.norm(dim=-2, keepdim=True) + epsilon)

    # Apply weight decay
    torch._foreach_mul_(X, 1 - lr * weight_decay)

    # Compute update scale factor
    fan_out = X[0].size(0)
    fan_in = X[0].size(1)
    scaled_lr = ((fan_out / fan_in) ** 0.5) * lr

    # Apply weight update
    # X = X - scaled_lr * P @ Q.T
    foreach_baddbmm_(
        X, P_batch, Q_batch, alpha=-scaled_lr, transpose=param_config.is_transposed
    )

    # Re-shard and update Q in place
    Q_new = Q_batch.to(Q_init_dtype).unbind(dim=0)
    if param_config.Q_sharded_placements is not None:
        # Sharded Q has shape (n/outer, r/inner)
        # Unsharded Q has shape (n/outer, r)
        Q_new = [
            q.redistribute(placements=param_config.Q_sharded_placements) for q in Q_new
        ]
    torch._foreach_copy_(Q, Q_new)


def all_reduce_gradient(
    G: List[Tensor],
    replicate_mesh: Optional[DeviceMesh] = None,
    return_dtensor: bool = True,
) -> List[Tensor]:
    """
    All-reduce a list of gradients across replicated data-parallel ranks.
    """
    if replicate_mesh is None:
        # No data parallelism, return gradients unmodified
        return G

    G_local = funcol.all_reduce_coalesced(
        to_local(G),
        reduceOp="avg",
        group=replicate_mesh,
    )
    if return_dtensor:
        return dtensor_from_local(G_local, ref=G[0])
    else:
        return G_local


def all_reduce_partial_tensor(
    X: Tensor,
    partial_mesh_dim: Optional[int] = None,
    replicate_mesh: Optional[DeviceMesh] = None,
    return_dtensor: bool = True,
) -> Tensor:
    """
    All-reduce the result of sharded matrix multiplication.

    If partial_mesh_dim is specified, we assume X is a DTensor
    with Partial() placement on that mesh dimension.

    If replicate_mesh is specified, we perform an additional all-reduce
    to average the results across all replicated data-parallel ranks.
    """
    X_local = X.to_local() if isinstance(X, DTensor) else X

    if partial_mesh_dim is not None:
        assert isinstance(
            X, DTensor
        ), "Input must be DTensor when partial_mesh_dim is specified."
        assert X.placements[partial_mesh_dim].is_partial(), (
            f"Expected DTensor to be Partial() on mesh dimension {partial_mesh_dim}, "
            f"but got placements: {X.placements}"
        )
        X_local = funcol.all_reduce(
            X_local,
            reduceOp="sum",
            group=(X.device_mesh, partial_mesh_dim),
        )

    if replicate_mesh is not None:
        X_local = funcol.all_reduce(
            X_local,
            reduceOp="avg",
            group=replicate_mesh,
        )

    if return_dtensor and isinstance(X, DTensor):
        new_placements = list(X.placements)
        if partial_mesh_dim is not None:
            new_placements[partial_mesh_dim] = Replicate()
        return DTensor.from_local(
            X_local,
            device_mesh=X.device_mesh,
            placements=new_placements,
        )
    else:
        return X_local


def reduce_scatter_partial_tensor(
    X: Tensor,
    partial_mesh_dim: Optional[int] = None,
    scatter_dim: int = 0,
    replicate_mesh: Optional[DeviceMesh] = None,
    return_dtensor: bool = True,
) -> Tensor:
    """
    Reduce-scatter the result of sharded matrix multiplication.

    If partial_mesh_dim is specified, we assume X is a DTensor
    with Partial() placement on that mesh dimension.
    The result will be sharded along the tensor dimension `scatter_dim`.

    If replicate_mesh is specified, we perform an additional all-reduce
    to average the results across all replicated data-parallel ranks.
    """
    X_local = X.to_local() if isinstance(X, DTensor) else X

    if partial_mesh_dim is not None:
        assert isinstance(
            X, DTensor
        ), "Input must be DTensor when partial_mesh_dim is specified."
        assert X.placements[partial_mesh_dim].is_partial(), (
            f"Expected DTensor to be Partial() on mesh dimension {partial_mesh_dim}, "
            f"but got placements: {X.placements}"
        )
        X_local = funcol.reduce_scatter_tensor(
            X_local,
            reduceOp="sum",
            scatter_dim=scatter_dim,
            group=(X.device_mesh, partial_mesh_dim),
        )

    if replicate_mesh is not None:
        X_local = funcol.all_reduce(
            X_local,
            reduceOp="avg",
            group=replicate_mesh,
        )

    if return_dtensor and isinstance(X, DTensor):
        new_placements = list(X.placements)
        if partial_mesh_dim is not None:
            new_placements[partial_mesh_dim] = Shard(scatter_dim)
        return DTensor.from_local(
            X_local,
            device_mesh=X.device_mesh,
            placements=new_placements,
        )
    else:
        return X_local


def generate_random_sketch_matrix(
    P: Tensor,
    oversample: float = 1.25,
    shard_mesh_dim: Optional[int] = None,
) -> Tensor:
    """
    Generate a random sketch matrix S for low-rank approximation.
    P is the input tensor with shape (batch_size, m, r).
    The sketch matrix S will have shape (batch_size, k, m),
    where k = round(oversample * r) to the next multiple of 128.
    """
    assert P.ndim == 3, "P must be a 3D batch of matrices"

    batch_size, m, r = P.shape
    k = math.ceil(oversample * r / 128.0) * 128
    std = math.sqrt(1.0 / k)

    if isinstance(P, DTensor):
        S_placements = list(P.placements)
        if shard_mesh_dim is not None:
            # Shard along tensor dimension -1 (size m dimension)
            S_placements[shard_mesh_dim] = Shard(-1)

        S = dtensor_randn(
            (batch_size, k, m),
            device_mesh=P.device_mesh,
            dtype=P.dtype,
            placements=S_placements,
        )
        return S * std

    else:
        # Regular tensor case
        if shard_mesh_dim is not None:
            raise ValueError("Must use DTensor for sharded random sketch.")
        return torch.empty((batch_size, k, m), device=P.device, dtype=P.dtype).normal_(
            std=std
        )


def qr_dtensor(
    A: DTensor,
    return_R: bool = False,
    out_dtype: Optional[torch.dtype] = None,
) -> DTensor:
    """
    Perform QR decomposition on a DTensor.
    If return_R is True, return the R matrix. Otherwise, return the Q matrix.
    If out_dtype is specified, the output will be cast to that dtype.
    Otherwise, the output will be float32.
    """
    assert isinstance(A, DTensor), "Input must be DTensor"
    for matrix_dim in (-2, -1):
        assert A.placements[matrix_dim].is_replicate(), "Must unshard matrix before QR"

    # Torch QR requires float32 precision
    A_local = to_local(A).to(dtype=torch.float32)
    mode = "r" if return_R else "reduced"
    Q_local, R_local = torch.linalg.qr(A_local, mode=mode)

    if return_R:
        return dtensor_from_local(
            R_local if out_dtype is None else R_local.to(out_dtype),
            ref=A,
        )
    else:
        return dtensor_from_local(
            Q_local if out_dtype is None else Q_local.to(out_dtype),
            ref=A,
        )


def cholesky_dtensor(
    A: DTensor,
    upper: bool = True,
    out_dtype: Optional[torch.dtype] = None,
) -> DTensor:
    """
    Perform Cholesky decomposition on a DTensor.
    """
    assert isinstance(A, DTensor), "Input must be DTensor"
    for matrix_dim in (-2, -1):
        assert A.placements[
            matrix_dim
        ].is_replicate(), "Must unshard matrix before Cholesky"

    # Torch Cholesky requires float32 precision
    A_local = to_local(A).to(dtype=torch.float32)
    L_local, _ = torch.linalg.cholesky_ex(A_local, upper=upper)

    return dtensor_from_local(
        L_local if out_dtype is None else L_local.to(out_dtype),
        ref=A,
    )


def solve_triangular_dtensor(
    A: DTensor,
    R: DTensor,
    out_dtype: Optional[torch.dtype] = None,
) -> DTensor:
    """
    Solve the linear system Q @ R = A for Q, where R is a triangular matrix.
    """
    assert isinstance(A, DTensor)
    assert isinstance(R, DTensor)
    assert A.placements[-1].is_replicate(), "Last dimension of A cannot be sharded"
    for matrix_dim in (-2, -1):
        assert R.placements[matrix_dim].is_replicate(), "R matrix cannot be sharded"

    # Convert to local tensors and solve
    A_local = to_local(A).to(dtype=torch.float32)
    R_local = to_local(R).to(dtype=torch.float32)
    Q_local = torch.linalg.solve_triangular(R_local, A_local, upper=True, left=False)

    return dtensor_from_local(
        Q_local if out_dtype is None else Q_local.to(out_dtype),
        ref=A,
    )


def fix_all_zero_or_nan(
    P: Tensor, Q: Tensor, Q_init: Tensor, B: Tensor
) -> Tuple[Tensor, Tensor]:
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
    To avoid additional communication, we handle sharded tensors independently.
    """
    B_local = to_local(B)
    is_all_zero = (B_local == 0).all(dim=(-2, -1), keepdim=True)
    not_all_zero = ~is_all_zero
    P_local = to_local(P).nan_to_num() * not_all_zero
    Q_local = to_local(Q).nan_to_num() * not_all_zero + to_local(Q_init) * is_all_zero
    P = dtensor_from_local(P_local, ref=P)
    Q = dtensor_from_local(Q_local, ref=Q)
    return P, Q


def foreach_baddbmm_(
    X: List[Tensor],  # List of 2D matrices (modified in place)
    A: Tensor,  # 3D batch of matrices
    B: Tensor,  # 3D batch of matrices
    alpha: float = 1.0,
    transpose: bool = False,
):
    """
    Perform batch matrix multiplication and in-place addition.
    This is basically a foreach version of torch.baddbmm().

    If transpose is False, we compute X[i] += alpha * (A @ B.mT)[i].
    If transpose is True, we compute X[i] += alpha * (B @ A.mT)[i].
    """
    assert len(X) == A.size(0), "X must have the same batch size as A"
    assert A.size(0) == B.size(0), "A and B must have the same batch size"

    if not transpose:
        update = A @ B.mT
    else:
        update = B @ A.mT

    update = update.unbind(dim=0)  # Split batch into list of tensors
    torch._foreach_add_(X, update, alpha=alpha)


def adamw_update_allreduce_grad(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    M: List[Tensor],  # Momentum buffer (modified in place)
    V: List[Tensor],  # Variance buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    step: int,
    epsilon: float,
    replicate_mesh: Optional[DeviceMesh] = None,
) -> Generator[None, None, None]:
    """
    AdamW optimizer algorithm with gradient all-reduce.
    """
    G_local = all_reduce_gradient(G, replicate_mesh, return_dtensor=False)
    yield
    G = dtensor_from_local(G_local, ref=G[0])
    adamw_update_foreach(X, G, M, V, lr, beta1, beta2, weight_decay, step, epsilon)


def lion_update_allreduce_grad(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    M: List[Tensor],  # Momentum buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    replicate_mesh: Optional[DeviceMesh] = None,
) -> Generator[None, None, None]:
    """
    Lion optimizer algorithm with gradient all-reduce.
    """
    G_local = all_reduce_gradient(G, replicate_mesh, return_dtensor=False)
    yield
    G = dtensor_from_local(G_local, ref=G[0])
    lion_update_foreach(X, G, M, lr, beta1, beta2, weight_decay)
