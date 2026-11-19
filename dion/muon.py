import math
import torch
import weakref
import torch.distributed as dist
from itertools import chain
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DeviceMesh, DTensor, Shard
from torch.distributed.tensor.placement_types import _StridedShard
from torch.optim.optimizer import Optimizer, ParamsT
from typing import Callable, Dict, Generator, List, Mapping, Optional, Sequence, Tuple, Union

from .muon_packed import (
    PackedPlan,
    exchange_buckets,
    foreach_chunks,
    foreach_length,
    make_plan,
    own_storage,
    pad_dim,
    pad_list,
    shard_shape,
)
from .newton_schulz_triton import newton_schulz_triton
from .opt_utils import (
    AsyncTask,
    AsyncRuntime,
    to_local,
    create_param_batches,
    pad_batch,
)
from .scalar_opts import (
    lion_update_foreach,
    adamw_update_foreach,
)

# Most matrices that one Newton-Schulz call batches across tensors
NEWTON_SCHULZ_BATCH = 32


class Muon(Optimizer):
    """
    Distributed Muon optimizer for PyTorch FSDP2. Also compatible with DDP.

    Args:
        params: Parameters for the optimizer.
        distributed_mesh: DeviceMesh or ProcessGroup for distributed training.
            Use DeviceMesh for FSDP2 and ProcessGroup for DistributedDataParallel.
            Can be overridden per parameter group by setting "distributed_mesh" in the group dict.
        lr: Base learning rate. For Muon, this will be scaled based on the matrix dimensions.
            For element-wise update rules, this is the actual learning rate and no additional scaling is done.
        mu: Momentum factor for Muon algorithm.
        betas: Tuple of (beta1, beta2) for AdamW and Lion algorithms.
        weight_decay: Weight decay factor.
        epsilon: Small value to avoid division by zero.
        nesterov: Whether to use Nesterov momentum.
        adjust_lr: How to adjust the learning rate for Muon updates ("spectral_norm" or "rms_norm" or None).
            "spectral_norm": Adjust based on spectral norm, for learning rate transfer across model scale.
            "rms_norm": Adjust based on RMS norm, for learning rate compatibility with Adam/AdamW.
            None: Do not adjust the learning rate.
        flatten: Whether to flatten 3D+ tensors to 2D for Muon updates.
            True: Tensors with 3+ dimensions are flattened to 2D. Use this for convolutional layers.
            False: Tensors are not flattened. 3D+ tensors are treated as batches of 2D matrices.
        use_triton: Whether to use Triton kernels for Newton-Schulz on CUDA tensors instead of the
            compiled PyTorch iteration. Ignored if custom function is provided.
        newton_schulz_func: Use a custom Newton-Schulz function for orthogonalization.
            Signature is `func(input: Tensor, epsilon: float) -> Tensor`.
        matrix_partitions: Maps a parameter to the sizes of the independent matrices packed
            along its penultimate dimension, e.g. {qkv_weight: (q_size, k_size, v_size)}.
            Each partition is orthogonalized on its own and gets its own adjusted learning
            rate, so a fused parameter is updated exactly like its unfused matrices would be.
            Parameters absent from this mapping are treated as a single matrix.

    Muon optimizer algorithm by Keller Jordan: https://kellerjordan.github.io/posts/muon/
    FSDP2 Muon uses all-to-all communications: https://www.essential.ai/blog/infra
    """

    _pack_sharded_matrices = True

    def __init__(
        self,
        params: ParamsT,
        fsdp_mesh_dim: int,
        world_mesh: DeviceMesh,
        distributed_mesh: Optional[Union[DeviceMesh, ProcessGroup]] = None,
        lr: float = 0.01,
        mu: float = 0.95,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        epsilon: float = 1e-8,
        nesterov: bool = False,
        adjust_lr: Optional[str] = "spectral_norm",
        flatten: bool = False,
        use_triton: bool = True,
        newton_schulz_func: Optional[Callable] = None,
        matrix_partitions: Optional[Mapping[Tensor, Sequence[int]]] = None,
    ):
        # Check hyperparameters
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if mu < 0.0:
            raise ValueError(f"Invalid momentum factor (mu): {mu}")
        if len(betas) != 2 or betas[0] < 0.0 or betas[1] < 0.0:
            raise ValueError(f"Invalid betas: {betas}")
        if adjust_lr not in ("spectral_norm", "rms_norm", "keller_muon", None):
            raise ValueError(
                f"Invalid adjust_lr value: {adjust_lr}. Must be 'spectral_norm', 'rms_norm', 'keller_muon', or None."
            )
            
            
        self.fsdp_mesh_dim = fsdp_mesh_dim
        self._world_mesh = world_mesh

        # Default arguments for each param group
        defaults = dict(
            lr=lr,
            mu=mu,
            beta1=betas[0],
            beta2=betas[1],
            weight_decay=weight_decay,
            algorithm="muon",
            step=0,
            epsilon=epsilon,
            nesterov=nesterov,
            flatten=flatten,
            adjust_lr=adjust_lr,
        )
        super().__init__(params, defaults)

        # Distributed configuration
        if isinstance(distributed_mesh, DeviceMesh):
            if distributed_mesh.ndim != 1:
                raise ValueError(
                    f"Only 1D DeviceMesh is supported, but got {distributed_mesh.ndim}D. For HSDP, provide the 1D sharded sub-mesh."
                )
            self._device_rank = distributed_mesh.get_local_rank()
            self._world_size = distributed_mesh.size()
            self._process_group = distributed_mesh.get_group()
        elif isinstance(distributed_mesh, ProcessGroup):
            self._device_rank = dist.get_rank(distributed_mesh)
            self._world_size = dist.get_world_size(distributed_mesh)
            self._process_group = distributed_mesh
        elif distributed_mesh is None:
            self._device_rank = 0
            self._world_size = 1
            self._process_group = None
        else:
            raise TypeError(
                f"Invalid distributed_mesh type: {type(distributed_mesh)}. Expected DeviceMesh or ProcessGroup."
            )
        self._distributed_mesh = distributed_mesh

        # Newton-Schulz configuration
        if newton_schulz_func is not None:
            if not callable(newton_schulz_func):
                raise TypeError(
                    f"newton_schulz_func must be a callable function, got {type(newton_schulz_func)}"
                )
            self._newton_schulz_func = newton_schulz_func
        elif use_triton:
            self._newton_schulz_func = newton_schulz_triton_on_cuda
        else:
            self._newton_schulz_func = zeropower_via_newtonschulz5

        # Matrix partition configuration
        self._matrix_partitions: Dict[Tensor, Tuple[int, ...]] = {}
        for param, partitions in (matrix_partitions or {}).items():
            partitions = tuple(partitions)
            if not partitions or any(size <= 0 for size in partitions):
                raise ValueError(f"Matrix partitions must be positive sizes, got {partitions}")
            if sum(partitions) != param.size(-2):
                raise ValueError(
                    f"Matrix partitions {partitions} sum to {sum(partitions)}, but the "
                    f"parameter's penultimate dimension is {param.size(-2)}"
                )
            self._matrix_partitions[param] = partitions

        num_partitioned = sum(
            1
            for group in self.param_groups
            for param in group["params"]
            if param in self._matrix_partitions
        )
        if num_partitioned != len(self._matrix_partitions):
            raise ValueError(
                "matrix_partitions contains parameters that were not given to the optimizer"
            )

        self._matrix_shard_dims: Dict[tuple, Tuple[Optional[int], bool]] = {}
        self._packed_plans: Dict[tuple, Tuple[Tuple[weakref.ref, ...], PackedPlan]] = {}

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        muon_groups = []
        lion_groups = []
        adamw_groups = []

        for group in self.param_groups:
            # Increment step
            group["step"] += 1

            # Split parameter groups by algorithm
            algo = group["algorithm"]
            if algo == "muon":
                muon_groups.append(group)
            elif algo == "lion":
                lion_groups.append(group)
            elif algo == "adamw":
                adamw_groups.append(group)
            else:
                raise ValueError(f"Unknown algorithm: {algo}")

        # Create async tasks for each algorithm
        muon_tasks = self._create_muon_tasks(muon_groups)
        lion_tasks = self._create_lion_tasks(lion_groups)
        adamw_tasks = self._create_adamw_tasks(adamw_groups)

        all_tasks = chain(muon_tasks, lion_tasks, adamw_tasks)
        runtime = AsyncRuntime(all_tasks, max_concurrent_tasks=3)
        runtime.run()

        return loss

    def _get_or_initialize_state(self, param: Tensor, algo: str) -> dict:
        """
        Get optimizer state for the given parameter tensor,
        or lazy-initialize it if it doesn't exist.
        """
        state = self.state[param]
        if not state:
            state["momentum"] = torch.zeros_like(param)
            if algo == "adamw":
                state["variance"] = torch.zeros_like(param)
        return state

    def _get_group_mesh_info(self, group: dict) -> Tuple[int, int, Optional[ProcessGroup], Optional[Union[DeviceMesh, ProcessGroup]]]:
        """
        Get mesh-related info for a parameter group.
        Returns (device_rank, world_size, process_group, distributed_mesh).
        Falls back to default mesh if group doesn't specify one.
        """
        group_mesh_name = group.get("distributed_mesh_name", None)
        if group_mesh_name is None:
            return (self._device_rank, self._world_size, self._process_group, self._distributed_mesh)
        
        # Parse group-specific mesh
        group_mesh = self._world_mesh[group_mesh_name]
        if isinstance(group_mesh, DeviceMesh):
            if group_mesh.ndim != 1:
                raise ValueError(
                    f"Only 1D DeviceMesh is supported, but got {group_mesh.ndim}D. For HSDP, provide the 1D sharded sub-mesh."
                )
            device_rank = group_mesh.get_local_rank()
            world_size = group_mesh.size()
            process_group = group_mesh.get_group()
            return (device_rank, world_size, process_group, group_mesh)
        elif isinstance(group_mesh, ProcessGroup):
            device_rank = dist.get_rank(group_mesh)
            world_size = dist.get_world_size(group_mesh)
            return (device_rank, world_size, group_mesh, group_mesh)
        else:
            raise TypeError(
                f"Invalid distributed_mesh type in group: {type(group_mesh)}. Expected DeviceMesh or ProcessGroup."
            )

    def _create_muon_tasks(
        self,
        param_groups: List[dict],
        algo_name: str = "muon",
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to create batches of Muon matrices and generate
        AsyncTask objects so we can process multiple batches concurrently.
        """
        for group in param_groups:
            assert group["algorithm"] == algo_name
            assert all(
                p.ndim >= 2 for p in group["params"]
            ), "Muon optimizer only supports matrix parameters."

            group_params = [p for p in group["params"] if p.grad is not None]
            if not group_params:
                continue

            # Get group-specific mesh info
            device_rank, world_size, process_group, distributed_mesh = self._get_group_mesh_info(group)

            # Wrap hyperparameters in tensors for torch.compile
            lr = torch.tensor(group["lr"])
            mu = torch.tensor(group["mu"])
            weight_decay = torch.tensor(group["weight_decay"])
            epsilon = torch.tensor(group["epsilon"])
            nesterov = group["nesterov"]
            flatten = group["flatten"]
            adjust_lr = group["adjust_lr"]

            # Matrices sharded on a matrix dim go whole to one owner rank each
            packed, shard_dims, group_params = self._split_packed_params(
                group_params, flatten, world_size, process_group, distributed_mesh
            )
            # Each bucket of sharded matrices is exchanged on its own, so the runtime
            # overlaps one bucket's communication with another's Newton-Schulz
            keys = [
                (p.shape, p.dtype, shard_dim, self._matrix_partitions.get(p))
                for p, shard_dim in zip(packed, shard_dims)
            ]
            for bucket in exchange_buckets(packed, keys, world_size):
                params = [packed[i] for i in bucket]
                partitions = [self._matrix_partitions.get(p) for p in params]
                plan = self._get_packed_plan(
                    params, [shard_dims[i] for i in bucket], partitions, world_size
                )
                yield AsyncTask(
                    muon_update_packed_async(
                        X=params,
                        G=[p.grad for p in params],
                        M=[self._get_or_initialize_state(p, algo_name)["momentum"] for p in params],
                        plan=plan,
                        partitions=partitions,
                        lr=lr,
                        momentum=mu,
                        weight_decay=weight_decay,
                        epsilon=epsilon,
                        nesterov=nesterov,
                        flatten=flatten,
                        adjust_lr=adjust_lr,
                        device_rank=device_rank,
                        world_size=world_size,
                        process_group=process_group,
                        newton_schulz_func=self._newton_schulz_func,
                    )
                )

            # Create batches of parameters of size world_size
            for params in create_param_batches(
                group_params,
                batch_size=world_size,
                matrix_partitions=self._matrix_partitions,
            ):
                # Every parameter in a batch shares the same partitioning
                partitions = self._matrix_partitions.get(params[0])
                gradients = [p.grad for p in params]
                states = [self._get_or_initialize_state(p, algo_name) for p in params]
                momentums = [s["momentum"] for s in states]

                # Get sharding dimension
                sharded_tensor_dim = None
                is_local = False
                if isinstance(params[0], DTensor):
                    if not isinstance(distributed_mesh, DeviceMesh):
                        raise RuntimeError(
                            "Must create optimizer with DeviceMesh if using DTensor parameters."
                        )
                    sharded_tensor_dim, is_local = self._get_matrix_shard_dim(
                        params[0], flatten, process_group
                    )

                # Local batches need no communication, so they skip the dummy padding
                if not is_local:
                    params = pad_batch(params, world_size)
                    gradients = pad_batch(gradients, world_size)
                    momentums = pad_batch(momentums, world_size)

                yield AsyncTask(
                    muon_update_batch_async(
                        X=params,
                        G=gradients,
                        M=momentums,
                        lr=lr,
                        momentum=mu,
                        weight_decay=weight_decay,
                        epsilon=epsilon,
                        nesterov=nesterov,
                        flatten=flatten,
                        adjust_lr=adjust_lr,
                        device_rank=device_rank,
                        world_size=world_size,
                        shard_dim=sharded_tensor_dim,
                        is_local=is_local,
                        process_group=process_group,
                        newton_schulz_func=self._newton_schulz_func,
                        partitions=partitions,
                    )
                )

    def _get_packed_plan(
        self,
        params: List[Tensor],
        shard_dims: List[int],
        partitions: List[Optional[Tuple[int, ...]]],
        world_size: int,
    ) -> PackedPlan:
        """
        The cached owner plan of a bucket, rebuilt unless it was made for these very tensors.
        The cache holds weak references, so it neither keeps tensors alive nor trusts a reused id.
        """
        key = (tuple(id(p) for p in params), world_size)
        cached = self._packed_plans.get(key)
        if cached is None or any(ref() is not p for ref, p in zip(cached[0], params)):
            self._packed_plans = {
                k: v for k, v in self._packed_plans.items() if all(ref() is not None for ref in v[0])
            }
            cached = (
                tuple(weakref.ref(p) for p in params),
                make_plan(params, shard_dims, partitions, world_size),
            )
            self._packed_plans[key] = cached
        return cached[1]

    def _split_packed_params(
        self,
        params: List[Tensor],
        flatten: bool,
        world_size: int,
        process_group: Optional[ProcessGroup],
        distributed_mesh: Optional[Union[DeviceMesh, ProcessGroup]],
    ) -> Tuple[List[Tensor], List[int], List[Tensor]]:
        """
        Split parameters into those sharded on a matrix dim, with that dim, and the rest.
        """
        packed, shard_dims, rest = [], [], []
        for p in params:
            if (
                self._pack_sharded_matrices
                and world_size > 1
                and isinstance(p, DTensor)
                and isinstance(distributed_mesh, DeviceMesh)
            ):
                shard_dim, is_local = self._get_matrix_shard_dim(p, flatten, process_group)
                if shard_dim is not None and not is_local:
                    packed.append(p)
                    shard_dims.append(shard_dim)
                    continue
            rest.append(p)
        return packed, shard_dims, rest

    def _get_matrix_shard_dim(
        self,
        param: DTensor,
        flatten: bool,
        process_group: Optional[ProcessGroup],
    ) -> Tuple[Optional[int], bool]:
        """
        Return (tensor dim that Muon must gather over, whether every local shard is whole matrices).
        """
        cache_key = (param.device_mesh, param.placements, param.ndim, flatten, process_group)
        if cache_key in self._matrix_shard_dims:
            return self._matrix_shard_dims[cache_key]

        # Skip any shard placements on size-1 mesh dimension = Replicate().
        # _StridedShard is not a Shard subclass in newer torch releases, so check both.
        shard_placements = [
            (mesh_dim, placement)
            for mesh_dim, placement in enumerate(param.placements)
            if isinstance(placement, (Shard, _StridedShard))
            and param.device_mesh.size(mesh_dim) > 1
        ]
        has_batch_shard = False
        if not flatten:
            # Shards on batch dims (e.g. the expert dim) leave whole matrices on every device
            matrix_dims = (param.ndim - 2, param.ndim - 1)
            matrix_shard_placements = [
                (mesh_dim, placement)
                for mesh_dim, placement in shard_placements
                if placement.dim in matrix_dims
            ]
            has_batch_shard = len(matrix_shard_placements) < len(shard_placements)
            shard_placements = matrix_shard_placements

        if not shard_placements:
            # Replicated matrices are split across devices; batch-sharded ones differ per device
            result = (None, has_batch_shard)
        else:
            # The all-to-all rebuilds matrices over one mesh dim; a second matrix shard (e.g. TP)
            # would leave each device orthogonalizing a partial matrix
            if len(shard_placements) > 1:
                raise NotImplementedError(
                    f"Muon supports at most one sharded matrix dimension, but a parameter with "
                    f"placements {param.placements} is sharded on matrix dims over mesh dims "
                    f"{[mesh_dim for mesh_dim, _ in shard_placements]}"
                )
            sharded_mesh_dim, placement = shard_placements[0]
            if isinstance(placement, _StridedShard):
                raise NotImplementedError(
                    f"Muon does not support {placement} on a matrix dimension of a parameter "
                    f"with placements {param.placements}"
                )

            self._check_shard_group(param, sharded_mesh_dim, process_group)
            result = (placement.dim, False)

        self._matrix_shard_dims[cache_key] = result
        return result

    @staticmethod
    def _check_shard_group(
        param: DTensor, sharded_mesh_dim: int, process_group: Optional[ProcessGroup]
    ):
        """
        Raise unless Muon's all-to-all runs over the parameter's shard ranks, in shard order.
        """
        # Group objects can differ for the same ranks (e.g. a separately built HSDP mesh), so
        # compare ranks. Group ranks are sorted, but DTensor sizes shards by mesh coordinate,
        # so the mesh must also list the shard ranks in ascending order.
        mesh_index = list(param.device_mesh.get_coordinate())
        mesh_index[sharded_mesh_dim] = slice(None)
        shard_ranks = param.device_mesh.mesh[tuple(mesh_index)].tolist()
        comm_ranks = (
            dist.get_process_group_ranks(process_group) if process_group is not None else None
        )
        if shard_ranks != comm_ranks:
            raise ValueError(
                f"Parameter is sharded over ranks {shard_ranks}, but Muon communicates "
                f"over ranks {comm_ranks}"
            )

    def _create_lion_tasks(
        self,
        param_groups: List[dict],
        algo_name: str = "lion",
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to generate AsyncTask objects for Lion updates.
        """
        for group in param_groups:
            assert group["algorithm"] == algo_name

            # Get parameters and optimizer states
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            gradients = [p.grad for p in params]
            states = [self._get_or_initialize_state(p, algo_name) for p in params]
            momentums = [s["momentum"] for s in states]

            # Wrap hyperparameters in tensors for torch.compile
            lr = torch.tensor(group["lr"])
            beta1 = torch.tensor(group["beta1"])
            beta2 = torch.tensor(group["beta2"])
            weight_decay = torch.tensor(group["weight_decay"])

            yield AsyncTask(
                lion_update_foreach_async(
                    X=to_local(params),
                    G=to_local(gradients),
                    M=to_local(momentums),
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=weight_decay,
                )
            )

    def _create_adamw_tasks(
        self,
        param_groups: List[dict],
        algo_name: str = "adamw",
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to generate AsyncTask objects for AdamW updates.
        """
        for group in param_groups:
            assert group["algorithm"] == algo_name

            # Get parameters and optimizer states
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            gradients = [p.grad for p in params]
            states = [self._get_or_initialize_state(p, algo_name) for p in params]
            momentums = [s["momentum"] for s in states]
            variances = [s["variance"] for s in states]

            # Wrap hyperparameters in tensors for torch.compile
            lr = torch.tensor(group["lr"])
            beta1 = torch.tensor(group["beta1"])
            beta2 = torch.tensor(group["beta2"])
            weight_decay = torch.tensor(group["weight_decay"])
            epsilon = torch.tensor(group["epsilon"])
            step = torch.tensor(group["step"])

            yield AsyncTask(
                adamw_update_foreach_async(
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
                )
            )


def muon_update_batch_async(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    M: List[Tensor],  # Momentum buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    momentum: Tensor,  # Momentum factor (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    epsilon: Tensor,  # Epsilon (scalar tensor)
    nesterov: bool,  # Whether to use Nesterov momentum
    flatten: bool,  # Whether to flatten 3D+ tensors to 2D
    adjust_lr: Optional[str],  # How to adjust learning rate
    device_rank: int,  # Rank of the current device
    world_size: int,  # Total number of devices to parallelize over
    shard_dim: Optional[int] = None,  # Shard dimension for DTensor (if applicable)
    is_local: bool = False,  # Whether every device holds whole matrices of its own
    process_group: Optional[ProcessGroup] = None,
    newton_schulz_func: Optional[Callable] = None,
    partitions: Optional[Tuple[int, ...]] = None,  # Independent matrices packed along dim -2
) -> Generator[None, None, None]:
    """
    Batched version of Muon update. Batch size should be equal to number of GPUs.
    All tensors in a batch should have identical shape, sharding, and dtype.
    Identical hyperparameters are used for all tensors in the batch.
    """

    assert len(X) == len(G)
    assert len(X) == len(M)
    assert is_local or len(X) == world_size

    # Update momentum and compute the inputs for orthogonalization
    U = muon_update_pre_orthogonalize(
        G=to_local(G),
        M=to_local(M),
        momentum=momentum,
        nesterov=nesterov,
    )

    # Get one whole matrix for each device to orthogonalize
    if is_local:
        # Local shards are already whole matrices that differ across devices, so each
        # device orthogonalizes its own without communication
        U = muon_orthogonalize_local_batch(
            U,
            partitions=partitions,
            newton_schulz_func=newton_schulz_func,
            epsilon=epsilon,
        )

    elif shard_dim is not None:
        # Use all-to-all to transform from a batch of shards to a single whole matrix
        # https://www.essential.ai/blog/infra
        assert (
            process_group is not None
        ), "process_group must be provided for sharded DTensors"
        assert isinstance(X[0], DTensor), "X should contain DTensors"
        assert not isinstance(U[0], DTensor), "U should contain local shards"

        # FSDP permits uneven shards, including empty local shards when the
        # sharded dimension is smaller than the process group. NCCL all-to-all
        # requires every peer transfer to have the same shape, so pad local
        # shards for communication and remove that padding before Muon's matrix
        # update. The reverse all-to-all uses the same padded shape and trims
        # each rank back to its original local shard afterward.
        global_shard_size = X[0].size(shard_dim)
        padded_local_size = (global_shard_size + world_size - 1) // world_size
        local_shard_size = U[0].size(shard_dim)
        assert all(u.size(shard_dim) == local_shard_size for u in U)

        def pad_to_size(tensor: Tensor, size: int) -> Tensor:
            pad_amount = size - tensor.size(shard_dim)
            if pad_amount == 0:
                return tensor
            # F.pad takes (before, after) pairs starting from the last dim
            pad = [0, 0] * (tensor.ndim - 1 - shard_dim) + [0, pad_amount]
            return torch.nn.functional.pad(tensor, pad)

        padded_U = [pad_to_size(u, padded_local_size) for u in U]

        # Allocate buffers to receive shards of one whole matrix from other devices
        single_matrix_shards = [torch.empty_like(padded_U[0]) for _ in U]

        # Redistribute the shards to form one unique full tensor on each device
        work = dist.all_to_all(
            single_matrix_shards, padded_U, group=process_group, async_op=True
        )
        yield
        work.wait()

        # Concatenate shards and discard communication padding before
        # orthogonalization so the update is exactly the unpadded Muon update.
        single_matrix = torch.cat(single_matrix_shards, dim=shard_dim)
        single_matrix = single_matrix.narrow(shard_dim, 0, global_shard_size)
        single_matrix = muon_orthogonalize_update(
            single_matrix,
            partitions=partitions,
            newton_schulz_func=newton_schulz_func,
            flatten=flatten,
            epsilon=epsilon,
        )

        # Re-pad the updated matrix so every reverse all-to-all transfer has the
        # same shape, then trim this rank's received shards to its FSDP layout.
        single_matrix = pad_to_size(single_matrix, padded_local_size * world_size)
        single_matrix_shards = [
            x.contiguous()
            for x in torch.tensor_split(single_matrix, world_size, dim=shard_dim)
        ]

        # Redistribute the orthogonalized tensor back to original layout
        padded_U = [torch.empty_like(single_matrix_shards[0]) for _ in U]
        work = dist.all_to_all(
            padded_U, single_matrix_shards, group=process_group, async_op=True
        )
        yield
        work.wait()
        U = [u.narrow(shard_dim, 0, local_shard_size) for u in padded_U]

    else:
        # Matrices are not sharded, so we can directly orthogonalize
        # Get a single matrix corresponding to this device
        single_matrix = U[device_rank]
        assert not isinstance(single_matrix, DTensor)

        single_matrix = muon_orthogonalize_update(
            single_matrix,
            partitions=partitions,
            newton_schulz_func=newton_schulz_func,
            flatten=flatten,
            epsilon=epsilon,
        )

        if process_group is not None and process_group.size() > 1:
            # Allocate empty tensors to receive updates from other devices
            U = [torch.empty_like(u) for u in U]

            # All gather orthogonalized results from other devices into buffer
            work = dist.all_gather(
                U, single_matrix.contiguous(), group=process_group, async_op=True
            )
            yield
            work.wait()

        else:
            # Single GPU case, no need to gather
            assert world_size == 1
            U = [single_matrix]

    # Scale the learning rate and update model parameters with the orthogonalized output.
    # Compute the scaling before to_local(X) because it needs the full tensor shape.
    if partitions is None:
        muon_update_post_orthogonalize(
            X=to_local(X),
            U=U,
            base_lr=lr,
            adjusted_lr=adjust_lr_for_shape(lr, X[0].shape, adjust_lr),
            weight_decay=weight_decay,
        )
    else:
        # Every partition is a matrix of its own shape, so the adjustment varies by row
        row_lr = partition_row_learning_rates(partitions, X[0].size(-1), lr, adjust_lr)
        if shard_dim == X[0].ndim - 2:
            # This device holds the same slice of rows for every parameter in the
            # batch. DTensor Shard uses a torch.chunk layout: every rank holds
            # padded_local_size rows except a trailing short or empty shard, so
            # slice with chunk offsets rather than torch.tensor_split.
            start = min(device_rank * padded_local_size, row_lr.numel())
            row_lr = row_lr.narrow(0, start, min(padded_local_size, row_lr.numel() - start))
        muon_update_post_orthogonalize_partitioned(
            X=to_local(X),
            U=U,
            base_lr=lr,
            row_lr=row_lr.unsqueeze(-1).to(X[0].device),
            weight_decay=weight_decay,
        )


def muon_update_packed_async(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    M: List[Tensor],  # Momentum buffer (modified in place)
    plan: PackedPlan,  # Owner rank and shard sizes of every matrix
    partitions: List[Optional[Tuple[int, ...]]],  # Independent matrices packed along dim -2
    lr: Tensor,  # Learning rate (scalar tensor)
    momentum: Tensor,  # Momentum factor (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    epsilon: Tensor,  # Epsilon (scalar tensor)
    nesterov: bool,  # Whether to use Nesterov momentum
    flatten: bool,  # Whether to flatten 3D+ tensors to 2D
    adjust_lr: Optional[str],  # How to adjust learning rate
    device_rank: int,  # Rank of the current device
    world_size: int,  # Total number of devices to parallelize over
    process_group: ProcessGroup,
    newton_schulz_func: Callable,
) -> Generator[None, None, None]:
    """
    Muon update for DTensors sharded on a matrix dim, of any shapes, each orthogonalized
    whole by its owner rank. One all-to-all sends every owner the shards of its matrices
    and one sends the orthogonalized shards back.
    """
    length = foreach_length(world_size)
    U = muon_update_pre_orthogonalize_packed(G, M, momentum, nesterov, length)
    owned = plan.owned_by(device_rank)
    by_owner = [plan.owned_by(rank) for rank in range(world_size)]

    def piece_numel(index: int, rank: int) -> int:
        dim = plan.shard_dims[index]
        return plan.shard_sizes[index][rank] * X[index].numel() // X[index].size(dim)

    owner_splits = [sum(U[index].numel() for index in by_owner[rank]) for rank in range(world_size)]
    owned_splits = [sum(piece_numel(index, rank) for index in owned) for rank in range(world_size)]

    # Send every owner this rank's shards of its matrices
    send = [U[index].reshape(-1) for rank in range(world_size) for index in by_owner[rank]]
    outgoing = torch.cat(send) if send else U[0].new_empty(0)
    received = U[0].new_empty(sum(owned_splits))
    work = dist.all_to_all_single(
        received,
        outgoing,
        output_split_sizes=owned_splits,
        input_split_sizes=owner_splits,
        group=process_group,
        async_op=True,
    )
    yield
    work.wait()
    del outgoing

    pieces: Dict[int, List[Tensor]] = {index: [] for index in owned}
    for rank, chunk in enumerate(torch.split(received, owned_splits)):
        flat = torch.split(chunk, [piece_numel(index, rank) for index in owned])
        for index, piece in zip(owned, flat):
            dim = plan.shard_dims[index]
            pieces[index].append(piece.view(shard_shape(X[index], dim, plan.shard_sizes[index][rank])))

    # Orthogonalize each owned matrix and cut it back into every rank's shard
    back = []
    for index in owned:
        dim, padded = plan.shard_dims[index], plan.padded_sizes[index]
        matrix = torch.cat([pad_dim(piece, dim, padded) for piece in pieces[index]], dim=dim)
        matrix = matrix.narrow(dim, 0, X[index].size(dim))
        matrix = muon_orthogonalize_update(
            matrix,
            partitions=partitions[index],
            newton_schulz_func=newton_schulz_func,
            flatten=flatten,
            epsilon=epsilon,
        )
        # The return exchange is sized in the update's dtype on every rank, owners or not
        matrix = pad_dim(matrix.to(dtype=U[index].dtype), dim, padded * world_size)
        back.append(
            [
                shard.narrow(dim, 0, plan.shard_sizes[index][rank]).contiguous()
                for rank, shard in enumerate(torch.tensor_split(matrix, world_size, dim=dim))
            ]
        )
    del pieces, received

    send = [back[k][rank].reshape(-1) for rank in range(world_size) for k in range(len(owned))]
    outgoing = torch.cat(send) if send else U[0].new_empty(0)
    del send, back
    returned = U[0].new_empty(sum(owner_splits))
    work = dist.all_to_all_single(
        returned,
        outgoing,
        output_split_sizes=owner_splits,
        input_split_sizes=owned_splits,
        group=process_group,
        async_op=True,
    )
    yield
    work.wait()
    del outgoing

    updates: List[Optional[Tensor]] = [None] * len(X)
    for rank, chunk in enumerate(torch.split(returned, owner_splits)):
        flat = torch.split(chunk, [U[index].numel() for index in by_owner[rank]])
        for index, piece in zip(by_owner[rank], flat):
            updates[index] = piece.view(U[index].shape)
    del U

    # Partitioned matrices sharded on dim -2 need this rank's rows of the per-row learning rates
    row_slices = []
    for index, x in enumerate(X):
        if partitions[index] is None or plan.shard_dims[index] != x.ndim - 2:
            row_slices.append(None)
            continue
        padded = plan.padded_sizes[index]
        start = min(device_rank * padded, x.size(-2))
        row_slices.append((start, min(padded, x.size(-2) - start)))
    muon_update_post_orthogonalize_packed(
        X, updates, partitions, lr, weight_decay, adjust_lr, row_slices, length
    )


def muon_update_pre_orthogonalize_packed(
    G: List[Tensor],
    M: List[Tensor],
    momentum: Tensor,
    nesterov: bool,
    length: int,
) -> List[Tensor]:
    """
    muon_update_pre_orthogonalize on local shards, in lists of `length` entries of one shape
    and dtype each.
    """
    local_grads, local_momentums = to_local(G), to_local(M)
    lists: Dict[tuple, List[int]] = {}
    for index, (local_grad, local_momentum) in enumerate(zip(local_grads, local_momentums)):
        key = (local_momentum.shape, local_momentum.dtype, local_grad.dtype)
        lists.setdefault(key, []).append(index)
    U: List[Optional[Tensor]] = [None] * len(local_grads)
    for indices in lists.values():
        for chunk in foreach_chunks(indices, length):
            outputs = muon_update_pre_orthogonalize(
                G=pad_list([own_storage(local_grads[index]) for index in chunk], length),
                M=pad_list([local_momentums[index] for index in chunk], length),
                momentum=momentum,
                nesterov=nesterov,
            )
            for index, update in zip(chunk, outputs):
                U[index] = update
    return U


def muon_update_post_orthogonalize_packed(
    X: List[Tensor],
    updates: List[Tensor],
    partitions: List[Optional[Tuple[int, ...]]],
    lr: Tensor,
    weight_decay: Tensor,
    adjust_lr: Optional[str],
    row_slices: List[Optional[Tuple[int, int]]],
    length: int,
):
    """
    The weight update of every matrix, in lists of `length` entries of one local shape,
    learning rate and dtype. `row_slices` gives the (start, length) of this rank's rows of
    each partitioned matrix sharded on dim -2.
    """
    plain: Dict[tuple, List[int]] = {}
    partitioned: Dict[tuple, List[int]] = {}
    row_lrs: Dict[tuple, Tensor] = {}
    for index, x in enumerate(X):
        parts = partitions[index]
        if parts is None:
            plain.setdefault((tuple(x.shape), to_local(x).shape, x.dtype), []).append(index)
            continue
        rows = row_slices[index]
        lr_key = (parts, x.size(-1), rows)
        if lr_key not in row_lrs:
            row_lr = partition_row_learning_rates(parts, x.size(-1), lr, adjust_lr)
            if rows is not None:
                row_lr = row_lr.narrow(0, *rows)
            row_lrs[lr_key] = row_lr.unsqueeze(-1).to(x.device)
        partitioned.setdefault((lr_key, to_local(x).shape, x.dtype), []).append(index)

    for (shape, _, _), indices in plain.items():
        adjusted_lr = adjust_lr_for_shape(lr, torch.Size(shape), adjust_lr)
        for chunk in foreach_chunks(indices, length):
            muon_update_post_orthogonalize(
                X=pad_list([to_local(X[index]) for index in chunk], length),
                U=pad_list([own_storage(updates[index]) for index in chunk], length),
                base_lr=lr,
                adjusted_lr=adjusted_lr,
                weight_decay=weight_decay,
            )
    for (lr_key, _, _), indices in partitioned.items():
        for chunk in foreach_chunks(indices, length):
            muon_update_post_orthogonalize_partitioned(
                X=pad_list([to_local(X[index]) for index in chunk], length),
                U=pad_list([own_storage(updates[index]) for index in chunk], length),
                base_lr=lr,
                row_lr=row_lrs[lr_key],
                weight_decay=weight_decay,
            )


def adamw_update_foreach_async(
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
) -> Generator[None, None, None]:
    """
    Async wrapper around foreach AdamW update.
    """
    adamw_update_foreach(X, G, M, V, lr, beta1, beta2, weight_decay, step, epsilon)
    yield


def lion_update_foreach_async(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    M: List[Tensor],  # Momentum buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
) -> Generator[None, None, None]:
    """
    Async wrapper around foreach Lion update.
    """
    lion_update_foreach(X, G, M, lr, beta1, beta2, weight_decay)
    yield


@torch.compile(fullgraph=True)
def muon_update_pre_orthogonalize(
    G: List[Tensor],
    M: List[Tensor],
    momentum: Tensor,
    nesterov: bool,
) -> List[Tensor]:
    """
    Update momentum with gradient and compute the input to orthogonalization.
    Inputs and outputs should be lists of regular Tensor, not DTensor.
    This is a separate function for compatibility with torch.compile().
    """
    dtype = M[0].dtype
    G = [g.to(dtype=dtype) for g in G]

    # Update momentum with new gradient
    torch._foreach_mul_(M, momentum)
    torch._foreach_add_(M, G)

    if nesterov:
        U = torch._foreach_mul(M, momentum)
        torch._foreach_add_(U, G)
    else:
        U = M

    # Convert to bfloat16 before communication
    U = [u.to(dtype=torch.bfloat16) for u in U]

    return U


@torch.compile(fullgraph=True)
def muon_update_post_orthogonalize(
    X: List[Tensor],
    U: List[Tensor],
    base_lr: Tensor,
    adjusted_lr: Tensor,
    weight_decay: Tensor,
):
    """
    Apply weight decay and weight update after orthogonalization.
    Inputs and outputs should be lists of regular Tensor, not DTensor.
    This is a separate function for compatibility with torch.compile().
    """
    # Apply weight decay
    torch._foreach_mul_(X, 1 - base_lr * weight_decay)

    # Weight update
    U = torch._foreach_mul(U, adjusted_lr)
    torch._foreach_sub_(X, U)


@torch.compile(fullgraph=True)
def muon_update_post_orthogonalize_partitioned(
    X: List[Tensor],
    U: List[Tensor],
    base_lr: Tensor,  # Learning rate (scalar tensor)
    row_lr: Tensor,  # Adjusted learning rate per row of dimension -2
    weight_decay: Tensor,  # Weight decay (scalar tensor)
):
    """
    Weight decay and weight update for a parameter packing several matrices along
    dimension -2. Same arithmetic as muon_update_post_orthogonalize, except that the
    adjusted learning rate varies by row because each partition has its own shape.
    """
    # Apply weight decay
    torch._foreach_mul_(X, 1 - base_lr * weight_decay)

    # Weight update. Foreach ops only accept scalars, so broadcast one tensor at a time.
    U = [u * row_lr for u in U]
    torch._foreach_sub_(X, U)


def muon_update_newton_schulz(
    X: Tensor,
    newton_schulz_func: Callable,
    flatten: bool,
    epsilon: Tensor,
) -> Tensor:
    """
    Flatten the input tensor if needed and call the Newton-Schulz function.
    """
    original_shape = X.shape
    if flatten and X.ndim >= 3:
        # Flatten 3D+ tensors to 2D matrix
        X = X.flatten(start_dim=1)
    elif X.ndim >= 4:
        # Given 4D+ batch, flatten to 3D batch
        X = X.flatten(end_dim=-3)

    return newton_schulz_func(X, epsilon=epsilon).reshape(original_shape)


def muon_orthogonalize_update(
    X: Tensor,
    partitions: Optional[Tuple[int, ...]],
    newton_schulz_func: Callable,
    flatten: bool,
    epsilon: Tensor,
) -> Tensor:
    """
    Orthogonalize one whole (unsharded) momentum tensor.

    Without partitions this is a single Newton-Schulz over the whole tensor. With
    partitions, X packs several independent matrices along dimension -2, and each one
    is orthogonalized on its own; equal partitions share one batched call.
    """
    if partitions is None or len(partitions) == 1:
        return muon_update_newton_schulz(
            X, newton_schulz_func=newton_schulz_func, flatten=flatten, epsilon=epsilon
        )

    if len(set(partitions)) == 1 and not (flatten and X.ndim >= 3):
        # Equal partitions are a batch of matrices, orthogonalized in one call
        blocks = X.unflatten(-2, (len(partitions), partitions[0]))
        return muon_update_newton_schulz(
            blocks, newton_schulz_func=newton_schulz_func, flatten=False, epsilon=epsilon
        ).reshape(X.shape)

    updates = [
        # Newton-Schulz reads the partition many times, so pay for one contiguous copy
        muon_update_newton_schulz(
            partition.contiguous(),
            newton_schulz_func=newton_schulz_func,
            flatten=flatten,
            epsilon=epsilon,
        )
        for partition in torch.split(X, partitions, dim=-2)
    ]
    return torch.cat(updates, dim=-2)


def muon_orthogonalize_local_batch(
    U: List[Tensor],
    partitions: Optional[Tuple[int, ...]],
    newton_schulz_func: Callable,
    epsilon: Tensor,
) -> List[Tensor]:
    """
    Orthogonalize same-shape tensors of whole matrices, e.g. each device's experts.

    The matrices of tensors holding fewer than NEWTON_SCHULZ_BATCH matrices share
    Newton-Schulz calls of up to that many matrices; a larger tensor is one call of its own.
    Equal partitions count as matrices of their own.
    """
    if partitions is not None and len(set(partitions)) > 1:
        return [
            u
            if u.numel() == 0
            else muon_orthogonalize_update(
                u,
                partitions=partitions,
                newton_schulz_func=newton_schulz_func,
                flatten=False,
                epsilon=epsilon,
            )
            for u in U
        ]

    rows = U[0].size(-2) if partitions is None else partitions[0]
    stacks = {index: u.reshape(-1, rows, u.size(-1)) for index, u in enumerate(U) if u.numel()}
    calls: List[List[int]] = []
    count = 0
    for index, stack in stacks.items():
        if not calls or count + stack.size(0) > NEWTON_SCHULZ_BATCH:
            calls.append([])
            count = 0
        calls[-1].append(index)
        count += stack.size(0)

    U = list(U)
    for indices in calls:
        batch = [stacks[index] for index in indices]
        result = muon_update_newton_schulz(
            torch.cat(batch) if len(batch) > 1 else batch[0],
            newton_schulz_func=newton_schulz_func,
            flatten=False,
            epsilon=epsilon,
        )
        # Triton returns tall matrices transposed, so reshape rather than view
        for index, piece in zip(indices, torch.split(result, [b.size(0) for b in batch])):
            U[index] = piece.reshape(U[index].shape)
    return U


def partition_row_learning_rates(
    partitions: Tuple[int, ...],
    fan_in: int,
    lr: Tensor,
    adjust_lr: Optional[str],
) -> Tensor:
    """
    One adjusted learning rate per row of the penultimate dimension.

    Every partition is a matrix of its own shape, so its adjustment differs. Expressing
    the adjustment per row lets the caller apply it in the same place, and with the same
    arithmetic, as the single adjusted learning rate of an unpartitioned parameter.
    """
    return torch.cat(
        [
            adjust_lr_for_shape(lr, torch.Size((fan_out, fan_in)), adjust_lr).expand(fan_out)
            for fan_out in partitions
        ]
    )


def adjust_lr_for_shape(lr: Tensor, param_shape: torch.Size, adjust_lr: Optional[str]) -> Tensor:
    """
    Scale the learning rate for a matrix of the given shape.
    """
    if adjust_lr is None:
        return lr
    if adjust_lr == "spectral_norm":
        return adjust_lr_spectral_norm(lr, param_shape)
    if adjust_lr == "rms_norm":
        return adjust_lr_rms_norm(lr, param_shape)
    if adjust_lr == "keller_muon":
        return adjust_lr_keller_muon(lr, param_shape)
    raise ValueError(f"Unknown adjust_lr value: {adjust_lr}")


def adjust_lr_rms_norm(lr, param_shape):
    # Adjust learning rate for constant element-wise RMS norm
    # https://arxiv.org/abs/2502.16982
    A, B = param_shape[-2:]
    adjusted_ratio = 0.2 * math.sqrt(max(A, B))
    adjusted_lr = lr * adjusted_ratio
    return adjusted_lr


def adjust_lr_spectral_norm(lr, param_shape):
    # Adjust from spectral norm 1 to RMS operator norm 1
    # https://arxiv.org/abs/2310.17813
    fan_out, fan_in = param_shape[-2:]
    adjusted_lr = lr * math.sqrt(fan_out / fan_in)
    return adjusted_lr

def adjust_lr_keller_muon(lr, param_shape):
    # Adjust from spectral norm 1 to RMS operator norm 1
    # https://arxiv.org/abs/2310.17813
    fan_out, fan_in = param_shape[-2:] # need -2: instead of 2: because it would break experts params that are 3d
    adjusted_lr = lr * max(1, fan_out / fan_in)**0.5
    return adjusted_lr

def newton_schulz_triton_on_cuda(G: Tensor, epsilon: float = 1e-7) -> Tensor:
    """
    Triton Newton-Schulz for CUDA tensors, the compiled PyTorch iteration for others.
    """
    if G.is_cuda:
        return newton_schulz_triton(G, epsilon=epsilon)
    return zeropower_via_newtonschulz5(G, epsilon=epsilon)


@torch.compile(fullgraph=True)
def zeropower_via_newtonschulz5(G: Tensor, epsilon: float = 1e-7):
    """
    Newton-Schulz iteration to approximate the orthogonalization of X.
    """
    # Newton-Schulz constants
    ns_consts = [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]

    X = G.to(dtype=torch.bfloat16)
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + epsilon)

    for a, b, c in ns_consts:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X
