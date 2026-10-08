"""
Matrix partitions under FSDP-style sharding. Run with:

    torchrun --nproc-per-node 2 -m pytest tests/test_muon_matrix_partitions_distributed.py -q

The expert parallel tests need at least 4 ranks, and the HSDP tests 8:

    torchrun --nproc-per-node 8 -m pytest tests/test_muon_matrix_partitions_distributed.py -q
"""

import functools
import os

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.distributed.tensor.placement_types import _StridedShard


@pytest.fixture(autouse=True, scope="module")
def _raise_recompile_limit():
    # The muon_update_* helpers are @torch.compile(fullgraph=True): every new matrix
    # shape in this module's sweep recompiles them, and fullgraph turns dynamo's
    # recompile-limit fallback into a hard error. The sweep intentionally exceeds the
    # default limit of 8, so raise it here, scoped to this module, instead of mutating
    # the process-global config.
    with torch._dynamo.config.patch(recompile_limit=64):
        yield


from dion import Muon

if "RANK" not in os.environ:
    pytest.skip("requires torchrun", allow_module_level=True)


@pytest.fixture(scope="module")
def mesh():
    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank())
    device_mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    yield device_mesh
    dist.destroy_process_group()


def make_muon(mesh, params, matrix_partitions=None):
    return Muon(
        params=[dict(params=params, algorithm="muon")],
        fsdp_mesh_dim=0,
        world_mesh=mesh,
        distributed_mesh=mesh,
        lr=0.02,
        mu=0.95,
        weight_decay=0.01,
        adjust_lr="rms_norm",
        matrix_partitions=matrix_partitions,
    )


def make_local_muon(params, matrix_partitions=None):
    return Muon(
        params=[dict(params=params, algorithm="muon")],
        fsdp_mesh_dim=0,
        world_mesh=None,
        lr=0.02,
        mu=0.95,
        weight_decay=0.01,
        adjust_lr="rms_norm",
        matrix_partitions=matrix_partitions,
    )


def train_sharded_and_local(mesh, shape, shard_dim, partitions=None, steps=3, seed=0):
    """Compare uneven FSDP Muon updates with the same unsharded update."""
    torch.manual_seed(seed)
    placement = [Shard(shard_dim)]
    weight = torch.randn(shape, device="cuda")
    sharded = torch.nn.Parameter(distribute_tensor(weight.clone(), mesh, placement))
    local = torch.nn.Parameter(weight.clone())

    sharded_partitions = {sharded: partitions} if partitions is not None else None
    local_partitions = {local: partitions} if partitions is not None else None
    sharded_optimizer = make_muon(mesh, [sharded], sharded_partitions)
    local_optimizer = make_local_muon([local], local_partitions)

    for _ in range(steps):
        gradient = torch.randn(shape, device="cuda")
        sharded.grad = distribute_tensor(gradient, mesh, placement)
        local.grad = gradient.clone()
        sharded_optimizer.step()
        local_optimizer.step()

    return sharded.detach().full_tensor(), local.detach()


def train_fused_and_independent(mesh, shape, partitions, shard_dim, steps=3, seed=0):
    """
    Optimize a sharded parameter packing `partitions` along dim -2 and, on the same
    gradients, the independent sharded matrices it packs.
    """
    torch.manual_seed(seed)
    placements = [Shard(shard_dim)]
    weight = torch.randn(shape, device="cuda")

    fused = torch.nn.Parameter(distribute_tensor(weight, mesh, placements))
    independent = [
        torch.nn.Parameter(distribute_tensor(matrix.contiguous(), mesh, placements))
        for matrix in weight.split(partitions, dim=-2)
    ]

    fused_optimizer = make_muon(mesh, [fused], {fused: partitions})
    independent_optimizer = make_muon(mesh, independent)

    for _ in range(steps):
        gradient = torch.randn(shape, device="cuda")
        fused.grad = distribute_tensor(gradient, mesh, placements)
        for parameter, matrix in zip(independent, gradient.split(partitions, dim=-2)):
            parameter.grad = distribute_tensor(matrix.contiguous(), mesh, placements)
        fused_optimizer.step()
        independent_optimizer.step()

    expected = torch.cat([parameter.detach().full_tensor() for parameter in independent], dim=-2)
    return fused.detach().full_tensor(), expected


def test_partitions_sharded_across_the_partition_dimension(mesh):
    # Every rank holds part of both matrices, so a shard crosses a partition boundary
    fused, expected = train_fused_and_independent(mesh, (384, 128), (192, 192), shard_dim=0)
    torch.testing.assert_close(fused, expected, rtol=0, atol=0)


def test_unequal_partitions_sharded_across_the_partition_dimension(mesh):
    fused, expected = train_fused_and_independent(mesh, (384, 128), (256, 128), shard_dim=0)
    torch.testing.assert_close(fused, expected, rtol=0, atol=0)


def test_expert_partitions_sharded_across_the_expert_dimension(mesh):
    # Sharding on dim 0 leaves every rank with whole partitions along dim -2
    fused, expected = train_fused_and_independent(mesh, (4, 192, 64), (96, 96), shard_dim=0)
    torch.testing.assert_close(fused, expected, rtol=0, atol=0)


def test_expert_partitions_sharded_across_the_partition_dimension(mesh):
    fused, expected = train_fused_and_independent(mesh, (4, 192, 64), (96, 96), shard_dim=1)
    torch.testing.assert_close(fused, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "shape,shard_dim",
    [
        ((3, 8), 0),  # uneven non-empty shards
        ((1, 8), 0),  # an empty shard on rank 1
        ((8, 3), 1),  # uneven sharding along the input dimension
    ],
)
def test_uneven_shards_match_unsharded_muon(mesh, shape, shard_dim):
    sharded, expected = train_sharded_and_local(mesh, shape, shard_dim)
    torch.testing.assert_close(sharded, expected, rtol=0, atol=0)


def test_uneven_partitioned_shards_match_unsharded_muon(mesh):
    sharded, expected = train_sharded_and_local(mesh, (3, 8), 0, partitions=(1, 2))
    torch.testing.assert_close(sharded, expected, rtol=0, atol=0)


class Experts(torch.nn.Module):
    def __init__(self, weight, ep_mesh):
        super().__init__()
        self.weight = torch.nn.Parameter(distribute_tensor(weight, ep_mesh, [Shard(0)]))


def make_expert_muon(world_mesh, params, fsdp_mesh_dim, matrix_partitions=None, flatten=False):
    return Muon(
        params=[dict(params=params, algorithm="muon", distributed_mesh_name="fsdp")],
        fsdp_mesh_dim=fsdp_mesh_dim,
        world_mesh=world_mesh,
        lr=0.02,
        mu=0.95,
        weight_decay=0.01,
        adjust_lr="rms_norm",
        flatten=flatten,
        matrix_partitions=matrix_partitions,
    )


@functools.cache
def expert_mesh(replicate, ep):
    """World mesh shaped like prime-rl's expert layout, with the FSDP dim sized to fit."""
    world_size = dist.get_world_size()
    if world_size < 4 or world_size % (replicate * ep) != 0:
        pytest.skip(f"needs at least 4 ranks divisible by {replicate * ep}")
    fsdp = world_size // (replicate * ep)
    if replicate > 1 and fsdp < 2:
        pytest.skip("HSDP needs at least 8 ranks")
    if replicate == 1:
        return init_device_mesh("cuda", (fsdp, ep), mesh_dim_names=("fsdp", "ep"))
    return init_device_mesh(
        "cuda", (replicate, fsdp, ep), mesh_dim_names=("replicate", "fsdp", "ep")
    )


def shard_experts(world_mesh, weight):
    """EP-shard the experts, then FSDP-shard them the way prime-rl does."""
    experts = Experts(weight, world_mesh["ep"])
    if "replicate" in world_mesh.mesh_dim_names:
        fully_shard(experts, mesh=world_mesh["replicate", "fsdp"])
    else:
        fully_shard(experts, mesh=world_mesh["fsdp"])
    return experts.weight


def train_experts_and_references(
    shape, partitions=None, replicate=1, ep=2, fsdp_mesh_dim=0, steps=3, seed=0
):
    """
    Optimize FSDP+EP experts, a plain Muon on this rank's local experts, and a plain Muon
    on the full expert tensor, all on the same gradients.
    """
    world_mesh = expert_mesh(replicate, ep)
    torch.manual_seed(seed)
    weight = torch.randn(shape, device="cuda")
    sharded = shard_experts(world_mesh, weight.clone())
    local = torch.nn.Parameter(sharded.to_local().clone())
    full = torch.nn.Parameter(weight.clone())

    sharded_optimizer = make_expert_muon(
        world_mesh, [sharded], fsdp_mesh_dim, {sharded: partitions} if partitions else None
    )
    local_optimizer = make_local_muon([local], {local: partitions} if partitions else None)
    full_optimizer = make_local_muon([full], {full: partitions} if partitions else None)

    for step in range(steps):
        torch.manual_seed(seed + 1 + step + 100 * dist.get_rank())
        local_gradient = torch.randn_like(sharded.to_local())
        sharded.grad = DTensor.from_local(
            local_gradient,
            sharded.device_mesh,
            sharded.placements,
            shape=sharded.shape,
            stride=sharded.stride(),
        )
        local.grad = local_gradient.clone()
        full.grad = sharded.grad.full_tensor()
        sharded_optimizer.step()
        if local.numel() > 0:
            local_optimizer.step()
        full_optimizer.step()

    return sharded, local.detach(), full.detach()


def assert_experts_match(sharded, local, full):
    # A plain Muon on the same local experts runs the same Newton-Schulz batch shape
    torch.testing.assert_close(sharded.to_local(), local, rtol=0, atol=0)
    # The full tensor is a different batch shape, which compiles a different graph
    torch.testing.assert_close(sharded.detach().full_tensor(), full, rtol=1.6e-2, atol=1e-5)


def test_expert_placements_are_strided_fsdp_and_ep_shards(mesh):
    world_mesh = expert_mesh(replicate=1, ep=2)
    weight = shard_experts(world_mesh, torch.randn(16, 32, 16, device="cuda"))
    assert weight.placements == (_StridedShard(0, split_factor=2), Shard(0))


def test_hsdp_expert_placements_are_replicated_strided_fsdp_and_ep_shards(mesh):
    world_mesh = expert_mesh(replicate=2, ep=2)
    weight = shard_experts(world_mesh, torch.randn(16, 32, 16, device="cuda"))
    assert weight.placements == (Replicate(), _StridedShard(0, split_factor=2), Shard(0))


def test_fsdp_ep_experts_match_local_muon(mesh):
    assert_experts_match(*train_experts_and_references((16, 32, 16)))


def test_fsdp_ep_partitioned_experts_match_local_muon(mesh):
    assert_experts_match(*train_experts_and_references((16, 48, 16), partitions=(32, 16)))


def test_fsdp_ep_uneven_experts_match_local_muon(mesh):
    # Each EP group's experts split unevenly across its FSDP ranks
    num_experts = 2 * (dist.get_world_size() // 2 + 1)
    assert_experts_match(*train_experts_and_references((num_experts, 32, 16)))


def test_fsdp_ep_rank_without_experts_matches_local_muon(mesh):
    # Fewer experts per EP group than FSDP ranks leaves some ranks with no experts
    assert_experts_match(*train_experts_and_references((2, 32, 16)))


def test_pure_ep_experts_match_local_muon(mesh):
    assert_experts_match(*train_experts_and_references((16, 32, 16), ep=dist.get_world_size()))


def test_hsdp_ep_experts_match_local_muon(mesh):
    assert_experts_match(
        *train_experts_and_references((16, 32, 16), replicate=2, fsdp_mesh_dim=1)
    )


def test_flatten_experts_with_strided_fsdp_shard_raise(mesh):
    world_mesh = expert_mesh(replicate=1, ep=2)
    weight = shard_experts(world_mesh, torch.randn(16, 32, 16, device="cuda"))
    weight.grad = torch.zeros_like(weight)
    optimizer = make_expert_muon(world_mesh, [weight], fsdp_mesh_dim=0, flatten=True)
    with pytest.raises(NotImplementedError):
        optimizer.step()


def test_flatten_sharded_3d_matches_unsharded_muon(mesh):
    # With flatten, the leading dim is a matrix dim, so the shards still need the all-to-all
    shape = (2 * dist.get_world_size(), 8, 4)
    torch.manual_seed(0)
    weight = torch.randn(shape, device="cuda")
    sharded = torch.nn.Parameter(distribute_tensor(weight.clone(), mesh, [Shard(0)]))
    local = torch.nn.Parameter(weight.clone())
    muon_kwargs = dict(lr=0.02, mu=0.95, weight_decay=0.01, adjust_lr="rms_norm", flatten=True)
    sharded_optimizer = Muon(
        params=[dict(params=[sharded], algorithm="muon")],
        fsdp_mesh_dim=0,
        world_mesh=mesh,
        distributed_mesh=mesh,
        **muon_kwargs,
    )
    local_optimizer = Muon(
        params=[dict(params=[local], algorithm="muon")],
        fsdp_mesh_dim=0,
        world_mesh=None,
        **muon_kwargs,
    )
    for _ in range(3):
        gradient = torch.randn(shape, device="cuda")
        sharded.grad = distribute_tensor(gradient, mesh, [Shard(0)])
        local.grad = gradient.clone()
        sharded_optimizer.step()
        local_optimizer.step()
    torch.testing.assert_close(sharded.detach().full_tensor(), local.detach(), rtol=0, atol=0)


@functools.cache
def hsdp_meshes(misconfiguration=None):
    """A fresh HSDP mesh like prime-rl's, and a separately built 1D shard mesh for Muon."""
    world_size = dist.get_world_size()
    if world_size < 4 or world_size % 2 != 0:
        pytest.skip("needs an even number of at least 4 ranks")
    ranks = torch.arange(world_size).reshape(2, world_size // 2)
    hsdp_ranks, muon_ranks = ranks, ranks
    if misconfiguration == "mismatched_groups":
        # Muon's groups mix ranks from both replicas
        muon_ranks = ranks.T.reshape(2, world_size // 2)
    elif misconfiguration == "descending_mesh":
        hsdp_ranks = ranks.flip(-1)
    hsdp = DeviceMesh("cuda", hsdp_ranks, mesh_dim_names=("replicate", "shard"))
    shard = DeviceMesh("cuda", muon_ranks, mesh_dim_names=("replicate", "shard"))["shard"]
    return hsdp, shard


def hsdp_dense_layout(layout, shard_size):
    """Shape and partitions for each layout whose correctness depends on shard order."""
    rows = 4 * shard_size
    return {
        "even": ((rows, 16), None),
        # The short trailing shard is trimmed after the all-to-all
        "uneven": ((rows - 1, 16), None),
        # Partition boundaries are row indices
        "partitions": ((rows, 16), (rows // 2, rows // 2)),
        # Unequal partitions give each device's rows a different adjusted learning rate
        "row_lr": ((rows, 4), (3 * rows // 4, rows // 4)),
    }[layout]


def train_hsdp_dense(hsdp, shard_mesh, layout, steps=2):
    torch.manual_seed(0)
    shape, partitions = hsdp_dense_layout(layout, hsdp.size(1))
    placements = [Replicate(), Shard(0)]
    weight = torch.randn(shape, device="cuda")
    sharded = torch.nn.Parameter(distribute_tensor(weight.clone(), hsdp, placements))
    local = torch.nn.Parameter(weight.clone())
    sharded_optimizer = make_muon(
        shard_mesh, [sharded], {sharded: partitions} if partitions else None
    )
    local_optimizer = make_local_muon([local], {local: partitions} if partitions else None)
    for _ in range(steps):
        gradient = torch.randn(shape, device="cuda")
        sharded.grad = distribute_tensor(gradient, hsdp, placements)
        local.grad = gradient.clone()
        sharded_optimizer.step()
        local_optimizer.step()
    return sharded.detach().full_tensor(), local.detach()


HSDP_DENSE_LAYOUTS = ["even", "uneven", "partitions", "row_lr"]


@pytest.mark.parametrize("layout", HSDP_DENSE_LAYOUTS)
def test_hsdp_dense_with_matching_shard_ranks_matches_unsharded_muon(mesh, layout):
    # The HSDP mesh and Muon's mesh are built separately, so their groups differ
    sharded, expected = train_hsdp_dense(*hsdp_meshes(), layout)
    torch.testing.assert_close(sharded, expected, rtol=0, atol=0)


@pytest.mark.parametrize("misconfiguration", ["mismatched_groups", "descending_mesh"])
@pytest.mark.parametrize("layout", HSDP_DENSE_LAYOUTS)
def test_hsdp_dense_with_misconfigured_meshes_raises(mesh, layout, misconfiguration):
    with pytest.raises(ValueError, match="sharded over ranks"):
        train_hsdp_dense(*hsdp_meshes(misconfiguration), layout)


@pytest.mark.parametrize(
    "misconfiguration,layout",
    [
        ("mismatched_groups", "even"),
        # DTensor sizes uneven shards by mesh coordinate but places them by group rank
        ("descending_mesh", "uneven"),
    ],
)
def test_hsdp_dense_with_misconfigured_meshes_is_wrong_without_check(
    mesh, monkeypatch, misconfiguration, layout
):
    # Guards the check itself: each misconfiguration it rejects really corrupts the update
    monkeypatch.setattr(Muon, "_check_shard_group", staticmethod(lambda *args: None))
    sharded, expected = train_hsdp_dense(*hsdp_meshes(misconfiguration), layout)
    assert not torch.allclose(sharded, expected, rtol=1e-2, atol=1e-3)
