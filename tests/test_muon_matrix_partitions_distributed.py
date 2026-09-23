"""
Matrix partitions under FSDP-style sharding. Run with:

    torchrun --nproc-per-node 2 -m pytest tests/test_muon_matrix_partitions_distributed.py -q
"""

import os

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor


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
