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
