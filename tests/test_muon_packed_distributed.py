"""
Packed Muon on FSDP-sharded matrices against the batched path. Run with:

    torchrun --nproc-per-node 2 -m pytest tests/test_muon_packed_distributed.py -q

The HSDP case needs at least 4 ranks.
"""

import os

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor

import dion.muon_packed
from dion import Muon
from dion.muon import zeropower_via_newtonschulz5

if "RANK" not in os.environ:
    pytest.skip("requires torchrun", allow_module_level=True)


@pytest.fixture(autouse=True, scope="module")
def _raise_recompile_limit():
    with torch._dynamo.config.patch(recompile_limit=64):
        yield


@pytest.fixture(scope="module")
def world_size():
    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank())
    yield dist.get_world_size()
    dist.destroy_process_group()


# (group options, [(shape, shard dim or None for replicated, partitions, dtype)])
GROUPS = [
    (
        dict(),
        [
            ((7, 12), 0, None, torch.float32),
            ((1, 8), 0, None, torch.float32),
            ((16, 32), 0, None, torch.float32),
            ((16, 32), 0, None, torch.float32),
            ((16, 32), 0, None, torch.float32),
            ((16, 32), 0, None, torch.bfloat16),
            ((4, 6, 8), 0, None, torch.float32),
        ],
    ),
    (
        dict(nesterov=False, lr=0.01, weight_decay=0.0),
        [
            ((32, 16), 0, None, torch.float32),
            ((20, 16), 0, (4, 4, 4, 4, 4), torch.float32),
            ((15, 16), 0, (5, 5, 5), torch.float32),
            ((12, 9), 1, (4, 8), torch.float32),
            ((6, 10), None, None, torch.float32),
        ],
    ),
    (dict(lr=0.03), [((8, 16), 0, None, torch.float32)] * 17),
    (dict(lr=0.04), [((10, 12), 0, None, torch.float32)]),
]
MATRICES = [matrix for _, matrices in GROUPS for matrix in matrices]
PACKED_MATRICES = 28


def make_parameters(mesh):
    generator = torch.Generator().manual_seed(0)
    params = []
    for shape, shard_dim, _, dtype in MATRICES:
        placement = Replicate() if shard_dim is None else Shard(shard_dim)
        placements = [Replicate()] * (mesh.ndim - 1) + [placement]
        full = torch.randn(shape, generator=generator).to(device="cuda", dtype=dtype)
        params.append(torch.nn.Parameter(distribute_tensor(full, mesh, placements)))
    return params


def make_eager_muon(
    params, mesh, newton_schulz_func=zeropower_via_newtonschulz5._torchdynamo_orig_callable
):
    groups, start = [], 0
    for options, matrices in GROUPS:
        groups.append(dict(options, params=params[start : start + len(matrices)]))
        start += len(matrices)
    return Muon(
        params=groups,
        matrix_partitions={
            p: parts for p, (_, _, parts, _) in zip(params, MATRICES) if parts is not None
        },
        lr=0.02,
        mu=0.95,
        nesterov=True,
        weight_decay=0.1,
        adjust_lr="rms_norm",
        distributed_mesh=mesh["dp_shard"],
        world_mesh=mesh,
        fsdp_mesh_dim=mesh.ndim - 1,
        newton_schulz_func=newton_schulz_func,
    )


@pytest.mark.parametrize("matrices_per_rank", [1, 1000])
@pytest.mark.parametrize("replicas", [None, 1, 2])
def test_packed_matches_batched_bitwise(world_size, replicas, matrices_per_rank, monkeypatch):
    """
    With eager Newton-Schulz, packing whole matrices onto owner ranks gives the batched path's
    weights and momentum bitwise: uneven and empty shards, per-head partitioned rows split
    across shards, a column-sharded partitioned matrix, several shape, dtype and parameter
    groups, more matrices of one shape than one fixed-length list holds, a rank owning
    nothing, next to expert-like and replicated matrices left on the batched path, on FSDP
    and HSDP meshes, with small exchange buckets and with one bucket per group.
    """
    monkeypatch.setattr(dion.muon_packed, "EXCHANGE_MATRICES_PER_RANK", matrices_per_rank)
    if replicas is None:
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp_shard",))
    else:
        if world_size % replicas != 0 or world_size // replicas < 2:
            pytest.skip(f"needs a multiple of {2 * replicas} ranks")
        mesh = init_device_mesh(
            "cuda",
            (replicas, world_size // replicas),
            mesh_dim_names=("dp_replicate", "dp_shard"),
        )
    packed_params, batched_params = make_parameters(mesh), make_parameters(mesh)
    start = [p.detach().to_local().clone() for p in packed_params]
    packed = make_eager_muon(packed_params, mesh)
    batched = make_eager_muon(batched_params, mesh)
    monkeypatch.setattr(batched, "_pack_sharded_matrices", False)

    generator = torch.Generator().manual_seed(1)
    for _ in range(3):
        for packed_param, batched_param in zip(packed_params, batched_params):
            grad = torch.randn(packed_param.shape, generator=generator)
            grad = grad.to(device="cuda", dtype=packed_param.dtype)
            packed_param.grad = distribute_tensor(grad, mesh, packed_param.placements)
            batched_param.grad = distribute_tensor(grad, mesh, batched_param.placements)
        packed.step()
        batched.step()

    assert sum(len(plan.owners) for _, plan in packed._packed_plans.values()) == PACKED_MATRICES
    assert not batched._packed_plans
    for packed_param, batched_param, before in zip(packed_params, batched_params, start):
        assert set(packed.state[packed_param]) == set(batched.state[batched_param]) == {"momentum"}
        torch.testing.assert_close(
            packed_param.to_local(), batched_param.to_local(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            packed.state[packed_param]["momentum"].to_local(),
            batched.state[batched_param]["momentum"].to_local(),
            rtol=0,
            atol=0,
        )
        if before.numel():
            assert not torch.equal(packed_param.to_local(), before)


def float32_newton_schulz(G, epsilon):
    return zeropower_via_newtonschulz5._torchdynamo_orig_callable(G, epsilon).float()


def test_packed_tolerates_float32_newton_schulz(world_size, monkeypatch):
    """
    A custom Newton-Schulz function may return float32. The packed path casts its result to the
    update's bfloat16 before the return exchange, so it stays within bfloat16 rounding of the
    batched path, which applies the float32 result.
    """
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp_shard",))
    generator = torch.Generator().manual_seed(0)
    shapes = [(7, 12), (1, 8), (16, 32), (16, 32), (12, 9)]
    weights = [torch.randn(shape, generator=generator).cuda() for shape in shapes]
    grads = [torch.randn(shape, generator=generator).cuda() for shape in shapes]

    def train(pack):
        params = [
            torch.nn.Parameter(distribute_tensor(w.clone(), mesh, [Shard(0)])) for w in weights
        ]
        muon = Muon(
            params=[dict(params=params)],
            lr=0.02,
            mu=0.95,
            weight_decay=0.1,
            adjust_lr="rms_norm",
            distributed_mesh=mesh,
            world_mesh=mesh,
            fsdp_mesh_dim=0,
            newton_schulz_func=float32_newton_schulz,
        )
        monkeypatch.setattr(muon, "_pack_sharded_matrices", pack)
        for p, g in zip(params, grads):
            p.grad = distribute_tensor(g, mesh, [Shard(0)])
        muon.step()
        return [p.to_local() for p in params]

    for packed, batched in zip(train(True), train(False)):
        torch.testing.assert_close(packed, batched, rtol=1e-2, atol=1e-3)
