"""
CPU coverage for uneven FSDP shards at world_size >= 3 (PR #4 follow-up tests).

Why this file exists:
  * DTensor Shard splits uneven tensors with torch.chunk, e.g. 4 rows on 3 ranks
    give local shards [2, 2, 0]. Exercising that needs world_size >= 3.
  * NCCL refuses two ranks on one GPU ("Duplicate GPU detected"), so a 2-GPU box
    cannot run world_size=3 CUDA tests.
  * gloo runs on CPU with any world size but does not implement all_to_all, the
    one collective Muon's sharded path needs.

So these tests run the real optimizer on a CPU gloo mesh and emulate ONLY
dist.all_to_all with all_gather (which gloo supports). DTensor sharding,
padding/trim, Newton-Schulz, and the per-row learning-rate logic all run the
production code.

Run with:

    torchrun --nproc-per-node 3 -m pytest tests/test_muon_uneven_fsdp_shards_cpu_distributed.py -q
"""

import contextlib
import os

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

@pytest.fixture(autouse=True, scope="module")
def _raise_recompile_limit():
    # The muon_update_* helpers are @torch.compile(fullgraph=True), so every new
    # matrix shape in this module recompiles them and fullgraph turns dynamo's
    # recompile-limit fallback into a hard error. Scoped patch, restored on teardown.
    with torch._dynamo.config.patch(recompile_limit=64):
        yield


from dion import Muon

if "RANK" not in os.environ:
    pytest.skip("requires torchrun", allow_module_level=True)

WORLD_SIZE = int(os.environ["WORLD_SIZE"])


@pytest.fixture(scope="module")
def mesh():
    dist.init_process_group("gloo")
    yield init_device_mesh("cpu", (dist.get_world_size(),))
    dist.destroy_process_group()


class _SyncedWork:
    """Stand-in for the async Work handle: the exchange already happened."""

    def wait(self):
        pass


def _all_to_all_via_all_gather(output_list, input_list, group=None, async_op=False):
    """
    Emulate dist.all_to_all with all_gather, which gloo supports.
    output_list[j] on this rank must receive input_list[rank] from rank j.
    """
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    gathered = [[torch.empty_like(t) for t in input_list] for _ in range(world_size)]
    for index, tensor in enumerate(input_list):
        dist.all_gather([g[index] for g in gathered], tensor.contiguous(), group=group)
    for source in range(world_size):
        output_list[source].copy_(gathered[source][rank])
    return _SyncedWork()


@contextlib.contextmanager
def emulated_all_to_all():
    """Patch the all_to_all Muon uses; restored before full_tensor() runs."""
    original = dist.all_to_all
    dist.all_to_all = _all_to_all_via_all_gather
    try:
        yield
    finally:
        dist.all_to_all = original


def make_muon(mesh, params, matrix_partitions=None, **overrides):
    kwargs = dict(lr=0.02, mu=0.95, weight_decay=0.01, adjust_lr="rms_norm")
    kwargs.update(overrides)
    return Muon(
        params=[dict(params=params, algorithm="muon")],
        fsdp_mesh_dim=0,
        world_mesh=mesh,
        distributed_mesh=mesh,
        matrix_partitions=matrix_partitions,
        **kwargs,
    )


def make_local_muon(params, matrix_partitions=None, **overrides):
    kwargs = dict(lr=0.02, mu=0.95, weight_decay=0.01, adjust_lr="rms_norm")
    kwargs.update(overrides)
    return Muon(
        params=[dict(params=params, algorithm="muon")],
        fsdp_mesh_dim=0,
        world_mesh=None,
        matrix_partitions=matrix_partitions,
        **kwargs,
    )


def train_sharded_and_local(mesh, shapes, shard_dim, partitions=None, steps=3, seed=0, **overrides):
    """Optimize FSDP-sharded and identical unsharded parameters on the same gradients."""
    if isinstance(shapes, torch.Size) or isinstance(shapes[0], int):
        shapes = [tuple(shapes)]
    torch.manual_seed(seed)
    placement = [Shard(shard_dim)]
    weights = [torch.randn(shape) for shape in shapes]
    sharded = [torch.nn.Parameter(distribute_tensor(w.clone(), mesh, placement)) for w in weights]
    local = [torch.nn.Parameter(w.clone()) for w in weights]

    sharded_partitions = {p: partitions for p in sharded} if partitions is not None else None
    local_partitions = {p: partitions for p in local} if partitions is not None else None
    sharded_optimizer = make_muon(mesh, sharded, sharded_partitions, **overrides)
    local_optimizer = make_local_muon(local, local_partitions, **overrides)

    for _ in range(steps):
        gradients = [torch.randn(w.shape) for w in weights]
        for parameter, gradient in zip(sharded, gradients):
            parameter.grad = distribute_tensor(gradient, mesh, placement)
        for parameter, gradient in zip(local, gradients):
            parameter.grad = gradient.clone()
        with emulated_all_to_all():
            sharded_optimizer.step()
        local_optimizer.step()

    return [p.detach().full_tensor() for p in sharded], [p.detach() for p in local]


def assert_sharded_matches_local(mesh, shapes, shard_dim, **kwargs):
    sharded, expected = train_sharded_and_local(mesh, shapes, shard_dim, **kwargs)
    assert len(sharded) == len(expected)
    for got, want in zip(sharded, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Controls: these layouts make torch.chunk and torch.tensor_split agree,
# or never split per-row learning rates at all. All should pass.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "shape,shard_dim",
    [
        ((4, 8), 0),  # chunk layout [2, 2, 0]: uneven with a trailing empty shard
        ((5, 8), 0),  # chunk layout [2, 2, 1]: every rank holds rows
        ((1, 8), 0),  # chunk layout [1, 0, 0]: two empty shards
        ((8, 3), 1),  # uneven columns, chunk layout [1, 1, 1]
        ((7, 8), 0),  # chunk layout [3, 3, 1]
    ],
)
def test_uneven_shards_match_unsharded_muon(mesh, shape, shard_dim):
    assert_sharded_matches_local(mesh, shape, shard_dim)


def test_batch_of_uneven_shards_matches_unsharded_muon(mesh):
    # Three same-shape parameters fill a whole world_size-sized batch.
    assert_sharded_matches_local(mesh, [(5, 8)] * WORLD_SIZE, shard_dim=0)


def test_uneven_shards_3d_match_unsharded_muon(mesh):
    # Uneven across rows with whole matrices per rank; partitioned dim untouched.
    assert_sharded_matches_local(mesh, (2, 5, 8), shard_dim=1, partitions=(2, 3))


def test_uneven_partitioned_rows_agreeing_layouts(mesh):
    # N=5 on 3 ranks: chunk and tensor_split both give [2, 2, 1].
    assert_sharded_matches_local(mesh, (5, 8), shard_dim=0, partitions=(2, 3))


def test_empty_shard_partitioned_rows(mesh):
    # N=2 on 3 ranks: chunk and tensor_split both give [1, 1, 0].
    assert_sharded_matches_local(mesh, (2, 8), shard_dim=0, partitions=(1, 1))


def test_uneven_shards_nesterov_and_spectral_norm(mesh):
    assert_sharded_matches_local(
        mesh, (5, 8), shard_dim=0, nesterov=True, adjust_lr="spectral_norm"
    )


# ---------------------------------------------------------------------------
# Regression tests: torch.chunk and torch.tensor_split disagree on these shapes
# (4 rows on 3 ranks: chunk [2, 2, 0] vs tensor_split [2, 1, 1]). The partitioned
# path once sliced per-row learning rates with tensor_split, so a rank whose
# uneven shard crossed a partition boundary applied the wrong learning rate or
# crashed on a broadcast mismatch. row_lr is now sliced with chunk offsets.
# ---------------------------------------------------------------------------

needs_world3 = pytest.mark.skipif(WORLD_SIZE != 3, reason="requires torchrun --nproc-per-node 3")


@needs_world3
def test_partition_boundary_inside_an_uneven_shard(mesh):
    # Rank 1 holds rows 2 and 3, which sit in different partitions with different
    # spectral_norm learning rates; row 3 must not use row 2's rate.
    assert_sharded_matches_local(
        mesh, (4, 8), shard_dim=0, partitions=(3, 1), adjust_lr="spectral_norm"
    )


@needs_world3
def test_partitioned_rows_learning_rate_count_mismatch(mesh):
    # chunk [3, 3, 1] vs tensor_split [3, 2, 2]: rank 1 needs 3 learning rates.
    assert_sharded_matches_local(
        mesh, (7, 8), shard_dim=0, partitions=(3, 4), adjust_lr="spectral_norm"
    )


@needs_world3
def test_3d_partition_boundary_inside_an_uneven_shard(mesh):
    # Rows live on dim 1 (ndim - 2): chunk [2, 2, 0] vs tensor_split [2, 1, 1].
    assert_sharded_matches_local(
        mesh, (2, 4, 8), shard_dim=1, partitions=(3, 1), adjust_lr="spectral_norm"
    )
