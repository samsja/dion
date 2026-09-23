"""
Extra coverage for uneven FSDP shards in Muon (PR #4, "fix(muon): support uneven FSDP shards").

Complements tests/test_muon_matrix_partitions_distributed.py with:
  * batches holding several real parameters (the optimizer batches world_size
    same-shape matrices per all-to-all round and pads short batches with dummies)
  * uneven shards on 3D tensors, sharded on every dim, with and without flatten
  * nesterov momentum and different adjust_lr modes
  * world_size=3 runs, where torch.chunk (DTensor Shard layout) and
    torch.tensor_split genuinely disagree on uneven splits

Run with 2 GPUs:

    torchrun --nproc-per-node 2 -m pytest tests/test_muon_uneven_fsdp_shards_distributed.py -q

Run the world_size=3 tests on 2 GPUs (two ranks share GPU 0):

    torchrun --nproc-per-node 3 -m pytest tests/test_muon_uneven_fsdp_shards_distributed.py -q
"""

import os

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

# This module intentionally compiles several matrix shapes in one process.
torch._dynamo.config.recompile_limit = 64

from dion import Muon

if "RANK" not in os.environ:
    pytest.skip("requires torchrun", allow_module_level=True)

WORLD_SIZE = int(os.environ["WORLD_SIZE"])


@pytest.fixture(scope="module")
def mesh():
    dist.init_process_group("nccl")
    # Allow more ranks than GPUs so world_size=3 runs on a 2-GPU box.
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    device_mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    yield device_mesh
    dist.destroy_process_group()


def make_muon(mesh, params, matrix_partitions=None, **overrides):
    kwargs = dict(
        lr=0.02,
        mu=0.95,
        weight_decay=0.01,
        adjust_lr="rms_norm",
    )
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
    kwargs = dict(
        lr=0.02,
        mu=0.95,
        weight_decay=0.01,
        adjust_lr="rms_norm",
    )
    kwargs.update(overrides)
    return Muon(
        params=[dict(params=params, algorithm="muon")],
        fsdp_mesh_dim=0,
        world_mesh=None,
        matrix_partitions=matrix_partitions,
        **kwargs,
    )


def train_sharded_and_local(mesh, shapes, shard_dim, partitions=None, steps=3, seed=0, **overrides):
    """
    Optimize FSDP-sharded parameters and identical unsharded parameters on the same
    gradients, then return both results. `shapes` may be a single shape or a list;
    passing several shapes of equal size exercises the world_size-sized batches
    (and the dummy padding of a short final batch) in one optimizer step.
    """
    if isinstance(shapes, torch.Size) or isinstance(shapes[0], int):
        shapes = [tuple(shapes)]
    torch.manual_seed(seed)
    placement = [Shard(shard_dim)]
    weights = [torch.randn(shape, device="cuda") for shape in shapes]
    sharded = [torch.nn.Parameter(distribute_tensor(w.clone(), mesh, placement)) for w in weights]
    local = [torch.nn.Parameter(w.clone()) for w in weights]

    sharded_partitions = {p: partitions for p in sharded} if partitions is not None else None
    local_partitions = {p: partitions for p in local} if partitions is not None else None
    sharded_optimizer = make_muon(mesh, sharded, sharded_partitions, **overrides)
    local_optimizer = make_local_muon(local, local_partitions, **overrides)

    for _ in range(steps):
        gradients = [torch.randn(w.shape, device="cuda") for w in weights]
        for parameter, gradient in zip(sharded, gradients):
            parameter.grad = distribute_tensor(gradient, mesh, placement)
        for parameter, gradient in zip(local, gradients):
            parameter.grad = gradient.clone()
        sharded_optimizer.step()
        local_optimizer.step()

    return [p.detach().full_tensor() for p in sharded], [p.detach() for p in local]


def assert_sharded_matches_local(mesh, shapes, shard_dim, **kwargs):
    sharded, expected = train_sharded_and_local(mesh, shapes, shard_dim, **kwargs)
    assert len(sharded) == len(expected)
    for got, want in zip(sharded, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# world_size = 2
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num_params", [2, 3])
def test_batch_of_uneven_shards_matches_unsharded_muon(mesh, num_params):
    # world_size-sized batch fills the all-to-all with real matrices only;
    # num_params=3 leaves a batch of one real matrix plus one pad_batch dummy.
    shapes = [(3, 8)] * num_params
    assert_sharded_matches_local(mesh, shapes, shard_dim=0)


@pytest.mark.parametrize(
    "shape,shard_dim",
    [
        ((3, 4, 8), 0),  # uneven across the batch/expert dimension
        ((2, 3, 8), 1),  # uneven across rows, whole experts per rank
        ((2, 8, 3), 2),  # uneven across columns
    ],
)
def test_uneven_shards_3d_match_unsharded_muon(mesh, shape, shard_dim):
    assert_sharded_matches_local(mesh, shape, shard_dim)


def test_uneven_shards_3d_flattened_match_unsharded_muon(mesh):
    # flatten=True reshapes (3, 2, 4) to a (3, 8) matrix; dim 0 stays uneven
    assert_sharded_matches_local(mesh, (3, 2, 4), shard_dim=0, flatten=True)


def test_uneven_shards_nesterov_match_unsharded_muon(mesh):
    assert_sharded_matches_local(mesh, (3, 8), shard_dim=0, nesterov=True)


@pytest.mark.parametrize("adjust_lr", ["spectral_norm", "keller_muon", None])
def test_uneven_shards_adjust_lr_match_unsharded_muon(mesh, adjust_lr):
    assert_sharded_matches_local(mesh, (3, 8), shard_dim=0, adjust_lr=adjust_lr)


def test_uneven_partitioned_shards_sharded_off_partition_dim(mesh):
    # 3D, rows (the partitioned dim) replicated on every rank; shard the uneven dim 0
    assert_sharded_matches_local(mesh, (3, 4, 8), shard_dim=0, partitions=(1, 3))


# ---------------------------------------------------------------------------
# world_size = 3 (run on 2 GPUs; two ranks share a device)
#
# DTensor Shard splits with torch.chunk: for N=4 rows on 3 ranks the local
# shards are [2, 2, 0], while torch.tensor_split would give [2, 1, 1].
# ---------------------------------------------------------------------------

needs_world3 = pytest.mark.skipif(WORLD_SIZE != 3, reason="requires torchrun --nproc-per-node 3")


@needs_world3
@pytest.mark.parametrize(
    "shape,shard_dim",
    [
        ((4, 8), 0),  # chunk layout [2, 2, 0]: uneven with a trailing empty shard
        ((5, 8), 0),  # chunk layout [2, 2, 1]: every rank holds rows
        ((2, 8), 0),  # chunk layout [1, 1, 0]
        ((8, 3), 1),  # uneven columns
    ],
)
def test_world3_uneven_shards_match_unsharded_muon(mesh, shape, shard_dim):
    assert_sharded_matches_local(mesh, shape, shard_dim)


@needs_world3
def test_world3_uneven_partitioned_rows_match_unsharded_muon(mesh):
    # chunk layout [2, 2, 1] agrees with tensor_split: partitioned rows stay aligned
    assert_sharded_matches_local(mesh, (5, 8), shard_dim=0, partitions=(2, 3))


@needs_world3
def test_world3_empty_shard_partitioned_rows_match_unsharded_muon(mesh):
    # chunk layout [1, 1, 0] with a partition boundary at every row
    assert_sharded_matches_local(mesh, (2, 8), shard_dim=0, partitions=(1, 1))


@needs_world3
def test_world3_uneven_partition_boundary_inside_a_shard(mesh):
    # N=4 rows on 3 ranks: chunk gives local shards [2, 2, 0], so rank 1 holds
    # rows 2 and 3, which sit in DIFFERENT partitions with different adjusted
    # learning rates. tensor_split(row_lr, 3) gives rank 1 a single lr entry,
    # so row 3 is updated with row 2's learning rate.
    assert_sharded_matches_local(
        mesh, (4, 8), shard_dim=0, partitions=(3, 1), adjust_lr="spectral_norm"
    )


@needs_world3
def test_world3_uneven_partitioned_rows_lr_count_mismatch(mesh):
    # N=7 rows on 3 ranks: chunk gives [3, 3, 1] but tensor_split(row_lr, 3)
    # gives [3, 2, 2], so rank 1 receives 2 learning rates for its 3 rows.
    assert_sharded_matches_local(
        mesh, (7, 8), shard_dim=0, partitions=(3, 4), adjust_lr="spectral_norm"
    )
