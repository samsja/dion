import gc
import weakref

import torch
from torch._dynamo.eval_frame import _debug_get_cache_entry_list

from dion import Muon
from dion.muon import (
    muon_update_post_orthogonalize,
    muon_update_post_orthogonalize_packed,
    muon_update_post_orthogonalize_partitioned,
    muon_update_pre_orthogonalize,
    muon_update_pre_orthogonalize_packed,
)
from dion.muon_packed import chunk_sizes, make_plan

# (local shard shape, number of layers holding it, partitions): one rank's shards of a deep
# model's interleaved dense matrices at a large shard group, so some shards have 0 or 1 rows,
# with the fan-in scaled down
DENSE_SHARDS = [
    ((1, 32), 56, None),
    ((2, 32), 42, None),
    ((16, 32), 28, None),
    ((28, 32), 165, None),
    ((56, 16), 110, None),
    ((56, 40), 42, None),
    ((56, 64), 14, None),
    ((56, 128), 1, None),
    ((80, 8), 42, None),
    ((80, 32), 126, (16,) * 5),
    ((80, 32), 42, None),
    ((112, 32), 14, None),
    ((224, 32), 2, None),
]


def test_chunk_sizes_follow_torch_chunk():
    for size in range(0, 20):
        for world_size in range(1, 9):
            expected = [len(c) for c in torch.arange(size).chunk(world_size)]
            expected += [0] * (world_size - len(expected))
            assert chunk_sizes(size, world_size) == tuple(expected)


def test_plan_balances_owners_by_newton_schulz_cost():
    params = [torch.empty(16384, 16384, device="meta")] + [torch.empty(64, 64, device="meta")] * 6
    plan = make_plan(params, [0] * len(params), [None] * len(params), world_size=2)
    assert plan.owners[0] == 0
    assert set(plan.owners[1:]) == {1}
    assert plan.owned_by(2) == ()


def test_plan_cache_neither_keeps_tensors_alive_nor_reuses_plans_of_dead_ones():
    kept = torch.nn.Parameter(torch.zeros(8, 4))
    muon = Muon(params=[kept], fsdp_mesh_dim=0, world_mesh=None)
    params = [torch.nn.Parameter(torch.zeros(8, 4))]
    plan = muon._get_packed_plan(params, [0], [None], world_size=2)
    assert muon._get_packed_plan(params, [0], [None], world_size=2) is plan
    dead = weakref.ref(params[0])
    del params
    gc.collect()
    assert dead() is None
    replacement = [torch.nn.Parameter(torch.zeros(6, 4))]
    replacement_plan = muon._get_packed_plan(replacement, [0], [None], world_size=2)
    assert replacement_plan.shard_sizes == ((3, 3),)
    assert len(muon._packed_plans) == 1


def test_packed_updates_compile_few_graphs_for_many_shapes():
    """
    The muon_update_* list functions are compiled with fullgraph=True, so a function past the
    recompile limit fails the step. The packed updates of many interleaved shapes, with rows of
    1, per-head partitions and updates viewed from one buffer, must compile a few graphs per
    function, as same-shape batches do.
    """
    torch._dynamo.reset()
    layout = [
        (shape, parts)
        for layer in range(max(count for _, count, _ in DENSE_SHARDS))
        for shape, count, parts in DENSE_SHARDS
        if layer < count
    ]
    X = [torch.randn(shape, device="cuda") for shape, _ in layout]
    M = [torch.zeros(shape, device="cuda") for shape, _ in layout]
    partitions = [parts for _, parts in layout]
    lr, momentum, weight_decay = torch.tensor(0.01), torch.tensor(0.95), torch.tensor(0.1)
    with torch._dynamo.config.patch(recompile_limit=64):
        for _ in range(3):
            U = muon_update_pre_orthogonalize_packed(
                [torch.randn_like(x) for x in X], M, momentum, nesterov=True, length=16
            )
            returned = torch.cat([u.reshape(-1) for u in U])
            updates = [
                piece.view(u.shape)
                for piece, u in zip(torch.split(returned, [u.numel() for u in U]), U)
            ]
            muon_update_post_orthogonalize_packed(
                X, updates, partitions, lr, weight_decay, "rms_norm", [None] * len(X), 16
            )
    functions = (
        muon_update_pre_orthogonalize,
        muon_update_post_orthogonalize,
        muon_update_post_orthogonalize_partitioned,
    )
    graphs = {
        fn.__name__: len(_debug_get_cache_entry_list(fn._torchdynamo_orig_callable.__code__))
        for fn in functions
    }
    assert all(1 <= count <= 6 for count in graphs.values()), graphs
