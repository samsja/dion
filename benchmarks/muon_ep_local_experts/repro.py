"""
Runtime repro: FSDP+EP experts through Dion Muon. Pick the Dion under test with PYTHONPATH.

    torchrun --nproc-per-node 8 repro.py --ep 2
    torchrun --nproc-per-node 8 repro.py --ep 8   # pure EP
"""

import argparse
import time

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

import dion
import dion.muon
from dion import Muon

parser = argparse.ArgumentParser()
parser.add_argument("--ep", type=int, default=2)
parser.add_argument("--num-experts", type=int, default=64)
parser.add_argument("--dim", type=int, default=256)
parser.add_argument("--layers", type=int, default=4)
parser.add_argument("--steps", type=int, default=10)
args = parser.parse_args()

dist.init_process_group("nccl")
rank, world_size = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank % torch.cuda.device_count())
fsdp = world_size // args.ep
world_mesh = init_device_mesh("cuda", (fsdp, args.ep), mesh_dim_names=("fsdp", "ep"))


def log0(*msg):
    if rank == 0:
        print(*msg, flush=True)


log0(f"dion from {dion.__file__}, torch {torch.__version__}, fsdp={fsdp} ep={args.ep}")


class Experts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        gate_up = torch.randn(args.num_experts, 2 * args.dim, args.dim, device="cuda")
        down = torch.randn(args.num_experts, args.dim, args.dim, device="cuda")
        self.gate_up_proj = torch.nn.Parameter(distribute_tensor(gate_up, world_mesh["ep"], [Shard(0)]))
        self.down_proj = torch.nn.Parameter(distribute_tensor(down, world_mesh["ep"], [Shard(0)]))


layers = [Experts() for _ in range(args.layers)]
for layer in layers:
    fully_shard(layer, mesh=world_mesh["fsdp"])
params = [p for layer in layers for p in layer.parameters()]
partitions = {layer.gate_up_proj: (args.dim, args.dim) for layer in layers}

p = layers[0].gate_up_proj
log0(f"placements={p.placements} global={tuple(p.shape)} local={tuple(p.to_local().shape)}")
log0(f"baseline padded_local_size would be ceil({p.size(0)}/{fsdp}) = {-(-p.size(0) // fsdp)}")

ns_shapes = []


def recording_newton_schulz(X, epsilon):
    ns_shapes.append(tuple(X.shape))
    return dion.muon.zeropower_via_newtonschulz5(X, epsilon=epsilon)


batch_calls = []
original_batch_async = dion.muon.muon_update_batch_async


def recording_batch_async(**kwargs):
    batch_calls.append(
        dict(
            batch=len(kwargs["X"]),
            shard_dim=kwargs.get("shard_dim"),
            is_local=kwargs.get("is_local", "n/a"),
            local=tuple(kwargs["X"][0].to_local().shape),
        )
    )
    return original_batch_async(**kwargs)


dion.muon.muon_update_batch_async = recording_batch_async

optimizer = Muon(
    params=[dict(params=params, algorithm="muon", distributed_mesh_name="fsdp")],
    fsdp_mesh_dim=0,
    world_mesh=world_mesh,
    lr=0.02,
    mu=0.95,
    weight_decay=0.01,
    adjust_lr="rms_norm",
    newton_schulz_func=recording_newton_schulz,
    matrix_partitions=partitions,
)

torch.cuda.reset_peak_memory_stats()
times = []
for step in range(args.steps):
    for param in params:
        param.grad = DTensor.from_local(
            torch.randn_like(param.to_local()),
            param.device_mesh,
            param.placements,
            shape=param.shape,
            stride=param.stride(),
        )
    ns_shapes.clear()
    batch_calls.clear()
    torch.cuda.synchronize()
    start = time.perf_counter()
    optimizer.step()
    torch.cuda.synchronize()
    times.append(time.perf_counter() - start)
    if step == 0:
        log0(f"batches: {batch_calls}")
        log0(f"newton-schulz input shapes: {sorted(set(ns_shapes))} ({len(ns_shapes)} calls)")

steady = sorted(times[2:]) if len(times) > 2 else times
log0(f"step time ms: median {1e3 * steady[len(steady) // 2]:.2f} min {1e3 * steady[0]:.2f} max {1e3 * steady[-1]:.2f}")
log0(f"peak memory GiB: {torch.cuda.max_memory_allocated() / 2**30:.3f}")
dist.destroy_process_group()
