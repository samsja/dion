# Dion Optimizer in Pytorch FSDP2 
 
 
## 1. Building Parameter Groups


| Type                 | Example tensor(s)                            | Optimizer `algorithm` |
|----------------------|----------------------------------------------|--------------------------------|
| **Matrix** weights   | Transformer blocks: `W_qkv`, `W_out`, `W_ff` | `"dion"` / `"muon"`           |
| **Scalar/Vector** weights   | Token embeddings `W_te`, positional `W_pe`   | `"lion"` / `"adamw"`           |
| **LM head**          | Final projection: `lm_head.weight`           |`"lion"` / `"adamw"` (scaled LR)           |

### Example code snippets

```python
# Split params once – then reuse everywhere
matrix_params = list(model.transformer.h.parameters())
embed_params  = list(model.transformer.wte.parameters())
lm_head_params= list(model.lm_head.parameters())

param_groups = [
    dict(params=matrix_params),                         # Dion defaults
    dict(params=embed_params,                           # lion for scalar/vector params
         algorithm="lion", lr=base_lr, weight_decay=0),
    dict(params=lm_head_params,
         algorithm="lion",
         lr=base_lr / math.sqrt(hp.model_dim),          # scaled-down LR for LM-head
         weight_decay=0),
]
```
- Every group inherits defaults (`algorithm="dion"`) unless overridden.
- Supply `algorithm`, `lr`, `betas`, `weight_decay` only when they deviate.

## 2. Choosing Device Meshes

```python
# (a) build a 3-D mesh  (dp, fs, tp)
mesh = init_device_mesh(
    device_type="cuda",
    mesh_shape=(dp, fs, tp),
    mesh_dim_names=("dp", "fs", "tp")
)

opt = Dion(
    param_groups,
    replicate_mesh = mesh["dp"],      # data-parallel replica set
    outer_shard_mesh = mesh["fs"],    # outer (row) shard
    inner_shard_mesh = mesh["tp"],    # inner (col) shard (optional)
    replicate_mesh_grad_sync = True,
    rank_fraction = 0.5,            # r / d
    qr_method = "rcqr"
)
```

### "Flattened" Meshes
Sometimes, depending on infra, (Byron works on this)
```python 
opt = Dion(..., outer_shard_mesh = flat, inner_shard_mesh =)
```


## 4. Replicated Gradient Sync (FSDP-centric)

Dion can **skip the usual full-gradient all-reduce** and only synchronize the compressed *P / Q* factors (the *PowerSGD* trick— see [Vogels et al., 2019](https://arxiv.org/abs/1905.13727)). This is controlled by **`replicate_mesh_grad_sync`** and how you call `fully_shard()`.

| Option | What `fully_shard()` sees | Who syncs gradients? | When to use |
|----------------------------|----------------------------------|----------------------|-------------|
| **`True`**  |  It only sees a **1D mesh** (just the sharding axis). | **Dion** compresses & `all_reduce`s *P/Q* over `replicate_mesh`. FSDP does **no** DP sync. | **Preferred** when you want the PowerSGD-style comm savings. |
| **`False`** |  It also sees the DP axis and will reduce-scatter the full gradients. | **FSDP** performs the usual full-grad sync. Dion assumes grads are already identical. | Useful when you *really* need exact DP sums. |

In other words, if Dion is going to compress & all-reduce on its own (i.e., `replicate_mesh_grad_sync=True`), keep the data-parallel axis out of `fully_shard()` (1D mesh). Otherwise let FSDP see that axis and set `replicate_mesh_grad_sync=False`. 

### Example code snippets

```python
# ------------------------------------------------------------
#  Mode 1: Dion handles DP sync (compressed P/Q)  <-- recommended
# ------------------------------------------------------------
mesh = init_device_mesh("cuda", (dp, fs), ("dp", "fs"))

fully_shard(model, mesh=mesh["fs"]) 

opt = Dion(
    param_groups,
    replicate_mesh           = mesh["dp"],   # still give Dion the DP ranks
    outer_shard_mesh         = mesh["fs"], 
    replicate_mesh_grad_sync = True
)

# ------------------------------------------------------------
#  Mode 2: FSDP handles DP sync (classic full gradients)
# ------------------------------------------------------------
mesh = init_device_mesh("cuda", (dp, fs), ("dp", "fs"))

fully_shard(model, mesh=mesh["dp", "fs"])  # <- FSDP now *knows* about DP 

opt = Dion(
    param_groups,S
    replicate_mesh           = mesh["dp"],    
    outer_shard_mesh         = mesh["fs"], 
    replicate_mesh_grad_sync = False         # Dion expects synced grads
)
```

## 5. End-to-End Examples

### Pure DDP

```python!
dist.init_process_group("nccl")
model = DDP(model, device_ids=[local_rank])
opt   = Dion(param_groups,
             replicate_mesh = dist.group.WORLD,
             outer_shard_mesh = None,
             inner_shard_mesh = None,
             replicate_mesh_grad_sync = True)
```

### Hybrid-Sharded (FSDP2 | HSDP)
```python!
mesh = init_device_mesh("cuda", (dp, fs, tp), ("dp", "fs", "tp"))

fully_shard(model, mesh=mesh["dp", "fs", "tp"]) 

opt = Dion(param_groups,
           replicate_mesh = mesh["dp"],
           outer_shard_mesh = mesh["fs"],
           inner_shard_mesh = mesh["tp"],
           replicate_mesh_grad_sync = True)
```
 


## 6. Dion vs. Dion-Async

`dion_async.py` improves communication efficiency by:

- **Processing parameter groups in batches**, amortizing overhead across multiple tensors.
- **Splitting communication into `reduce-scatter` + `all-gather`** to better distribute the QR workload.
- **Overlapping communication with compute** by interleaving batches asynchronously, reducing idle time on slower networks.

---

| Feature                          | **`dion.py` (synchronous)**                                             | **`dion_async.py` (asynchronous)**                                                                 |
|----------------------------------|------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------- 
| **When to use**                  | • Fast interconnect (e.g., NVLink)<br>• Simpler to debug and profile   | • Network-constrained clusters<br>• High-latency environments<br>• Heavy compute (e.g. Triton, FlashAttention) |
| **Feature**            | More deterministic and profiling-friendly                              | Higher throughput in realistic multi-node or cloud environments                                    |

**Switching is drop-in**: just change the import and config:

```python
# In train.py or your config
from optimizers.dion_async import Dion as DionAsync
# Then set with the cli-arg:
--optimizer dion_async
```

# Dion Optimizer

This repository provides a preliminary implementation of the Dion optimizer (https://arxiv.org/pdf/2504.05295).

## 🔧 Requirements

This code is written for modern PyTorch (version 2.6 or newer) using DTensor-based parallelism. This includes FSDP2 with `fully_shard`and tensor parallelism (TP) with `parallelize_module`. Support for other distributed training APIs is not guaranteed.

## ⚡ Quick Start

Most of the relevant code is located in the `scripts/` directory.

```bash
cd scripts
pip install -r requirements.txt
```

To download the dataset, run:

```bash
python data/cached_fineweb10B.py 16
```

To quickly train a 160M model, run:

```bash
torchrun --standalone --nproc_per_node=8 train.py --config configs/dion_160m.yaml
``` 

After the training you should be able to reproduce the first plot in [validation curves for GPT-small](https://microsoft-research.wandb.io/t-gmagakyan/dion-exp/reports/Validation-curves-for-GPT-small--VmlldzoxNjk5OA?accessToken=52e6z4d18yfkewz1bawlkmwc2m91al9ssa7rpwvnx1f1xa66j15lr7x315wj2kys).


## 🚀 Accelerating Optimization step for lower ranks


After a few warmup iterations, the expensive QR decomposition can be replaced with **CQR**, leading to **2X** optimization step speedups.


> ⚠️ **Note:** Speedup for **rank fraction = 1.0** is still under development. For models **larger than 10B**, you may need to **further reduce the rank fraction** to see benefits.

To train the accelerated 160M model:
```bash
torchrun --standalone --nproc_per_node=8 train.py --config configs/dion_efficient_160m.yaml
```

After the training you should be able to reproduce the second plot in [validation curves for GPT-small](https://microsoft-research.wandb.io/t-gmagakyan/dion-exp/reports/Validation-curves-for-GPT-small--VmlldzoxNjk5OA?accessToken=52e6z4d18yfkewz1bawlkmwc2m91al9ssa7rpwvnx1f1xa66j15lr7x315wj2kys).


# Citation 

If you use Dion in your research, please cite:

```bash
@article{ahn2025dion,
  title={Dion: Distributed Orthonormalized Updates},
  author={Ahn, Kwangjun and Xu, Byron and Abreu, Natalie and Langford, John},
  journal={arXiv preprint: 2504.05295},
  year={2025}
}
``` 
---
