# Dion: Distributed Orthonormal Updates

This repository provides efficient implementations of Dion and Muon optimizers for distributed ML training.
 
* See our paper for more information on Dion: https://arxiv.org/pdf/2504.05295.
* See the original blog post on Muon: https://kellerjordan.github.io/posts/muon/

## 🔧 Requirements

This code is written for modern PyTorch (version 2.7 or newer) using DTensor-based parallelism. This includes FSDP2 with `fully_shard`and tensor parallelism (TP) with `parallelize_module`. Support for other distributed training APIs is not guaranteed.

## ⚡ Quick Start

Install dependences:
```bash
pip install -r requirements.txt
```

Download pretokenized FineWeb dataset:
```bash
python data/cached_fineweb10B.py 16
```

Example with training a 160M model:
```bash
torchrun --standalone --nproc_per_node=8 train.py --config configs/dion_160m.yaml
``` 

After the training you should be able to reproduce the first plot in [validation curves for GPT-small](https://microsoft-research.wandb.io/t-gmagakyan/dion-exp/reports/Validation-curves-for-GPT-small--VmlldzoxNjk5OA?accessToken=52e6z4d18yfkewz1bawlkmwc2m91al9ssa7rpwvnx1f1xa66j15lr7x315wj2kys).


## Building Parameter Groups

Unlike typical PyTorch optimizers (e.g. `Adam`/`AdamW`), Dion and Muon require separating your model's parameters into different groups. The Dion and Muon algorithms are only applicable to two-dimensional matrix weights. Other parameters are optimized using a different algorithm (Lion and AdamW are currently implemented) and may also use a different learning rate. The details of parameter grouping are dependent on model architecture and implementation. Therefore, we leave it up to you to categorize your model's parameters and create the necessary parameter groups.

* In transformer models and many other neural networks, most parameters are `nn.Linear` layers with two-dimensional weight matrices. These parameters should use Dion or Muon. A shape-dependent learning rate scale factor will be automatically applied for each matrix.
* Biases in `nn.Linear` layers (if used) are one-dimensional vectors, which must be placed into a separate parameter group from the weight matrices. Use Lion or AdamW.
* Normalization layers (`nn.LayerNorm`, `nn.RMSNorm`) may contain a vector of learnable weights. Use Lion or AdamW.
* Embedding layers (`nn.Embedding`) are stored as 2D tensors, should be treated as a collection of 1D vectors using Lion or AdamW. (Using Dion here will run without error, but will give poor performance.)
* Unembedding layers (e.g. LM head) are typically implemented as a `nn.Linear` layer, but shoud also be treated as a collection of 1D vectors. Furthermore, they should use a smaller *scaled learning rate*. It is very important to manually identify this layer and place it into its own parameter group, as it is otherwise indistinguishable from weight matrices!
* Convolution layers typically use parameter tensors with 3+ dimensions. These are currently not supported for Dion. Support for convolution layers in Muon is experimental, and can be enabled by passing `flatten=True` to automatically flatten them to 2D matrices when computing the optimizer update.

We summarize the above in this table. Let `d_in` be the input dimension of the unembedding layer. In transformer language models, this is the base dimension of the model.

| Type          | Example parameters                          | Optimizer `algorithm` | Learning rate `lr`     |
|---------------|---------------------------------------------|-----------------------|------------------------|
| Weight matrix | `nn.Linear.weight`                          | `"dion"` / `"muon"`   | `lr`                   |
| Bias vector   | `nn.Linear.bias`                            | `"lion"` / `"adamw"`  | `lr`                   |
| Normalization | `nn.LayerNorm.weight`, `nn.LayerNorm.bias`  | `"lion"` / `"adamw"`  | `lr`                   |
| Embedding     | `nn.Embedding.weight`                       | `"lion"` / `"adamw"`  | `lr`                   |
| Unembedding   | `nn.Linear.weight` (must identify manually) | `"lion"` / `"adamw"`  | `lr / math.sqrt(d_in)` |

It is permissible to place biases, embeddings, and normalization parameters into a single parameter group if they use the same hyperparameters.

### Example code

```python
class TransformerModel(nn.Module):
    embedding = nn.Embedding(vocab_dim, model_dim)
    blocks = nn.ModuleList([TransformerBlock(...) for _ in range(10)])
    lm_head = nn.Linear(model_dim, vocab_dim)

model = TransformerModel()
matrix_params = list(p for p in model.blocks.parameters() if p.ndim == 2)
vector_params = list(p for p in model.blocks.parameters() if p.ndim != 2)
embed_params  = list(model.embedding.parameters())
lm_head_params= list(model.lm_head.parameters())

param_groups = [
    dict(params=matrix_params),  # defaults to "dion" algorithm
    dict(params=vector_params, algorithm="lion"),
    dict(params=embed_params, algorithm="lion"),
    dict(params=lm_head_params, algorithm="lion", lr=lr / math.sqrt(model_dim))
]

optimizer = Dion(
    param_groups,
    lr=lr,  # used for all param groups except for lm_head_params
    weight_decay=0.1,  # default setting for all param groups
    ...
)
```

Additional hyperparameters may be specified on a per-parameter-group basis to override the defaults. For example, we may set the weight decay to 0 for only the embedding and unembedding parameters by modifying the above example:
```python
param_groups = [
    dict(params=matrix_params),
    dict(params=vector_params, algorithm="lion"),
    dict(params=embed_params, algorithm="lion", weight_decay=0),
    dict(params=lm_head_params, algorithm="lion", lr=lr / math.sqrt(model_dim), weight_decay=0)
]
```

## Distributed Training Configuration

In order for Dion to work, it must know about the parallelization scheme for training your model. This is done by passing in `DeviceMesh` objects when constructing the optimizer.

Dion supports up to two sharded mesh dimensions and any number of data-parallel replicated mesh dimensions. The `outer_shard_mesh` will be unsharded before entering Dion's orthogonalization sub-routine, while the `inner_shard_mesh` remains sharded throughout orthogonalization. The `inner_shard_mesh` is more communication-intensive and works best with intra-node tensor parallelism. Both sharding meshes must be one-dimensional.

Unused meshes may be omitted or given as `None`. If only one sharding dimension is used (e.g. only FSDP without TP), we recommend providing it as the `outer_shard_mesh`.

```python
# Example with a 3D mesh
mesh = init_device_mesh(
    device_type="cuda",
    mesh_shape=(dp_size, fs_size, tp_size),
    mesh_dim_names=("dp", "fs", "tp")
)

optimizer = Dion(
    param_groups,
    replicate_mesh = mesh["dp"],       # Replicated data parallel
    outer_shard_mesh = mesh["fs"],     # Sharded data parallel
    inner_shard_mesh = mesh["tp"],     # Tensor parallel
    ...
)
```

### Flattened Meshes

When more advanced parallelism strategies are used (such as context parallel or expert parallel), it is common for multiple mesh dimensions to be "flattened" into a 1D sub-mesh for sharding. In this scenario, the flattened mesh needs to be given to Dion.

```python
mesh = init_device_mesh(
    device_type="cuda",
    mesh_shape=(dp_size, cp_size, tp_size),
    mesh_dim_names=("dp", "cp", "tp")
)

# FSDP sharding applied across combined DP and CP meshes
fs_mesh = mesh["dp", "cp"]._flatten()
fully_shard(model, mesh=fs_mesh)

optimizer = Dion(
    param_groups,
    replicate_mesh = None,             # Replicated data parallel
    outer_shard_mesh = fs_mesh,        # Sharded data parallel
    inner_shard_mesh = mesh["tp"],     # Tensor parallel
    ...
)
```

### Device Mesh for Muon

Muon and Dion use different device mesh arguments.

Our implementation of Muon takes a single 1D device mesh as a generic `distributed_mesh` argument. If this mesh is used for sharding parameters, Muon will efficiently perform unsharding using all-to-all. If this mesh is not used for sharding, Muon will distribue work across this mesh and all-gather the final results.

2D sharding is not supported by Muon---use Dion instead. For hybrid-sharded data parallel, with a replicated mesh dimension and a sharded dimension, pass only the sharded sub-mesh to Muon.

```python
mesh = init_device_mesh(
    device_type="cuda",
    mesh_shape=(replicate_size, shard_size),
    mesh_dim_names=("replicate", "shard"),
)

# Hybrid sharded data parallel with 2D device mesh
fully_shard(model, mesh=mesh)

optimizer = Muon(
    param_groups,
    distributed_mesh = mesh["shard"],  # 1D sub-mesh
    ...
)
```

### Usage with ProcessGroup for DDP

Training with DistributedDataParallel (DDP) is also supported. Pass in the DDP-wrapped model's `process_group` instead of a device mesh. This will allow the optimizer to efficiently distribute work across all GPUs.

```python
ddp_model = DistributedDataParallel(model, ...)

optimizer = Dion(
    param_groups,
    replicated_mesh=ddp_model.process_group,
    ...
)
# - or -
optimizer = Muon(
    param_groups,
    distributed_mesh=ddp_model.process_group,
    ...
)
```

## Compressed Data-Parallel Gradient Sync

Dion is capable of *skipping the usual full-gradient all-reduce* by only synchronizing low-rank *P* and *Q* factors (the PowerSGD trick---see [Vogels et al., 2019](https://arxiv.org/abs/1905.13727)). Depending on the rank fraction used, we can greatly compress the amount of communication needed while producing the exact same end result (up to numerical precision).

This feature is applicable across any replicated data-parallel axis for DDP and hybrid-sharded HSDP. It can be enabled or disabled using the `replicate_mesh_grad_sync` option.

* If `replicate_mesh_grad_sync` is True (default) and a `replicate_mesh` is provided, Dion will all-reduce the low-rank compressed states during the optimizer step.
* If `replicate_mesh_grad_sync` is False, Dion will expect that all data-parallel gradients have already been synchronized prior to the optimizer step.

### Usage with HSDP
Typically, hybrid sharding with `fully_shard()` uses a 2D device mesh. To use with Dion's compressed gradient synchronization, pass only the sharded sub-mesh to `fully_shard()`.

In other words, we don't let `fully_shard()` see the replicated mesh dimension, so it will not all-reduce gradients across it. Instead, Dion receives the replicated dimension as its `replicate_mesh` argument, and it will synchronize low-rank matrices during the optimizer step.

Note that if we choose to disable Dion's compressed gradient synchronization, we must make sure to provide the 2D mesh to `fully_shard()`.

| Option                       | `fully_shard()` device mesh | `replicate_mesh_grad_sync` | Optimizer states | Model weights       |
|------------------------------|-----------------------------|----------------------------|------------------|---------------------|
| Dion syncs compressed states | 1D shard sub-mesh           | `True`                     | Decoupled        | Always synchronized |
| FSDP syncs full gradients    | 2D hybrid-sharding mesh     | `False`                    | Synchronous      | Always synchronized |

### Example code

```python
# ------------------------------------------------------------
#  Mode 1: Dion handles DP sync (compressed P/Q)  <-- recommended
# ------------------------------------------------------------
mesh = init_device_mesh("cuda", (dp, fs), ("dp", "fs"))

fully_shard(model, mesh=mesh["fs"])   # DP mesh not provided here

opt = Dion(
    param_groups,
    replicate_mesh           = mesh["dp"],  # Dion still gets DP mesh
    outer_shard_mesh         = mesh["fs"], 
    replicate_mesh_grad_sync = True         # Dion will synchronize low-rank matrices
)

# ------------------------------------------------------------
#  Mode 2: FSDP handles DP sync (classic full gradients)
# ------------------------------------------------------------
mesh = init_device_mesh("cuda", (dp, fs), ("dp", "fs"))

fully_shard(model, mesh=mesh["dp", "fs"])  # FSDP hybrid sharding

opt = Dion(
    param_groups,
    replicate_mesh           = mesh["dp"],    
    outer_shard_mesh         = mesh["fs"], 
    replicate_mesh_grad_sync = False        # Dion expects gradients already synced
)
```

### Usage with DDP

To use compressed gradient synchronization with DDP, always run the model with the `no_sync()` context.

```python
ddp_model = DistributedDataParallel(model, ...)

optimizer = Dion(
    param_groups,
    replicate_mesh=ddp_model.process_group,
    replicate_mesh_grad_sync=True,
    ...
)

for data in dataloader:
    # Always run with no_sync(), not just for gradient accumulation
    with ddp_model.no_sync():
        loss = ddp_model(data)
        loss.backward()

    optimizer.step()
    model.zero_grad()
```


## Comparison Between Optimizers

TODO rename files

* `dion_async.py`
* `dion.py`
* `dion_reference.py`
* `dion_simple.py`
* `muon.py`
* `muon_reference.py`

`dion_async.py` improves communication efficiency by:

- **Processing parameter groups in batches**, amortizing overhead across multiple tensors.
- **Splitting communication into `reduce-scatter` + `all-gather`** to better distribute the QR workload.
- **Overlapping communication with compute** by interleaving batches asynchronously, reducing idle time on slower networks.


## 🚀 Experimental Features

### Mixed Precision Dion

TODO

```python
from dion import Dion, DionMixedPrecisionConfig

dion_mixed_precision_config = DionMixedPrecisionConfig(
    momentum_dtype=torch.bfloat16,
    Q_dtype=torch.bfloat16,
)
optimizer = Dion(
    ...
    mixed_precision_config=dion_mixed_precision_config,
    ...
)
```

### Accelerating Optimization step for lower ranks

After a few warmup iterations, the expensive QR decomposition can be replaced with **CQR**, leading to **2X** optimization step speedups.

> ⚠️ **Note:** Speedup for **rank fraction = 1.0** is still under development. For models **larger than 10B**, you may need to **further reduce the rank fraction** to see benefits.

To train the accelerated 160M model:
```bash
torchrun --standalone --nproc_per_node=8 train.py --config configs/dion_efficient_160m.yaml
```

After the training you should be able to reproduce the second plot in [validation curves for GPT-small](https://microsoft-research.wandb.io/t-gmagakyan/dion-exp/reports/Validation-curves-for-GPT-small--VmlldzoxNjk5OA?accessToken=52e6z4d18yfkewz1bawlkmwc2m91al9ssa7rpwvnx1f1xa66j15lr7x315wj2kys).


### Triton Kernels for Muon Newton-Schulz

TODO

Maybe this should be disabled by default

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
