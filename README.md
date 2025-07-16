# Dion - Distributed Orthonormal Updates

High-performance implementations of Dion and Muon optimizer algorithms for distributed ML training.

Read our paper on Dion here: https://arxiv.org/pdf/2504.05295

## Requirements

This code is written for modern PyTorch (version 2.6 or newer) using DTensor-based parallelism. This includes FSDP2 with `fully_shard`and tensor parallelism (TP) with `parallelize_module`. Support for other distributed training APIs is not guaranteed.

## Quick Start

Install required Python packages:
```bash
pip install -r requirements.txt
```

Download the dataset:
```bash
python data/cached_fineweb10B.py
```

Train a 120M model with default hyperparameters:
```bash
torchrun --standalone train.py
``` 


## Citation

If you use Dion in your research, please cite:

```bash
@article{ahn2025dion,
  title={Dion: Distributed Orthonormalized Updates},
  author={Ahn, Kwangjun and Xu, Byron and Abreu, Natalie and Langford, John},
  journal={arXiv preprint: 2504.05295},
  year={2025}
}
