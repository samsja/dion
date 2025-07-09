# Dion Optimizer

This repository provides a preliminary implementation of the Dion optimizer (https://arxiv.org/pdf/2504.05295).

## ⚙️ Quick Start

Most of the relevant code is located in the `scripts/` directory.

```bash
cd scripts
pip install -r requirements.txt
```

To download the dataset, run:

```bash
python data/cached_fineweb10B.py 8
```

To quickly train a 160M model, run:

```bash
torchrun --standalone --nproc_per_node=8 train.py --config configs/dion_160m.yaml
``` 

## Faster Dion for lower ranks



## 🚀 Accelerating Optimization step for lower ranks
Gagik Magakyan and Ying Fan contributed to this.

After a few warmup iterations, the expensive QR decomposition can be replaced with **CQR**, leading to **2–5× optimization step speedups**—and even more for larger models!


> ⚠️ **Note:** Speedup for **rank fraction = 1.0** is still under development. For models **larger than 10B**, you may need to **further reduce the rank fraction** to see benefits.

To train the accelerated 160M model:
```bash
torchrun --standalone --nproc_per_node=8 train.py --config configs/dion_efficient_160m.yaml
```



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
