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
