import argparse
import math
import os
import time
import torch
import torch.distributed as dist
import uuid
import wandb
import yaml

from dataclasses import dataclass
from pathlib import Path
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DeviceMesh
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from typing import Optional

from models.gpt_model import GPT, GPTConfig, parallelize_gpt_model
from models.gpt_utils import DistributedDataLoader
from opts.dion import Dion
from opts.muon import Muon
from opts.dion_reference import Dion as DionReference
from opts.muon_reference import MuonMoonlight as MuonReference


@dataclass
class Hyperparameters:
    # Data directory
    data_dir: str = "data/fineweb10B"
    output: str = "output"  # output directory

    # Training config
    batch_size: int = 8 * 64  # global batch size (across devices)
    device_batch_size: int = 64  # per-device batch size
    sequence_length: int = 1024  # tokens per sequence
    num_iterations: int = 5000
    warmup_ratio: float = 0.01
    warmdown_ratio: float = 0.2

    # Model config
    model_dim: int = 768
    n_layer: int = 12
    n_head: int = 6

    # Evaluation and logging
    val_loss_every: int = 125
    val_tokens: int = 10485760
    save_every: int = 0
    wandb_project_name: str = "test"

    # Optimizer
    optimizer: str = "dion"
    scalar_opt: str = "lion"
    efficient: bool = False
    muon_adjust_lr: str = "spectral_norm"  # for Muon only

    # Hyperparameters
    lr: float = 0.02
    adam_lr: float = 0.002
    mu: float = 0.95
    beta: float = 0.9
    weight_decay: float = 0.01
    rank_fraction: float = 0.125
    oversample: float = 1.25
    qr_warmup: float = 0.05

    power_iters: int = 1
    approx_method: str = "qr"  # for DionOld and DionNorm optimizers


# Helper function to only print on global rank 0
MASTER_PROCESS = False


def print0(*args):
    if MASTER_PROCESS:
        print(*args)


def init_distributed(dp_size, fs_size, tp_size) -> Optional[DeviceMesh]:
    # Initialize device mesh for distributed training
    # If all mesh dimensions are None, we default to using DDP
    assert torch.cuda.is_available(), "CUDA must be available!"
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Set global master process flag
    global MASTER_PROCESS
    MASTER_PROCESS = rank == 0

    mesh_dims = (dp_size, fs_size, tp_size)
    if all(d is None for d in mesh_dims):
        # If no mesh dimensions given, initialize process group for DDP
        device_mesh = None
        dist.init_process_group(backend="nccl")
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        print0(f"Initialized DDP with world size {world_size}")

    else:
        # Use device mesh for distributed training
        # All mesh dimensions must be specified
        assert all(
            d is not None for d in mesh_dims
        ), f"All mesh dimensions (dp_size, fs_size, tp_size) must be specified, but got ({dp_size}, {fs_size}, {tp_size})"

        # Check if we have the right number of GPUs
        total_gpus = dp_size * fs_size * tp_size
        assert world_size == total_gpus, (
            f"World size {world_size} does not match expected size {total_gpus} "
            f"(DP {dp_size}, FS {fs_size}, TP {tp_size})"
        )

        device_mesh = init_device_mesh(
            device_type="cuda",
            mesh_shape=(dp_size, fs_size, tp_size),
            mesh_dim_names=("dp", "fs", "tp"),
        )
        print0("Device mesh:", device_mesh)

    return device_mesh


def override_args_from_cli(
    args: Hyperparameters, cli_args: argparse.Namespace
) -> Hyperparameters:
    for key, value in vars(cli_args).items():
        if value is not None:
            if hasattr(args, key):
                print(f"Setting hyperparameter {key}={value}")
                setattr(args, key, value)
    return args


def main():
    # --- Command-line argument parsing ---
    parser = argparse.ArgumentParser(
        description="Training script with input and output directories"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a YAML file whose keys match train.py flags "
        "(CLI values always override the YAML).",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory that contains fineweb_train_*.bin and fineweb_val_*.bin",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Directory where logs and checkpoints will be saved",
    )

    # ---------- optimizer ----------
    parser.add_argument(
        "--optimizer", type=str, default=None, help="Choice of optimizer algorithm"
    )
    parser.add_argument(
        "--scalar_opt", type=str, help="Optimizer for scalar parameters", default=None
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument(
        "--adam_lr",
        type=float,
        default=None,
        help="Adam learning rate for scalar params",
    )
    parser.add_argument(
        "--muon_adjust_lr",
        type=str,
        default=None,
        help="Adjust learning rate method for Muon",
    )
    parser.add_argument(
        "--inv_rank_fraction",
        type=int,
        default=None,
        help="Sparsity level for the optimizer",
    )

    # ---------- model ----------
    parser.add_argument("--model_dim", type=int, default=None)
    parser.add_argument("--n_layer", type=int, default=None)
    parser.add_argument("--n_head", type=int, default=None)

    # ---------- training hyperparameters ----------
    parser.add_argument(
        "--num_iterations", type=int, default=None, help="Number of training steps"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Global batch size"
    )
    parser.add_argument("--device_batch_size", type=int, default=None)
    parser.add_argument("--sequence_length", type=int, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=None)

    # ---------- wandb logging ----------
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument(
        "--wandb_project_name", type=str, default=None, help="Wandb project name"
    )
    parser.add_argument(
        "--wandb_job_name",
        type=str,
        default=None,
        help="Append custom text to wandb job name",
    )

    # ---------- distributed training ----------
    parser.add_argument(
        "--dp_size", type=int, default=None, help="Data Parallel size (no sharding)"
    )
    parser.add_argument(
        "--fs_size", type=int, default=None, help="Fully Sharded Data Parallel size"
    )
    parser.add_argument(
        "--tp_size", type=int, default=None, help="Tensor Parallel size"
    )
    parser.add_argument(
        "--opt_grad_sync",
        action="store_true",
        help="Do data-parallel gradient sync inside Dion optimizer",
    )

    # ---------- debugging ----------
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--no_compile", action="store_true", help="Disable torch.compile"
    )

    cli_args = parser.parse_args()
    if cli_args.config:
        # Read YAML → dict
        cfg_path = Path(cli_args.config)
        with cfg_path.open("r") as f:
            yaml_cfg = yaml.safe_load(f)

        # Copy any key the user did NOT supply on the CLI
        for k, v in yaml_cfg.items():
            if getattr(cli_args, k, None) is None:
                setattr(cli_args, k, v)

    # --- Distributed setup ---
    device_mesh = init_distributed(
        dp_size=cli_args.dp_size,
        fs_size=cli_args.fs_size,
        tp_size=cli_args.tp_size,
    )
    if device_mesh is not None:
        # Combine replicated and sharded data parallel meshes
        data_parallel_mesh = device_mesh["dp", "fs"]._flatten()
        data_parallel_size = data_parallel_mesh.size()
        data_parallel_rank = data_parallel_mesh.get_local_rank()
    else:
        # We are using DDP with one global process group
        data_parallel_mesh = None
        data_parallel_size = dist.get_world_size()
        data_parallel_rank = dist.get_rank()

    # --- Logging and wandb initialization ---
    run_id = str(uuid.uuid4())
    logdir = (
        os.path.join(cli_args.output, run_id)
        if cli_args.output
        else os.path.join("logs", run_id)
    )
    os.makedirs(logdir, exist_ok=True)
    logfile = os.path.join(logdir, f"{run_id}.txt")
    with open(logfile, "w") as f:
        f.write("Starting training run...\n")

    # Load hyperparameters and update with CLI arguments
    args = Hyperparameters()
    args = override_args_from_cli(args, cli_args)
    if cli_args.inv_rank_fraction:
        args.rank_fraction = 1.0 / cli_args.inv_rank_fraction

    if args.efficient == True:
        print0("Using efficient Dion optimizer")
        if args.rank_fraction > 0.5:
            raise ValueError(
                "For efficient Dion, rank_fraction must be <= 0.5 to use CQR trick. Speedup"
                "for rank_fraction=1 is under development"
            )

    if MASTER_PROCESS and not cli_args.no_wandb:
        assert args.wandb_project_name, "wandb project name is required"
        opt_name = f"{args.optimizer}+{args.scalar_opt}"
        run_name = f"({opt_name})_bs={args.batch_size}_lr={args.lr}"
        if "dion" in args.optimizer:
            run_name += f"_sp={args.rank_fraction}"
            if args.efficient:
                run_name += f"_eff=True"

        if cli_args.dp_size is not None:
            run_name += (
                f"_dp={cli_args.dp_size}_fs={cli_args.fs_size}_tp={cli_args.tp_size}"
            )
        if cli_args.wandb_job_name:
            run_name += f"_{cli_args.wandb_job_name}"
        if not cli_args.debug:
            wandb.login(
                key=os.environ.get("WANDB_API_KEY"),
                host=os.environ.get("WANDB_HOST"),
                timeout=0,
            )
            wandb.init(
                project=args.wandb_project_name, name=run_name, config=args.__dict__
            )

    # --- DataLoader Setup ---
    if cli_args.debug:
        # in debug mode, make batch size very small
        args.batch_size = 2 * data_parallel_size
        args.device_batch_size = 1

    B, T = args.device_batch_size, args.sequence_length
    assert args.val_tokens % (B * T * data_parallel_size) == 0, "Invalid val_tokens"
    val_steps = args.val_tokens // (B * T * data_parallel_size)

    if cli_args.debug:
        # train for just a few steps
        args.num_iterations = 300
        val_steps = min(val_steps, 2)

    assert args.batch_size % (B * data_parallel_size) == 0, "Invalid batch_size"
    train_accumulation_steps = args.batch_size // (B * data_parallel_size)

    train_glob = os.path.join(args.data_dir, "fineweb_train_*.bin")
    val_glob = os.path.join(args.data_dir, "fineweb_val_*.bin")

    # Each data parallel rank gets different data
    # TP ranks must all use identical data
    train_loader = DistributedDataLoader(
        train_glob, B, T, data_parallel_rank, data_parallel_size
    )
    val_loader = DistributedDataLoader(
        val_glob, B, T, data_parallel_rank, data_parallel_size
    )

    print0(
        f"Training DataLoader: {train_loader.ntok_total} tokens across {len(train_loader.files)} files"
    )
    print0(
        f"Validation DataLoader: {val_loader.ntok_total} tokens across {len(val_loader.files)} files"
    )
    x, y = train_loader.next_batch()

    # --- Model Initialization ---
    num_vocab = 50304  # nearest multiple of 128 for efficiency
    gpt_config = GPTConfig(
        block_size=args.sequence_length,
        vocab_size=num_vocab,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.model_dim,
    )
    with torch.device("meta"):
        model = GPT(gpt_config)

    # Shard the model if using a device mesh
    # If opt_grad_sync is True, FSDP will not handle data-parallel gradient sync
    # If opt_grad_sync if False, we use Pytorch HSDP to do data-parallel gradient sync
    if device_mesh is not None:
        parallelize_gpt_model(
            model,
            device_mesh=device_mesh,
            dp_name=None if cli_args.opt_grad_sync else "dp",
            fs_name="fs",
            tp_name="tp",
        )
        raw_model = model

    # Move model to GPU
    model.to_empty(device="cuda")
    model.init_weights()
    if not cli_args.no_compile:
        model.compile()

    # If no device mesh, we are using DDP
    if device_mesh is None:
        model = DDP(model, device_ids=[data_parallel_rank])
        raw_model = model.module  # the underlying model

    # Ensure parameters are contiguous
    for i, p in enumerate(model.parameters()):
        if not p.is_contiguous():
            raise ValueError(f"Parameter {i} is not contiguous")

    num_params = sum(p.numel() for p in model.parameters())
    print0(f"----------Total parameters: {num_params}")

    # --- Optimizer Setup ---
    matrix_params = list(raw_model.transformer.h.parameters())
    embedding_params = list(raw_model.transformer.wte.parameters())
    lm_head_params = list(raw_model.lm_head.parameters())
    param_groups = [dict(params=matrix_params)]

    # Create parameter groups for scalar optimizer
    assert args.scalar_opt in ["adamw", "lion", "clion"]
    param_groups.append(
        dict(
            params=embedding_params,
            algorithm=args.scalar_opt,
            lr=args.lr,
            betas=(0.95, 0.98),
            weight_decay=0,
        )
    )
    param_groups.append(
        dict(
            params=lm_head_params,
            algorithm=args.scalar_opt,
            lr=args.lr / math.sqrt(args.model_dim),  # scale LR for lm_head
            betas=(0.95, 0.98),
            weight_decay=0,
        )
    )

    if device_mesh is not None:
        data_parallel_mesh = device_mesh["dp"]
        outer_shard_mesh = device_mesh["fs"]
        inner_shard_mesh = device_mesh["tp"]
    else:
        data_parallel_mesh = model.process_group
        outer_shard_mesh = None
        inner_shard_mesh = None

    # Create the main optimizer
    optimizers = []
    if args.optimizer == "dion":
        opt = Dion(
            param_groups,
            data_parallel_mesh=data_parallel_mesh,
            outer_shard_mesh=outer_shard_mesh,
            inner_shard_mesh=inner_shard_mesh,
            data_parallel_grad_sync=cli_args.opt_grad_sync,
            rank_fraction=args.rank_fraction,
            lr=args.lr,
            mu=args.mu,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "muon":
        if device_mesh is not None:
            assert not (
                cli_args.fs_size > 1 and cli_args.dp_size > 1
            ), "Hybrid sharded data parallel is not supported by Muon."
            assert cli_args.tp_size == 1, "Tensor parallel is not supported by Muon."
            distributed_mesh = (
                device_mesh["dp"] if cli_args.dp_size > 1 else device_mesh["fs"]
            )
        else:
            distributed_mesh = model.process_group  # using DDP
        opt = Muon(
            param_groups,
            distributed_mesh=distributed_mesh,
            lr=args.lr,
            mu=args.mu,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "dion_reference":
        assert (
            device_mesh is None
        ), "Simplified version of Dion does not support device mesh"
        assert cli_args.approx_method is not None
        opt = DionReference(
            param_groups,
            lr=args.lr,
            mu=args.mu,
            weight_decay=args.weight_decay,
            rank=round(args.rank_fraction * args.model_dim),
            approx_method=cli_args.approx_method,
            qr_warmup_steps=int(args.qr_warmup * args.num_iterations),
            efficient=args.efficient,
        )
    elif args.optimizer == "muon_moonlight":
        assert (
            device_mesh is None
        ), "Simplified version of Dion does not support device mesh"
        opt = MuonReference(
            muon_params=matrix_params,
            adamw_params=[pg for pg in param_groups if pg["algorithm"] == "adamw"],
            lion_params=[pg for pg in param_groups if pg["algorithm"] == "lion"],
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.mu,
            adjust_lr=args.muon_adjust_lr,
        )
    elif args.optimizer == "adam":
        opt = torch.optim.AdamW(
            param_groups,
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    optimizers.append(opt)

    # Check opt_grad_sync and optimizer combination
    if cli_args.opt_grad_sync and args.optimizer != "dion":
        print0("Warning: --opt_grad_sync is set for non-Dion optimizer")
    if not cli_args.opt_grad_sync and args.optimizer == "dion":
        print0("Warning: Not using --opt_grad_sync for Dion optimizer")

    def get_lr(it):
        warmup_iters = round(args.warmup_ratio * args.num_iterations)
        warmdown_iters = round(args.warmdown_ratio * args.num_iterations)
        if it < warmup_iters:
            return (it + 1) / warmup_iters
        elif it < args.num_iterations - warmdown_iters:
            return 1.0
        else:
            return (args.num_iterations - it) / warmdown_iters

    schedulers = [
        torch.optim.lr_scheduler.LambdaLR(opt, lambda it: get_lr(it))
        for opt in optimizers
    ]

    # --- Training Loop ---
    training_time_ms = 0
    torch.cuda.synchronize()
    t0 = time.time()
    pbar = tqdm(total=args.num_iterations, desc="Training", disable=not MASTER_PROCESS)

    # Use autocast for mixed precision
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    for step in range(args.num_iterations + 1):
        last_step = step == args.num_iterations
        if step == 10:
            training_time_ms = 0
            t0 = time.time()
        timed_steps = (step - 9) if step > 10 else float("nan")

        # --- Validation ---
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.time() - t0)
            model.eval()
            val_loader.reset()
            val_loss = torch.tensor(0.0, device=x.device)
            for _ in range(val_steps):
                with torch.no_grad():
                    x_val, y_val = val_loader.next_batch()
                    with ctx:
                        loss = model(x_val, y_val)
                        val_loss += loss
            # Average loss across devices
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            val_loss = val_loss.item() / val_steps
            log_message = (
                f"step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms"
            )
            print0(log_message)
            if MASTER_PROCESS and not cli_args.no_wandb and not cli_args.debug:
                wandb.log(
                    {
                        "val/loss": val_loss,
                        "step": step,
                        "time/training_time_ms": training_time_ms,  # Log total elapsed training time in ms
                    }
                )
            pbar.set_postfix(val_loss=f"{val_loss:.4f}")
            torch.cuda.synchronize()
            t0 = time.time()
            model.train()

        if last_step:
            break

        model.train()
        for i in range(1, train_accumulation_steps + 1):
            with ctx:
                loss = model(x, y)
            train_loss = loss.detach()
            loss = loss / train_accumulation_steps
            x, y = train_loader.next_batch()

            # Turn off DDP grad sync if opt_grad_sync is True
            ddp_no_sync = i < train_accumulation_steps or cli_args.opt_grad_sync
            if isinstance(model, DDP) and ddp_no_sync:
                with model.no_sync():
                    loss.backward()
            else:
                if isinstance(model, FSDPModule):
                    # Gradient accumulation for DP on top of FSDP
                    # FSDP always synchronizes sharded gradients via reduce-scatter
                    model.set_is_last_backward(i == train_accumulation_steps)
                loss.backward()

        # Gradient norm
        grad_norm = torch.nn.utils.get_total_norm(
            [p.grad for p in model.parameters() if p.grad is not None]
        )

        for opt, sched in zip(optimizers, schedulers):
            opt.step()
            sched.step()
        model.zero_grad(set_to_none=True)

        # Approximate updated training time just before logging
        approx_time = training_time_ms + 1000 * (time.time() - t0)
        if MASTER_PROCESS and not cli_args.no_wandb and not cli_args.debug:
            wandb.log(
                {
                    "train/loss": train_loss.item(),
                    "train/grad_norm": grad_norm.item(),
                    "step": step + 1,
                    "time/training_time_ms": approx_time,  # Log approximate elapsed training time in ms
                }
            )
        if MASTER_PROCESS and cli_args.debug:
            print(
                f"Step {step}: train_loss={train_loss.item():.4f}, grad_norm={grad_norm.item():.4f}"
            )
        pbar.update(1)
        pbar.set_postfix(train_loss=f"{train_loss.item():.4f}")
        t0 = time.time()  # reset timer after optimizer step

    pbar.close()
    print0(
        f"Peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB"
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
