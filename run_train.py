"""
Runs a sweep over the specified file.
To use, specify `config`, `sweep_config`, `fsdp_config`, and `script_name` arguments.
"""

import subprocess
from itertools import product
from omegaconf import OmegaConf
import os
import time
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def flatten(config):
    """Flatten a nested dictionary."""
    flat_config = {}
    for k, v in config.items():
        if isinstance(v, dict) or OmegaConf.is_dict(v):
            for k2, v2 in flatten(v).items():
                flat_config[f"{k}.{k2}"] = v2
        else:
            flat_config[k] = v
    return flat_config


def grid_to_list(grid):
    """Convert a grid to a list of configs."""
    flat_grid = flatten(grid)
    iter_overwrites = {}
    flat_overwrites = {}
    for k, v in flat_grid.items():
        if isinstance(v, list) or OmegaConf.is_list(v):
            iter_overwrites[k] = v
        else:
            flat_overwrites[k] = v

    product_values = list(product(*iter_overwrites.values()))
    grid_list = []
    for values in product_values:
        overwrite_dict = dict(zip(iter_overwrites.keys(), values))
        overwrite_dict.update(flat_overwrites)
        grid_list.append(overwrite_dict)
    return grid_list


def run(cli_args):

    # Compute overrides
    base_sweep = OmegaConf.load(cli_args.config)
    index = cli_args.index if "index" in cli_args else 0
    
    list_of_sweeps = base_sweep.pop("sweep")
    config_list = []
    for sweep in list_of_sweeps:
        sweep_config = OmegaConf.merge(base_sweep, sweep)
        config_list += grid_to_list(sweep_config)
    overrides = config_list[index]
    
    

    if "debug" in cli_args:
        print(len(config_list), config_list)    
        
    if "debug" in cli_args:
        os.environ["WORLD_SIZE"] = str(1)
        run_id = cli_args.job_name if "job_name" in cli_args else f"debug-{str(int(time.time()))}"
        ckpt_dir = cli_args.checkpoint_dir if "checkpoint_dir" in cli_args else None
        launch_args = [
            f"torchrun",
            f"--nproc_per_node=2",
            f"--nnodes=1",
            "train.py",
            f'run_id={run_id}', 
        ]
    else:
        os.environ["WORLD_SIZE"] = str(cli_args.nproc_per_node * cli_args.nnodes)
        run_id = cli_args.job_name
        
        launch_args = [
            f"torchrun",
            f"--nproc_per_node={cli_args.nproc_per_node}",
            f"--nnodes={cli_args.nnodes}",
            f"--node_rank={os.environ['NODE_RANK']}",
            f"--master_addr={os.environ['MASTER_ADDR']}",
            f"--master_port={os.environ['MASTER_PORT']}",
            "train.py",
            f"--data_dir {cli_args.input}",
            f"--output {cli_args.output}",  
            # f"wandb_run_name={run_name}",
        ]
    
    for k, v in overrides.items():
        launch_args.append(f"{k}={v}")
        
    checkpoint_dir = cli_args.checkpoint_dir if "checkpoint_dir" in cli_args else None
    print('checkpoint_dir', checkpoint_dir, flush=True)
    if checkpoint_dir is not None:
        checkpoint_dir = os.path.join(checkpoint_dir, run_id, 'state.pt')
        if os.path.exists(checkpoint_dir):
            launch_args.append(f"checkpoint_path={os.path.join(run_id, 'state.pt')}")
            launch_args.append("resume_mode=resume")
            print(f'---Resuming run {run_id} from checkpoint {checkpoint_dir}---', flush=True)
        else:
            print(f'---Starting new run {run_id}---', flush=True)
    else:
        print(f'---Starting new run {run_id} (none checkpoint_dir)---', flush=True)

    if "debug" in cli_args:
        print(launch_args)
        

    subprocess.run(launch_args)


if __name__ == "__main__":
    cli_args = OmegaConf.from_cli()
    run(cli_args)