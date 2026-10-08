#!/usr/bin/env bash
# The runs behind the PR's results, in order. Not meant to be rerun as one script: each block ran
# inside Slurm allocation 3409 on H200 nodes.
#
# prime-rl arms, each a worktree detached at its commit with its own venv (uv sync --all-extras --all-packages):
#   A: PrimeIntellect-ai/prime-rl 8cbab9561, Dion samsja/dion 6f9242d
#   B: PrimeIntellect-ai/prime-rl 199cee1bf, Dion garrett361/dion 2c87bda (this PR's fix)
# Both commits sit on prime-rl main 72834b3b5 plus b92eb2f23, which adds the time/optimizer metric.
# prime_rl/ holds that branch's benchmarks/scripts/dion-muon-ep/ harness; run_cell.sh runs from there.
set -euo pipefail

A=~/github/PrimeIntellect-ai/prime-rl-dion-ab-a
B=~/github/PrimeIntellect-ai/prime-rl-dion-ab-b
H=benchmarks/scripts/dion-muon-ep
N=prime-nebius-puku-h200-gpu
P1=$N-062
P2=$N-064,$N-043
F8=$N-022,$N-024,$N-025,$N-027,$N-043,$N-048,$N-062,$N-064
OUT=~/tmp/profiling/dion-muon-ep

# Toy repro on one node: FSDP 4 x EP 2 and pure EP 8, expert width 256, 2048 and 4096
for dion in ~/github/garrett361/dion-upstream-fix-ep ~/github/garrett361/dion-fix-muon-ep-local-experts; do
    for args in "--ep 2" "--ep 8" "--ep 2 --dim 2048" "--ep 2 --dim 4096 --layers 2" "--ep 8 --dim 4096 --layers 2"; do
        srun --jobid=3409 --overlap -N1 -n1 --nodelist=$P1 env PYTHONPATH=$dion \
            <prime-rl venv>/bin/torchrun --nproc-per-node 8 benchmarks/muon_ep_local_experts/repro.py $args
    done
done

# P1: 1 node, 6 layers. Arm A ran out of memory at step 1.
$A/$H/run_cell.sh $A p1 dion-ab-p1-a-1 $P1
$B/$H/run_cell.sh $B p1 dion-ab-p1-b-1 $P1

# P2: 2 nodes, 6 layers, untraced timing then a 5-step trace per arm
$A/$H/run_cell.sh $A p2 dion-ab-p2-a-1 $P2
$B/$H/run_cell.sh $B p2 dion-ab-p2-b-1 $P2
$A/$H/run_cell.sh $A p2 dion-ab-p2-a-trace3 $P2 --max-steps 5 --trace-path $OUT/traces/p2-a3
$B/$H/run_cell.sh $B p2 dion-ab-p2-b-trace $P2 --max-steps 5 --trace-path $OUT/traces/p2-b

# F8: 8 nodes, full model. Arm A ran out of memory at step 1.
$A/$H/run_cell.sh $A f8 dion-ab-f8-a-2 $F8 --dist-timeout-seconds 1800
$B/$H/run_cell.sh $B f8 dion-ab-f8-b-1 $F8 --dist-timeout-seconds 1800

# Optimizer kernel tables and the figure, from local profiling scripts that are not published
# (top_kernels.py and timeline_figure.py, both with --launched-in)
python3 top_kernels.py $OUT/traces/p2-a3/trace_0.json.gz --launched-in optimizer --by name --top 8
python3 top_kernels.py $OUT/traces/p2-b/trace_0.json.gz --launched-in optimizer --by name --top 10
uv run --script timeline_figure.py --out optimizer-before-after.png \
    --window -20 2200 --launched-in optimizer --region-label "optimizer step" --edge-width 0 \
    --trace "Before: Dion 6f9242d=$OUT/traces/p2-a3/trace_0.json.gz" \
    --trace "After: this PR (2c87bda)=$OUT/traces/p2-b/trace_0.json.gz" \
    --category "Newton-Schulz GEMM:^nvjet|gemm|cutlass:#0072b2" \
    --category "Muon elementwise / norm:^triton|elementwise|reduce|foreach:#e69f00" \
    --comm-kernel "^nccl" --launch-annotation "^Optimizer.step" --comm-label "Muon all-to-all:Optimizer:#d55e00" \
    --comm-row-label "NCCL" --comm-legend-suffix "" --comm-step-marker optimizer \
    --title "DeepSeek V4 Flash SFT, 6 layers, 2 nodes, EP 8, CP 8: Muon optimizer step, rank 0"
