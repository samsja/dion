# Note for Shital:
# - nproc_per_node and nnodes should be changed to 8 and 2
# - data_dir and output should be changed according to your data stroage
#
#


torchrun --nproc_per_node=8\
         --nnodes=1 \
         train.py \
         --data_dir           dion/fineweb100B \
         --output             logs \
         --optimizer          adam \
         --scalar_opt         adam \
         --model_dim          2048 \
         --n_layer            24 \
         --n_head             32 \
         --batch_size         2048 \
         --device_batch_size  2 \
         --sequence_length    2048 \
         --num_iterations     7500 \
         --warmup_ratio       0.1 \
         --sparsity           1.0 \
         --lr                 0.0012 \
         --no_wandb