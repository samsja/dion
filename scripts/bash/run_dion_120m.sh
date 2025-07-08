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
         --optimizer          dion \
         --scalar_opt         lion \
         --model_dim          768 \
         --n_layer            12 \
         --n_head             6 \
         --batch_size         1024 \
         --device_batch_size  32 \
         --sequence_length    1024 \
         --num_iterations     3000 \
         --warmup_ratio       0.0 \
         --sparsity           1.0 \
         --lr                 0.01 \
         --no_wandb