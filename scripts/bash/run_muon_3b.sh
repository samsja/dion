
torchrun --nproc_per_node=8\
         --nnodes=2 \
         train.py \
         --data_dir           dion/fineweb100B \
         --output             logs \
         --optimizer          muon_moonlight \
         --scalar_opt         adam \
         --model_dim          3072 \
         --n_layer            24 \
         --n_head             32 \
         --batch_size         4096 \
         --device_batch_size  1 \
         --sequence_length    2048 \
         --num_iterations     5000 \
         --warmup_ratio       0.0 \
         --sparsity           1.0 \
         --lr                 0.01 \
         --muon_adjust_lr     spectral_norm \
         --no_wandb