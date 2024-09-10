#!/bin/bash
#SBATCH --job-name=ExoGround_ks_multi_egoexo_rand_narr_train
#SBATCH --output=/checkpoint/%u/slurm_logs/exoground/train_%j.out
#SBATCH --error=/checkpoint/%u/slurm_logs/exoground/train_%j.out
#SBATCH --partition=learnfair
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=80
#SBATCH --exclusive
#SBATCH --time=72:00:00
#SBATCH --constraint=volta32gb

### init virtual environment if needed
source activate sounding_narrations

srun --label torchrun --nproc_per_node=8 \
    main_egoexo4d_distributed.py \
    --batch_size 16 \
    --epochs 500 \
    --num_workers 0 \
    --use_keysteps \
    --views multi \
    --multi_view_egoexo \
    --use_pairwise_distill_nce_loss \
    --final_phase_prop 0.5
