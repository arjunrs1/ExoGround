#!/bin/bash
#SBATCH --job-name=ExoGround_ks_multi_train_400
#SBATCH --output=/checkpoint/%u/slurm_logs/exoground/train_%j.out
#SBATCH --error=/checkpoint/%u/slurm_logs/exoground/train_%j.out
#SBATCH --partition=learnfair
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=80
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --constraint=volta32gb

### init virtual environment if needed
source activate sounding_narrations

srun --label torchrun --nproc_per_node=8 \
    main_egoexo4d_distributed.py \
    --batch_size 16 \
    --epochs 100 \
    --num_workers 0 \
    --use_keysteps \
    --views multi