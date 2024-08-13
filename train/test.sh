#!/bin/bash
#SBATCH --job-name=ExoGround_ks_exo_distill_test
#SBATCH --output=/checkpoint/%u/slurm_logs/exoground/%j.out
#SBATCH --error=/checkpoint/%u/slurm_logs/exoground/%j.out
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

srun --label torchrun --nproc_per_node=8 main_egoexo4d_distributed.py --batch_size 16 --epochs 100 --num_workers 0 --no_audio --use_keysteps --views exo --use_distill_nce_loss --test $1