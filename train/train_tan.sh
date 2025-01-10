#!/bin/bash
#SBATCH --job-name=tan_train
#SBATCH --output=/checkpoint/%u/slurm_logs/TAN/train_tan_%j.out
#SBATCH --error=/checkpoint/%u/slurm_logs/TAN/train_tan_%j.out
#SBATCH --partition=learnfair
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=80
#SBATCH --exclusive
#SBATCH --time=72:00:00
#SBATCH --constraint=volta32gb

if [ -z "$1" ]; then
    echo "Error: No prefix name provided."
    echo "Usage: sbatch $0 <prefix_name>"
    exit 1
fi

### init virtual environment if needed
source activate sounding_narrations

srun --label torchrun --nproc_per_node=8 \
    main.py \
    --model init \
    --loss nce \
    --batch_size 16 \
    --epochs 20 \
    --num_workers 0 \
    --use_keysteps \
    --minimum_four_exo_takes \
    --name_prefix $1