#!/bin/bash
#SBATCH --account=CCR24058
#SBATCH --job-name=jnt_trn_lemma
#SBATCH --output=/scratch/10323/asomaya1/exoground/outputs/lemma/train/joint_%j.out
#SBATCH --error=/scratch/10323/asomaya1/exoground/outputs/lemma/train/joint_%j_err.out
#SBATCH --partition=gh-dev
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00

if [ -z "$1" ]; then
    echo "Error: No prefix name provided."
    echo "Usage: sbatch $0 <prefix_name>"
    exit 1
fi

### init virtual environment if needed
module load gcc cuda
source activate exoground

srun --label torchrun --nproc_per_node=1 \
    main_egoexo4d_distributed.py \
    --dataset lemma \
    --batch_size 16 \
    --epochs 100 \
    --num_workers 0 \
    --use_keysteps \
    --views all \
    --exos all \
    --model joint \
    --use_distill_nce_loss \
    --name_prefix $1


#    --use_distill_nce_loss \
#    --same_view_negative \
