#!/bin/bash
#SBATCH --job-name=jnt_trn
#SBATCH --output=/checkpoint/%u/slurm_logs/joint/train_joint_%j.out
#SBATCH --error=/checkpoint/%u/slurm_logs/joint/train_joint_%j.out
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
    main_egoexo4d_distributed.py \
    --batch_size 16 \
    --epochs 100 \
    --num_workers 0 \
    --use_keysteps \
    --views all \
    --exos all \
    --model joint \
    --minimum_four_exo_takes \
    --use_distill_nce_loss \
    --same_view_negative \
    --reverse_ranking \
    --name_prefix $1

# --same_view_negative \
### --use_distill_nce_loss \
### --curriculum_train \
### --start_frac 0.1 \
### --end_epoch_frac 0.5 \
### --final_phase_prop 0.5 \
### --sorted_curr_train ("phased": cycle the positive features, "sorted": sort the training data, don't cycle the positive)

#CURR TRAIN:
    # --curriculum_train \
    # --start_frac 0.75 \
    # --end_epoch_frac 0.5 \
    # --sorted_curr_train sorted \

# --use_tf_video_features \

    # --use_distill_nce_loss \
    # --same_view_negative \
    # --only_same_view_negative \