#!/bin/bash
#SBATCH --account=CCR24058
#SBATCH --job-name=jnt_trn
#SBATCH --output=/scratch/10323/asomaya1/exoground/outputs/egoexo4d/train/joint_%j.out
#SBATCH --error=/scratch/10323/asomaya1/exoground/outputs/egoexo4d/train/joint_%j_err.out
#SBATCH --partition=gh
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00

if [ -z "$1" ]; then
    echo "Error: No prefix name provided."
    echo "Usage: sbatch $0 <prefix_name>"
    exit 1
fi

### init virtual environment if needed
module load gcc cuda
source activate exoground

# Add debugging information
# echo "=== Environment Debug Info ==="
# echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
# echo "Checking nvidia-smi:"
# nvidia-smi
# echo "Checking PyTorch CUDA:"
# python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count())"
# echo "==========================="

srun --label torchrun --nproc_per_node=1 \
    main_egoexo4d_distributed.py \
    --dataset egoexo4d \
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
    --curriculum_train \
    --exo_exo_distill \
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

# --exo_exo_distill \