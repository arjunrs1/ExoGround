#!/bin/bash
#SBATCH --account=CCR24058
#SBATCH --job-name=jnt_test_lemma
#SBATCH --output=/scratch/10323/asomaya1/exoground/outputs/lemma/test/joint_%j.out
#SBATCH --error=/scratch/10323/asomaya1/exoground/outputs/lemma/test/joint_%j_err.out
#SBATCH --partition=gh-dev
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:30:00

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
    --test /work/10323/asomaya1/vista/code/exo_narration_grounding/ExoGround/train/log/wo_sameview_2025_03_07_00_59_joint_iou_l1_lemma_len64_e6d6_bs16_lr0.0001_view=all_distill=True_pair_ds=False_pair_ds_mode=all_multi_ego=False_narr_rand=False/model/model_best_epoch19.pth.tar \
    --name_prefix $1

#   --use_distill_nce_loss \
#   --same_view_negative \
# --visualize
