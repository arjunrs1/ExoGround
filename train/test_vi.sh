#!/bin/bash
#SBATCH --job-name=ExoGround_ks_all_test
#SBATCH --output=/checkpoint/%u/slurm_logs/exoground/test_vi_%j.out
#SBATCH --error=/checkpoint/%u/slurm_logs/exoground/test_vi_%j.out
#SBATCH --partition=learnfair
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=80
#SBATCH --exclusive
#SBATCH --time=1:00:00
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
    --model view_invariant \
    --use_distill_nce_loss \
    --name_prefix $1 \
    --minimum_four_exo_takes \
    --same_view_negative \
    --test /private/home/arjunrs1/exo_narration_grounding/ExoGround/train/log/min4filt_2024_10_28_05_25_view_invariant_iou_l1_egoexo4d_len64_e6d6_bs16_lr0.0001_view=all_distill=True_pair_ds=False_pair_ds_mode=all_multi_ego=False_narr_rand=False/model/model_best_epoch0.pth.tar