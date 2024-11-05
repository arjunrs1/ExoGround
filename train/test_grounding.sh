#!/bin/bash
#SBATCH --job-name=ExoGround_ks_all_test
#SBATCH --output=/checkpoint/%u/slurm_logs/exoground/test_g_%j.out
#SBATCH --error=/checkpoint/%u/slurm_logs/exoground/test_g_%j.out
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

# srun --label torchrun --nproc_per_node=8 \
#     main_egoexo4d_distributed.py \
#     --batch_size 16 \
#     --epochs 300 \
#     --num_workers 0 \
#     --use_keysteps \
#     --views all \
#     --exos random \
#     --test /private/home/arjunrs1/exo_narration_grounding/ExoGround/train/log/2024_09_10_17_38_init_iou_l1_egoexo4d_len64_e6d6_pos-learned_textpos-0_policy-default_bs16_lr0.0001_audio=False_decoder=True_keysteps=True_view=all_meandur=True_distill=False_pair_ds=False_pair_ds_mode=all_multi_ego=False_narr_rand=False/model/model_best_epoch12.pth.tar \
#     --name_prefix $1 \
#     --visualize

srun --label torchrun --nproc_per_node=8 \
    main_egoexo4d_distributed.py \
    --batch_size 16 \
    --epochs 100 \
    --num_workers 0 \
    --use_keysteps \
    --views all \
    --exos all \
    --model grounding \
    --use_egovlp_features \
    --test /private/home/arjunrs1/exo_narration_grounding/ExoGround/train/log/egovlp_g_2024_10_22_23_14_grounding_iou_l1_egoexo4d_len64_e6d6_bs16_lr0.0001_view=all_distill=False_pair_ds=False_pair_ds_mode=all_multi_ego=False_narr_rand=False/model/model_best_epoch45.pth.tar \
    --name_prefix $1

### --use_distill_nce_loss \
### --curriculum_train \



### 0.1 final phase prop
### /private/home/arjunrs1/exo_narration_grounding/ExoGround/train/log/2024_09_07_23_52_init_iou_l1_egoexo4d_len64_e6d6_pos-learned_textpos-0_policy-default_bs64_lr0.0001_audio=False_decoder=True_keysteps=True_view=multi_meandur=True_distill=False_pair_ds=True_pair_ds_mode=all_multi_ego=True_narr_rand=False/model/model_best_epoch995.pth.tar

### 0.3 final phase prop
### /private/home/arjunrs1/exo_narration_grounding/ExoGround/train/log/2024_09_07_23_59_init_iou_l1_egoexo4d_len64_e6d6_pos-learned_textpos-0_policy-default_bs64_lr0.0001_audio=False_decoder=True_keysteps=True_view=multi_meandur=True_distill=False_pair_ds=True_pair_ds_mode=all_multi_ego=True_narr_rand=False/model/model_best_epoch995.pth.tar

### 0.5 final phase prop
### /private/home/arjunrs1/exo_narration_grounding/ExoGround/train/log/2024_09_08_00_06_init_iou_l1_egoexo4d_len64_e6d6_pos-learned_textpos-0_policy-default_bs64_lr0.0001_audio=False_decoder=True_keysteps=True_view=multi_meandur=True_distill=False_pair_ds=True_pair_ds_mode=all_multi_ego=True_narr_rand=False/model/model_best_epoch865.pth.tar

### view 1 view 2 (no distill):
### /private/home/arjunrs1/exo_narration_grounding/ExoGround/train/log/2024_09_03_05_05_init_iou_l1_egoexo4d_len64_e6d6_pos-learned_textpos-0_policy-default_bs16_lr0.0001_audio=False_decoder=True_keysteps=True_view=all_meandur=True_distill=False_pair_ds=False_pair_ds_mode=all_multi_ego=False_narr_rand=False/model/epoch51.pth.tar


### all:
### /private/home/arjunrs1/exo_narration_grounding/ExoGround/train/log/2024_08_27_16_29_init_iou_l1_egoexo4d_len64_e6d6_pos-learned_textpos-0_policy-default_bs16_lr0.0001_audio=False_decoder=True_keysteps=True_view=all_meandur=True_distill=False_pair_ds=False_pair_ds_mode=all_multi_ego=False_narr_rand=False/model/epoch43.pth.tar

### all rand:
### /private/home/arjunrs1/exo_narration_grounding/ExoGround/train/log/2024_08_27_16_29_init_iou_l1_egoexo4d_len64_e6d6_pos-learned_textpos-0_policy-default_bs16_lr0.0001_audio=False_decoder=True_keysteps=True_view=all_meandur=True_distill=False_pair_ds=False_pair_ds_mode=all_multi_ego=False_narr_rand=True/model/epoch52.pth.tar

### multi egoexo rand
### /private/home/arjunrs1/exo_narration_grounding/ExoGround/train/log/2024_08_27_16_32_init_iou_l1_egoexo4d_len64_e6d6_pos-learned_textpos-0_policy-default_bs16_lr0.0001_audio=False_decoder=True_keysteps=True_view=multi_meandur=True_distill=False_pair_ds=False_pair_ds_mode=all_multi_ego=True_narr_rand=True/model/model_best_epoch86.pth.tar

### multi egoexo unmasked rand 31581327
### /private/home/arjunrs1/exo_narration_grounding/ExoGround/train/log/2024_08_27_16_31_init_iou_l1_egoexo4d_len64_e6d6_pos-learned_textpos-0_policy-default_bs16_lr0.0001_audio=False_decoder=True_keysteps=True_view=multi_meandur=True_distill=False_pair_ds=True_pair_ds_mode=unmasked_multi_ego=True_narr_rand=True/model/epoch47.pth.tar

#### multi egoexo unmasked 31581433
### /private/home/arjunrs1/exo_narration_grounding/ExoGround/train/log/2024_08_27_16_31_init_iou_l1_egoexo4d_len64_e6d6_pos-learned_textpos-0_policy-default_bs16_lr0.0001_audio=False_decoder=True_keysteps=True_view=multi_meandur=True_distill=False_pair_ds=True_pair_ds_mode=unmasked_multi_ego=True_narr_rand=False/model/epoch47.pth.tar

### multi egoexo all 31581455
### /private/home/arjunrs1/exo_narration_grounding/ExoGround/train/log/2024_08_27_16_30_init_iou_l1_egoexo4d_len64_e6d6_pos-learned_textpos-0_policy-default_bs16_lr0.0001_audio=False_decoder=True_keysteps=True_view=multi_meandur=True_distill=False_pair_ds=True_pair_ds_mode=all_multi_ego=True_narr_rand=False/model/epoch47.pth.tar

### multi egoexo 31581595
### /private/home/arjunrs1/exo_narration_grounding/ExoGround/train/log/2024_08_27_16_30_init_iou_l1_egoexo4d_len64_e6d6_pos-learned_textpos-0_policy-default_bs16_lr0.0001_audio=False_decoder=True_keysteps=True_view=multi_meandur=True_distill=False_pair_ds=False_pair_ds_mode=all_multi_ego=True_narr_rand=False/model/epoch104.pth.tar




###OLD ONES:

### exo_ks_distill:
### /private/home/arjunrs1/exo_narration_grounding/ExoGround/train/log/2024_08_12_20_53_init_iou_l1_egoexo4d_len64_e6d6_pos-learned_textpos-0_policy-default_bs16_lr0.0001_audio=False_decoder=True_keysteps=True_view=exo_meandur=True_distill=True/model/epoch99.pth.tar

### all_ks_distill:
### /private/home/arjunrs1/exo_narration_grounding/ExoGround/train/log/2024_08_12_20_53_init_iou_l1_egoexo4d_len64_e6d6_pos-learned_textpos-0_policy-default_bs16_lr0.0001_audio=False_decoder=True_keysteps=True_view=all_meandur=True_distill=True/model/epoch70.pth.tar

### exo_ks:
### /private/home/arjunrs1/exo_narration_grounding/ExoGround/train/log/2024_08_12_20_52_init_iou_l1_egoexo4d_len64_e6d6_pos-learned_textpos-0_policy-default_bs16_lr0.0001_audio=False_decoder=True_keysteps=True_view=exo_meandur=True_distill=False/model/epoch99.pth.tar

### all_ks:
### /private/home/arjunrs1/exo_narration_grounding/ExoGround/train/log/2024_08_12_20_52_init_iou_l1_egoexo4d_len64_e6d6_pos-learned_textpos-0_policy-default_bs16_lr0.0001_audio=False_decoder=True_keysteps=True_view=all_meandur=True_distill=False/model/epoch99.pth.tar

### ego_ks:
### /private/home/arjunrs1/exo_narration_grounding/ExoGround/train/log/2024_08_12_20_40_init_iou_l1_egoexo4d_len64_e6d6_pos-learned_textpos-0_policy-default_bs16_lr0.0001_audio=False_decoder=True_keysteps=True_view=ego_meandur=True_distill=False/model/epoch99.pth.tar
