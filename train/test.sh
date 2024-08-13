#!/bin/bash
#SBATCH --job-name=ExoGround_ks_ego_test
#SBATCH --output=/checkpoint/%u/slurm_logs/exoground/test_%j.out
#SBATCH --error=/checkpoint/%u/slurm_logs/exoground/test_%j.out
#SBATCH --partition=learnfair
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=80
#SBATCH --exclusive
#SBATCH --time=1:00:00
#SBATCH --constraint=volta32gb

### init virtual environment if needed
source activate sounding_narrations

srun --label torchrun --nproc_per_node=8 \
    main_egoexo4d_distributed.py \
    --batch_size 16 \
    --epochs 100 \
    --num_workers 0 \
    --use_keysteps \
    --views ego \
    --test /private/home/arjunrs1/exo_narration_grounding/ExoGround/train/log/2024_08_12_20_40_init_iou_l1_egoexo4d_len64_e6d6_pos-learned_textpos-0_policy-default_bs16_lr0.0001_audio=False_decoder=True_keysteps=True_view=ego_meandur=True_distill=False/model/epoch99.pth.tar \
    --visualize

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
