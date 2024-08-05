import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from einops import rearrange
import os
import sys
sys.path.append('..')
from utils.utils import get_youtube_link, second_to_time
import copy
from tqdm import tqdm
import ffmpeg
from torch.nn.utils.rnn import pad_sequence
import cv2
from moviepy.editor import VideoFileClip

def circulant(tensor, dim):
    """get a circulant version of the tensor along the {dim} dimension.
    The additional axis is appended as the last dimension.
    E.g. 
    circulant(tensor([0,1,2]), dim=0) --> [[0,1,2],[2,0,1],[1,2,0]]"""
    S = tensor.shape[dim]
    tmp = torch.cat([tensor.flip((dim,)), torch.narrow(tensor.flip((dim,)), dim=dim, start=0, length=S-1)], dim=dim)
    return tmp.unfold(dim, S, 1).flip((-1,))


def get_mask_from_time(start_list, end_list, num_timestamp, num_text, device='cuda'):
    """get a binary mask of shape [Batchsize, Num_text, Time].
    For the n-th sentence in the b-th video, 
    the vector [1x1xTime] has value 1 if the text corresponds this time segment."""
    B = len(start_list)
    steps = torch.arange(num_timestamp, device=device)[None,None,:].repeat(B, num_text, 1)
    start_list = pad_sequence(
        [torch.FloatTensor(i) for i in start_list],
        batch_first=True, 
        padding_value=num_timestamp+1e2).to(device, non_blocking=True)
    end_list = pad_sequence(
        [torch.FloatTensor(i) for i in end_list],
        batch_first=True, 
        padding_value=-1e2).to(device, non_blocking=True)
    mask = (start_list[:,:,None] <= steps) * (steps < end_list[:,:,None]) 
    return mask, start_list, end_list


def get_text_pos(start_list, end_list, device='cuda'):
    B = len(start_list)
    start_list = pad_sequence(
        [torch.FloatTensor(i) for i in start_list],
        batch_first=True, padding_value=0).to(device, non_blocking=True)
    end_list = pad_sequence(
        [torch.FloatTensor(i) for i in end_list],
        batch_first=True, padding_value=0).to(device, non_blocking=True)
    return torch.stack((start_list, end_list), dim=-1)

def get_loss(input_data, logits, text_padding_mask, args):
    if args.model in ['init', 'cotrain']:
        grounding_preds = logits['interval_preds']

    device = grounding_preds.device
    
    starts_pred = grounding_preds[:, :, 0]
    ends_pred = grounding_preds[:, :, 1]

    # Extract ground truth and predictions
    starts_gt = input_data['starts'].to(device, non_blocking=True)
    ends_gt = input_data['ends'].to(device, non_blocking=True)

    # Apply the narration padding mask to filter out padded data
    starts_gt_trunc = starts_gt[~text_padding_mask]
    ends_gt_trunc = ends_gt[~text_padding_mask]
    starts_pred_trunc = starts_pred[~text_padding_mask]
    ends_pred_trunc = ends_pred[~text_padding_mask]

    # Compute L1 loss for valid start and end timestamps
    l1_loss_start = F.l1_loss(starts_pred_trunc, starts_gt_trunc, reduction='mean')
    l1_loss_end = F.l1_loss(ends_pred_trunc, ends_gt_trunc, reduction='mean')

    # Compute IoU loss for valid intervals
    intersection = torch.clamp(torch.min(ends_pred_trunc, ends_gt_trunc) - torch.max(starts_pred_trunc, starts_gt_trunc), min=0)
    union = torch.max(ends_pred_trunc, ends_gt_trunc) - torch.min(starts_pred_trunc, starts_gt_trunc)
    iou = intersection / (union + args.iou_loss_eps) #TODO: Why is union zero???
    iou_loss = 1.0 - iou.mean()  # IoU loss is 1 - IoU

    # Store losses in loss_dict
    loss_dict = {}
    loss_dict['Timestamp L1 loss'] = (l1_loss_start + l1_loss_end) / 2
    loss_dict['IoU loss'] = iou_loss
    loss_dict['mean IoU'] = iou.mean()
    if args.test:
        for theta in args.iou_thresholds:
            iou_count = (iou > theta).sum().item()  / (~text_padding_mask).sum().item() #NOTE: We do mean, not sum, bc AverageMeter muls by n
            loss_dict[f'IoU>={theta}'] = iou_count

    # Combine losses into a single loss term if needed
    loss_dict['loss'] = loss_dict['Timestamp L1 loss'] + loss_dict['IoU loss']

    return loss_dict

def visualize(input_data, logits, args, epoch):
    # Open the video file
    sentences = input_data['metadata']['narrations']
    take_ids = input_data['metadata']['video_id']
    exo_cameras = input_data['metadata']['exo_camera']
    start_secs = input_data['metadata']['start_sec']
    
    text_padding_mask = input_data['narration_padding_mask'].cpu().numpy()
    grounding_preds = logits['interval_preds'].cpu().numpy()
    pred_starts = grounding_preds[:, :, 0]
    pred_ends = grounding_preds[:, :, 1]

    base_video_path = "/datasets01/egoexo4d/v2/takes/"
    video_count = 0
    for take_id, exo_cam, start_sec, pred_start, pred_end, narrs, pad_mask  in zip(take_ids, exo_cameras, start_secs, pred_starts, pred_ends, sentences, text_padding_mask):
        if video_count >= min(args.visualization_videos_per_epoch, len(take_ids)):
            break
        video_path = os.path.join(base_video_path, take_id, "frame_aligned_videos", "downscaled", "448", f"{exo_cam}.mp4")
        cap = cv2.VideoCapture(video_path)
        
        # Get the frames per second (fps) of the video
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate the frame number to start and end the clipping
        start_frame = int(start_sec * fps)
        end_frame = int((start_sec + args.seq_len) * fps) #NOTE: We can make the vis videos any duration by changing args.seq_len to args.vis_video_len, just a thought
        
        # Set the starting frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Calculate the absolute start and end frames for each sentence
        start_frames = pred_start * args.seq_len * fps + start_frame
        end_frames = pred_end * args.seq_len * fps + start_frame

        #Create vis dir
        vis_dir = os.path.join(args.log_path, "visualization")
        if not os.path.isdir(vis_dir):
            os.makedirs(vis_dir)

        # Prepare to write the output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_out_path = os.path.join(vis_dir, f'epoch={epoch}_{take_id}_{exo_cam}_start={start_sec}_duration={args.seq_len}.mp4')
        out = cv2.VideoWriter(video_out_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
        
        # Read and process each frame
        current_frame = start_frame
        current_sentence = None
        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Check each sentence to see if it should be displayed on this frame
            for sentence, s_frame, e_frame, mask in zip(narrs, pred_start * args.seq_len * fps + start_frame, pred_end * args.seq_len * fps + start_frame, pad_mask):
                if s_frame <= current_frame < e_frame and not mask:
                    if current_sentence is not None and s_frame < current_sentence[1]:
                        # End the current sentence
                        cv2.putText(frame, current_sentence[0], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        current_sentence = None
                    
                    # Start the new sentence
                    if current_sentence is None:
                        current_sentence = (sentence, e_frame)
                    
                    # Display the sentence
                    cv2.putText(frame, sentence, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    break
            
            # Write the frame to the output video
            out.write(frame)
            current_frame += 1
        
        # Release everything when done
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        video_count += 1
        print(f"Generating epoch {epoch} video: {video_out_path}...")