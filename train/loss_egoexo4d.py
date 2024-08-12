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

    # Store losses in loss_dict
    loss_dict = {}
    # Extract ground truth and predictions
    if args.use_center_duration:
        # Calculate starts and ends from centers and durations
        centers_pred = grounding_preds[:, :, 0]
        durations_pred = grounding_preds[:, :, 1]
        centers_gt = input_data['mean'].to(device, non_blocking=True)
        durations_gt = input_data['duration'].to(device, non_blocking=True)
        # Apply the narration padding mask to filter out padded data
        centers_gt_trunc = centers_gt[~text_padding_mask]
        durations_gt_trunc = durations_gt[~text_padding_mask]
        centers_pred_trunc = centers_pred[~text_padding_mask]
        durations_pred_trunc = durations_pred[~text_padding_mask]
        # Compute L1 loss for valid centers and durations
        l1_loss_start = F.l1_loss(centers_pred_trunc, centers_gt_trunc, reduction='mean')
        l1_loss_end = F.l1_loss(durations_pred_trunc, durations_gt_trunc, reduction='mean')

        starts_pred_trunc = centers_pred_trunc - durations_pred_trunc / 2
        ends_pred_trunc = centers_pred_trunc + durations_pred_trunc / 2
        starts_gt_trunc = centers_gt_trunc - durations_gt_trunc / 2
        ends_gt_trunc = centers_gt_trunc + durations_gt_trunc / 2
        loss_dict['Center L1 loss'] = l1_loss_start
        loss_dict['Duration L1 loss'] = l1_loss_end
    else:
        starts_pred = grounding_preds[:, :, 0]
        ends_pred = grounding_preds[:, :, 1]
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
        loss_dict['Timestamp L1 loss'] = (l1_loss_start + l1_loss_end) / 2

    # Compute IoU loss for valid intervals
    intersection = torch.clamp(torch.min(ends_pred_trunc, ends_gt_trunc) - torch.max(starts_pred_trunc, starts_gt_trunc), min=0)
    union = torch.max(ends_pred_trunc, ends_gt_trunc) - torch.min(starts_pred_trunc, starts_gt_trunc)
    iou = intersection / (union + args.iou_loss_eps) #TODO: Why is union zero???
    iou_loss = 1.0 - iou.mean()  # IoU loss is 1 - IoU
    loss_dict['IoU loss'] = iou_loss
    loss_dict['mean IoU'] = iou.mean()
    if args.use_distill_nce_loss and 'distill_infonce_loss' in logits.keys():
        loss_dict['InfoNCE loss'] = logits['distill_infonce_loss']
    if args.test:
        for theta in args.iou_thresholds:
            iou_count = (iou > theta).sum().item()  / (~text_padding_mask).sum().item() #NOTE: We do mean, not sum, bc AverageMeter muls by n
            loss_dict[f'IoU>={theta}'] = iou_count
    # Combine losses into a single loss term:
    loss_dict['loss'] = loss_dict['IoU loss']
    if args.use_center_duration:
        loss_dict['loss'] += loss_dict['Duration L1 loss']
        loss_dict['loss'] += loss_dict['Center L1 loss']
    else:
        loss_dict['loss'] += loss_dict['Timestamp L1 loss']
    if args.use_distill_nce_loss and 'InfoNCE loss' in loss_dict.keys():
        loss_dict['loss'] += loss_dict['InfoNCE loss']
    return loss_dict

def visualize(input_data, logits, args, epoch):
    sentences = input_data['metadata']['narrations']
    take_ids = input_data['metadata']['video_id']
    exo_cameras = input_data['metadata']['exo_camera']
    start_secs = input_data['metadata']['start_sec']
    
    text_padding_mask = input_data['narration_padding_mask'].cpu().numpy()
    grounding_preds = logits['interval_preds'].cpu().numpy()
    
    if args.use_center_duration:
        centers_gt = input_data['mean'].cpu().numpy()
        durations_gt = input_data['duration'].cpu().numpy()
        gt_starts = centers_gt - durations_gt / 2
        gt_ends = centers_gt + durations_gt / 2

        centers_pred = grounding_preds[:, :, 0]
        durations_pred = grounding_preds[:, :, 1]
        pred_starts = centers_pred - durations_pred / 2
        pred_ends = centers_pred + durations_pred / 2
    else:
        gt_starts = input_data['starts'].cpu().numpy()
        gt_ends = input_data['ends'].cpu().numpy()
        pred_starts = grounding_preds[:, :, 0]
        pred_ends = grounding_preds[:, :, 1]

    base_video_path = "/datasets01/egoexo4d/v2/takes/"
    video_count = 0
    for take_id, exo_cam, start_sec, pred_start, pred_end, gt_start, gt_end, narrs, pad_mask in zip(take_ids, exo_cameras, start_secs, pred_starts, pred_ends, gt_starts, gt_ends, sentences, text_padding_mask):
        if video_count >= min(args.visualization_videos_per_epoch, len(take_ids)):
            break
        video_path = os.path.join(base_video_path, take_id, "frame_aligned_videos", "downscaled", "448", f"{exo_cam}.mp4")
        cap_pred = cv2.VideoCapture(video_path)
        cap_gt = cv2.VideoCapture(video_path)
        
        fps = cap_pred.get(cv2.CAP_PROP_FPS)
        start_frame = int(start_sec * fps)
        end_frame = int((start_sec + args.seq_len) * fps)
        
        cap_pred.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        cap_gt.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        vis_dir = os.path.join(args.log_path, "visualization")
        if not os.path.isdir(vis_dir):
            os.makedirs(vis_dir)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_out_path = os.path.join(vis_dir, f'epoch={epoch}_{take_id}_{exo_cam}_start={start_sec}_duration={args.seq_len}.mp4')
        out = cv2.VideoWriter(video_out_path, fourcc, fps, (int(cap_pred.get(3)) * 2, int(cap_pred.get(4))))
        
        current_frame = start_frame
        while current_frame < end_frame:
            ret_pred, frame_pred = cap_pred.read()
            ret_gt, frame_gt = cap_gt.read()
            if not ret_pred or not ret_gt:
                break
            
            # Annotate predicted and ground truth frames
            frame_pred = annotate_frame(frame_pred, narrs, pred_start, pred_end, pad_mask, current_frame, start_frame, fps, args.seq_len, "P")
            frame_gt = annotate_frame(frame_gt, narrs, gt_start, gt_end, pad_mask, current_frame, start_frame, fps, args.seq_len, "GT")
            
            # Stitch frames side by side
            combined_frame = np.hstack((frame_pred, frame_gt))
            
            out.write(combined_frame)
            current_frame += 1
        
        cap_pred.release()
        cap_gt.release()
        out.release()
        cv2.destroyAllWindows()
        video_count += 1
        print(f"Generating epoch {epoch} dual video: {video_out_path}...")

def annotate_frame(frame, narrs, starts, ends, pad_mask, current_frame, start_frame, fps, seq_len, label):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (255, 255, 255)  # White color for text
    
    # Loop through each narration and its corresponding start and end times
    for i in range(len(starts)):
        if pad_mask[i] == 1:
            continue
        
        # Convert relative start and end times to frame numbers
        start_frame_num = int(starts[i] * seq_len * fps) + start_frame
        end_frame_num = int(ends[i] * seq_len * fps) + start_frame
        
        # Check if the current frame is within the interval
        if start_frame_num <= current_frame < end_frame_num:
            narr = narrs[i]
            
            # Calculate the position of the text on the frame
            x = 10
            y = 20 + (i * 20)
            
            # Draw a background rectangle for better text visibility
            text_size = cv2.getTextSize(f"{label}: {narr}", font, font_scale, 1)[0]
            cv2.rectangle(frame, (x, y - text_size[1] - 2), (x + text_size[0], y + 2), color, -1)
            
            # Draw the text on the frame
            cv2.putText(frame, f"{label}: {narr}", (x, y), font, font_scale, (0, 0, 0), 1)  # Black text

    return frame