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
from collections import Counter

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

def get_mode_cam_rank(batch_cam_rank_metadata):
    mode_cam_rank = []
    for cam_rank_metadata in batch_cam_rank_metadata:
        count = Counter(cam_rank_metadata)
        mode = max(count, key=count.get)
        mode_cam_rank.append(mode)
    return mode_cam_rank

def expand_ranks_with_mask(modes, text_padding_mask):
    expanded_modes = []
    for i, mask_row in enumerate(text_padding_mask):
        mode_string = modes[i]
        for mask_value in mask_row:
            if mask_value == 0:
                expanded_modes.append(mode_string)
    return expanded_modes

def get_grounding_loss_reg_head(input_data, logits, text_padding_mask, args):
    grounding_preds = logits['interval_preds']
    per_second_views = input_data['metadata']['per_second_views']
    cam_ranks = get_mode_cam_rank(per_second_views)
    cam_ranks_expanded = expand_ranks_with_mask(cam_ranks, text_padding_mask)  
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
    iou = intersection / (union + args.iou_loss_eps)
    iou_loss = 1.0 - iou.mean()  # IoU loss is 1 - IoU
    loss_dict['IoU loss'] = iou_loss
    loss_dict['mean IoU'] = iou.mean()
    for theta in args.iou_thresholds:
        iou_count = (iou > theta).sum().item()  / (~text_padding_mask).sum().item()
        loss_dict[f'IoU>={theta}'] = iou_count
    if args.test:
        for cam_rank in set(cam_ranks_expanded):
            rank_iou = iou[torch.tensor([r == cam_rank for r in cam_ranks_expanded])]
            for theta in args.iou_thresholds:
                cam_rank_theta_key = f'Rank {cam_rank} IoU>={theta}'
                iou_count = (rank_iou > theta).sum().item() / len(rank_iou)
                loss_dict[cam_rank_theta_key] = {}
                loss_dict[cam_rank_theta_key]['mean'] = iou_count
                loss_dict[cam_rank_theta_key]['count'] = len(rank_iou)
    # Combine losses into a single loss term:
    loss_dict['loss'] = loss_dict['IoU loss'].clone()
    if args.use_center_duration:
        loss_dict['loss'] += loss_dict['Duration L1 loss']
        loss_dict['loss'] += loss_dict['Center L1 loss']
    else:
        loss_dict['loss'] += loss_dict['Timestamp L1 loss']
    return loss_dict, iou

def flatten_list_of_lists(list_of_lists):
    """Flatten a list of lists into a single list."""
    return [item for sublist in list_of_lists for item in sublist]

def get_view_invariant_loss(input_data, logits, args):
    features = logits['high_dim_features'] if not args.test_egovlp else input_data['video_features']
    device = features.device

    view_rank_names = input_data['metadata']['per_second_views']
    ego_seq = input_data['ego_video_features'].to(device, non_blocking=True)
    positive_feature_idxs = input_data['view_rank_label'].to(device, non_blocking=True)
    negative_feature_idxs = input_data['view_rank_neg_label'].to(device, non_blocking=True)
    #valid_views_mask = input_data['valid_views_mask'].to(device, non_blocking=True)
    if args.same_view_negative:
        same_view_neg_idxs = input_data['same_view_neg_idxs'].to(device, non_blocking=True)
        same_view_features = input_data['video_features'].to(device, non_blocking=True)
    else:
        same_view_neg_idxs = None
        same_view_features = None

    info_nce_losses = compute_info_nce_loss_cross_view(features, ego_seq, positive_feature_idxs, negative_feature_idxs, same_view_neg_idxs, same_view_features, only_same_view_negative=args.only_same_view_negative)
    l1_losses, pos_cos_sim, avg_neg_cos_sim = compute_l1_cosine_losses(features, ego_seq, positive_feature_idxs, negative_feature_idxs)

    flat_view_rank_names = flatten_list_of_lists(view_rank_names)
    flat_l1_losses = l1_losses.flatten()
    flat_pos_cosine_similarities = pos_cos_sim.flatten()
    flat_neg_cosine_similarities = avg_neg_cos_sim.flatten()
    flat_info_nce_losses = info_nce_losses.flatten()
    loss_dict = {}
    # Accumulate losses/metrics into proper view rank bin
    for i, view_name in enumerate(flat_view_rank_names):
        prefix = f"Rank {view_name}"
        metrics = ["L1", "pos_cosine", "avg_neg_cosine", "InfoNCE"]
        losses = [flat_l1_losses[i], flat_pos_cosine_similarities[i], flat_neg_cosine_similarities[i], flat_info_nce_losses[i]]
        for metric, loss in zip(metrics, losses):
            key = f"{prefix} {metric}"
            if key not in loss_dict:
                loss_dict[key] = {'total': 0.0, 'count': 0}
            loss_dict[key]['total'] += loss.item()
            loss_dict[key]['count'] += 1
    # Compute the average for each metric
    for key in loss_dict:
        loss_dict[key]['mean'] = loss_dict[key]['total'] / loss_dict[key]['count']

    # Compute mean losses:
    loss_dict["L1 loss"] = l1_losses.mean()
    loss_dict["Pos cosine sim"] = pos_cos_sim.mean() #NOTE: Cosine loss does not contribute to loss
    loss_dict["Avg neg cosine sim"] = avg_neg_cos_sim.mean() #NOTE: Cosine loss does not contribute to loss
    if args.use_distill_nce_loss:
        loss_dict['InfoNCE loss'] = info_nce_losses.mean()
        total_loss = loss_dict['InfoNCE loss'].clone()
    loss_dict['loss'] = total_loss
    return loss_dict, None

def compute_l1_cosine_losses(output_features, video_features, positive_indices, negative_indices):
    """
    Compute the L1 loss and cosine similarity between output features and positive features.
    
    Args:
    - output_features (torch.Tensor): Tensor of shape (batch_size, seq_len, feat_dim)
    - video_features (torch.Tensor): Tensor of shape (batch_size, max_num_views, seq_len, feat_dim)
    - positive_indices (torch.Tensor): Tensor of shape (batch_size, seq_len) containing indices of the positive views
    - positive_indices (torch.Tensor): Tensor of shape (batch_size, seq_len) containing indices of the positive views

    Returns:
    - torch.Tensor: Scalar tensor containing the L1 loss.
    - torch.Tensor: Scalar tensor containing the cosine similarity.
    """
    batch_size, seq_len, feat_dim = output_features.shape
    max_num_views = video_features.size(1)
    # Gather positive features using positive_indices
    positive_features = torch.gather(video_features, 1, positive_indices.unsqueeze(1).unsqueeze(-1).expand(-1, -1, seq_len, feat_dim))
    positive_features = positive_features.squeeze(1)  # Remove the singleton dimension after gather

    # Gather negative features using positive_indices
    negative_features = torch.gather(video_features, 1, negative_indices.unsqueeze(1).unsqueeze(-1).expand(-1, -1, seq_len, feat_dim))
    negative_features = negative_features.squeeze(1)
    
    output_features_normalized = F.normalize(output_features, p=2, dim=2)
    positive_features_normalized = F.normalize(positive_features, p=2, dim=2)
    negative_features_normalized = F.normalize(negative_features, p=2, dim=2)
    # Compute L1 loss per timestep per sample
    l1_loss = F.l1_loss(output_features_normalized, positive_features_normalized, reduction='none').mean(dim=2)
    pos_cosine_similarity = (output_features_normalized * positive_features_normalized).sum(dim=2)
    neg_cosine_similarity = (output_features_normalized * negative_features_normalized).sum(dim=2)
    
    return l1_loss, pos_cosine_similarity, neg_cosine_similarity

def compute_info_nce_loss_cross_view_OLD(output_features, video_features, positive_indices, valid_views_mask, temperature=0.1):
    """
    Compute the InfoNCE loss for a batch of features with multiple views, aligning each output feature with a specific "positive" view.
    
    Args:
    - output_features (torch.Tensor): Tensor of shape (batch_size, seq_length, feature_dim)
    - video_features (torch.Tensor): Tensor of shape (batch_size, num_views, seq_length, feature_dim)
    - positive_indices (torch.Tensor): Tensor of shape (batch_size, seq_length) containing indices of the positive views
    - valid_views_mask (torch.Tensor): Boolean tensor of shape (batch_size, num_views, seq_len) indicating valid views. Set to False everywhere except the positive indices at each step in the sequence.
    - temperature (float): A temperature scaling factor (default 0.1)
    
    Returns:
    - torch.Tensor: Scalar tensor containing the InfoNCE loss.
    """
    batch_size, seq_length, feature_dim = output_features.shape
    # Normalize features
    output_features = F.normalize(output_features, p=2, dim=2)
    video_features = F.normalize(video_features, p=3, dim=3)
    # Compute similarities between all pairs of features across all views and all timesteps
    similarities = torch.einsum('bsf,bvsf->bsv', output_features, video_features) / temperature
    # Mask out similarities for invalid views
    #similarities = similarities.masked_fill(~valid_views_mask.unsqueeze(1).expand(-1, seq_length, -1), float('-inf'))
    #print(f"valid views mask: {valid_views_mask}")
    similarities = similarities.masked_fill(~valid_views_mask.transpose(1,2), float('-inf'))
    #print(f"similarities: {similarities.mean()}")
    #TODO: Should we do log softmax before doing the inf fill? Doesn't make sense otherwise
    log_prob = F.log_softmax(similarities, dim=2)
    #print(f"Sample sim: {similarities[0, 0]} log_prob slice:{log_prob[0, 0]}")
    # Gather log probabilities of positive samples
    log_prob_positive = torch.gather(log_prob, 2, positive_indices.unsqueeze(2)).squeeze(2)
    # Compute the mean of the log probabilities of the positive samples
    #nce_loss = -log_prob_positive.mean()
    return -log_prob_positive

def compute_info_nce_loss_cross_view(output_features, video_features, positive_indices, negative_indices, same_view_neg_idxs=None, same_view_features=None, only_same_view_negative=False, temperature=0.1):
    """
    Compute the InfoNCE loss using positive and negative indices.
    Args:
    - output_features (torch.Tensor): Tensor of shape (batch_size, seq_len, feat_dim)
    - video_features (torch.Tensor): Tensor of shape (batch_size, max_num_views, seq_len, feat_dim)
    - positive_indices (torch.Tensor): Tensor of shape (batch_size, seq_len) containing indices of the positive views
    - negative_indices (torch.Tensor): Tensor of shape (batch_size, seq_len) containing indices of the negative views
    - same_view_neg_idxs (torch.Tensor): Tensor of shape (batch_size, seq_len) containing indices of negative views from the same (source) video
    - temperature (float): A temperature scaling factor (default 0.1)
    Returns:
    - torch.Tensor: Scalar tensor containing the InfoNCE loss.
    """
    # Normalize features
    output_features_normalized = F.normalize(output_features, p=2, dim=2)
    # Gather positive and negative features
    positive_features = torch.gather(
        video_features, 
        1, 
        positive_indices.unsqueeze(1).unsqueeze(-1).expand(-1, -1, output_features.size(1), output_features.size(2))
    ).squeeze(1)
    negative_features = torch.gather(
        video_features, 
        1, 
        negative_indices.unsqueeze(1).unsqueeze(-1).expand(-1, -1, output_features.size(1), output_features.size(2))
    ).squeeze(1)
    # Normalize positive and negative features
    positive_features_normalized = F.normalize(positive_features, p=2, dim=2)
    negative_features_normalized = F.normalize(negative_features, p=2, dim=2)
    # Compute similarities
    pos_similarities = (output_features_normalized * positive_features_normalized).sum(dim=2) / temperature
    neg_similarities = (output_features_normalized * negative_features_normalized).sum(dim=2) / temperature
    # Concatenate positive and negative similarities
    # Concatenate positive and negative similarities
    if same_view_neg_idxs is not None:
        same_view_negative_features = torch.gather(
            same_view_features, 
            1, 
            same_view_neg_idxs.unsqueeze(-1).expand(-1, -1, same_view_features.size(-1))
        )
        same_view_negative_features_normalized = F.normalize(same_view_negative_features, p=2, dim=2)
        same_view_neg_similarities = (output_features_normalized * same_view_negative_features_normalized).sum(dim=2) / temperature
        if only_same_view_negative:
            similarities = torch.cat([pos_similarities.unsqueeze(2), same_view_neg_similarities.unsqueeze(2)], dim=2)
        else:
            similarities = torch.cat([pos_similarities.unsqueeze(2), neg_similarities.unsqueeze(2), same_view_neg_similarities.unsqueeze(2)], dim=2)
    else:
        similarities = torch.cat([pos_similarities.unsqueeze(2), neg_similarities.unsqueeze(2)], dim=2)
    # Compute log probabilities
    log_prob = F.log_softmax(similarities, dim=2)
    # Gather log probabilities of positive samples
    log_prob_positive = log_prob[:, :, 0]
    # Compute the mean of the log probabilities of the positive samples
    nce_loss = -log_prob_positive
    return nce_loss

def compute_info_nce_loss(features1, features2, temperature=0.1):
    """
    Compute the InfoNCE loss between two sequences of features.
    
    Args:
    - features1 (torch.Tensor): Tensor of shape (batch_size, seq_length, feature_dim)
    - features2 (torch.Tensor): Tensor of shape (batch_size, seq_length, feature_dim)
    - temperature (float): A temperature scaling factor (default 0.1)
    
    Returns:
    - torch.Tensor: Scalar tensor containing the InfoNCE loss.
    """

    # Normalize features to get unit vectors
    features1 = F.normalize(features1, p=2, dim=2)
    features2 = F.normalize(features2, p=2, dim=2)
    similarities = torch.bmm(features1, features2.transpose(1, 2)) / temperature
    labels = torch.arange(features2.size(1)).to(features1.device)
    log_prob = F.log_softmax(similarities, dim=2)
    log_prob_positive = log_prob.gather(2, labels.view(1, -1).expand(features1.size(0), -1).unsqueeze(2)).squeeze(2)
    # Compute the mean of the log probabilities of the positive samples
    nce_loss = -log_prob_positive.mean()
    return nce_loss

def get_loss(input_data, logits, text_padding_mask, args):
    if args.model in ['view_invariant']:
        return get_view_invariant_loss(input_data, logits, args)
    elif (args.model in ['grounding']) or ((args.model in ['joint']) and (not args.use_distill_nce_loss)):
        return get_grounding_loss_reg_head(input_data, logits, text_padding_mask, args)
    elif args.model in ['joint']:
        gnd_loss_dict, iou = get_grounding_loss_reg_head(input_data, logits, text_padding_mask, args)
        vi_loss_dict, _ = get_view_invariant_loss(input_data, logits, args)
        # Combine the loss values
        combined_loss = vi_loss_dict['loss'] + gnd_loss_dict['loss']
        # Merge the dictionaries
        combined_loss_dict = {**vi_loss_dict, **gnd_loss_dict}
        # Update the combined loss in the dictionary
        combined_loss_dict['loss'] = combined_loss
        return combined_loss_dict, iou

def visualize(input_data, logits, args, epoch):
    sentences = input_data['metadata']['narrations']
    take_ids = input_data['metadata']['video_id']
    exo_cameras = input_data['metadata']['exo_camera']
    start_secs = input_data['metadata']['start_sec']
    
    grounding_preds = logits['interval_preds']#.cpu().numpy()
    device = grounding_preds.device
    text_padding_mask = input_data['narration_padding_mask'].to(device, non_blocking=True)#.cpu().numpy()
    
    
    if args.use_center_duration:
        centers_gt = input_data['mean'].to(device, non_blocking=True)#.cpu().numpy()
        durations_gt = input_data['duration'].to(device, non_blocking=True)#.cpu().numpy()
        gt_starts = centers_gt - durations_gt / 2
        gt_ends = centers_gt + durations_gt / 2

        centers_pred = grounding_preds[:, :, 0]
        durations_pred = grounding_preds[:, :, 1]
        pred_starts = centers_pred - durations_pred / 2
        pred_ends = centers_pred + durations_pred / 2

        centers_gt_trunc = centers_gt[~text_padding_mask]
        durations_gt_trunc = durations_gt[~text_padding_mask]
        centers_pred_trunc = centers_pred[~text_padding_mask]
        durations_pred_trunc = durations_pred[~text_padding_mask]

        starts_pred_trunc = centers_pred_trunc - durations_pred_trunc / 2
        ends_pred_trunc = centers_pred_trunc + durations_pred_trunc / 2
        starts_gt_trunc = centers_gt_trunc - durations_gt_trunc / 2
        ends_gt_trunc = centers_gt_trunc + durations_gt_trunc / 2
    else:
        gt_starts = input_data['starts']#.cpu().numpy()
        gt_ends = input_data['ends']#.cpu().numpy()
        pred_starts = grounding_preds[:, :, 0]
        pred_ends = grounding_preds[:, :, 1]

    intersection = torch.clamp(torch.min(ends_pred_trunc, ends_gt_trunc) - torch.max(starts_pred_trunc, starts_gt_trunc), min=0)
    union = torch.max(ends_pred_trunc, ends_gt_trunc) - torch.min(starts_pred_trunc, starts_gt_trunc)
    ious = intersection / (union + args.iou_loss_eps)
    ious = ious.cpu().numpy()

    text_padding_mask = text_padding_mask.cpu().numpy()
    grounding_preds = grounding_preds.cpu().numpy()

    base_video_path = "/datasets01/egoexo4d/v2/takes/"
    test_epoch_num = args.test.split(".pth")[0].split("h")[-1]
    assert test_epoch_num.isdigit()
    video_count = 0
    for take_id, exo_cam, start_sec, pred_start, pred_end, gt_start, gt_end, narrs, pad_mask, iou in zip(take_ids, exo_cameras, start_secs, pred_starts, pred_ends, gt_starts, gt_ends, sentences, text_padding_mask, ious):
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
        
        vis_dir = os.path.join(args.log_path, "visualization", test_epoch_num)
        if not os.path.isdir(vis_dir):
            os.makedirs(vis_dir)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_out_path = os.path.join(vis_dir, f'IoU={iou}_{take_id}_{exo_cam}_start={start_sec}.mp4')
        out = cv2.VideoWriter(video_out_path, fourcc, fps, (int(cap_pred.get(3)) * 2, int(cap_pred.get(4))))
        
        current_frame = start_frame
        while current_frame < end_frame:
            ret_pred, frame_pred = cap_pred.read()
            ret_gt, frame_gt = cap_gt.read()
            if not ret_pred or not ret_gt:
                break
            
            # Annotate predicted and ground truth frames
            if ("PX" in args.test) or ("svn" in args.test) or ("joint_nodist" in args.test):
                frame_pred = annotate_frame(frame_pred, narrs, pred_start, pred_end, pad_mask, current_frame, start_frame, fps, args.seq_len, "P")
                frame_gt = annotate_frame(frame_gt, narrs, gt_start, gt_end, pad_mask, current_frame, start_frame, fps, args.seq_len, "GT")
            else:
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

def annotate_frame_OLD(frame, narrs, starts, ends, pad_mask, current_frame, start_frame, fps, seq_len, label):
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
            y = 20  # Fixed position for a single narration
            
            # Draw a background rectangle for better text visibility
            text_size = cv2.getTextSize(f"{label}: {narr}", font, font_scale, 1)[0]
            cv2.rectangle(frame, (x, y - text_size[1] - 2), (x + text_size[0], y + 2), color, -1)
            
            # Draw the text on the frame
            cv2.putText(frame, f"{label}: {narr}", (x, y), font, font_scale, (0, 0, 0), 1)  # Black text
            
            # Break after adding the first valid narration
            break

    return frame


def save_features_to_dir(input_data, logits, args, epoch, low_dim_target_features=None):
    # Define the base directory for saving features
    base_dir = os.path.join(args.log_path, "saved_features")
    try:
        os.makedirs(base_dir, exist_ok=True)
    except OSError as e:
        logging.error(f"Error creating base directory: {e}")
        return
    # Extract metadata
    take_ids = input_data['metadata']['video_id']
    exo_cameras = input_data['metadata']['exo_camera']
    start_secs = input_data['metadata']['start_sec']
    # Extract features
    #input_features = input_data['video_features'].cpu().numpy()
    output_features = logits['low_dim_features'].cpu().numpy()
    if low_dim_target_features is not None:
        print(low_dim_target_features)
        ego_seq = low_dim_target_features.cpu().numpy()
        positive_feature_idxs = input_data['view_rank_label'].cpu().numpy()
    # Iterate over each sample in the batch
    for i, (take_id, exo_cam, start_sec) in enumerate(zip(take_ids, exo_cameras, start_secs)):
        # Create a directory for each video take
        features_dir = os.path.join(base_dir, take_id, exo_cam, str(start_sec))
        try:
            os.makedirs(features_dir, exist_ok=True)
        except OSError as e:
            logging.error(f"Error creating features directory: {e}")
            continue
        # Save input features
        #np.save(os.path.join(features_dir, f"input_features.npy"), input_features[i])
        # Save output features
        np.save(os.path.join(features_dir, f"output_features.npy"), output_features[i])
        
        if low_dim_target_features is not None:
            # Save ego sequence
            np.save(os.path.join(features_dir, f"ego_seq.npy"), ego_seq[i])
            # Save positive feature indices
            np.save(os.path.join(features_dir, f"positive_feature_idxs_epoch.npy"), positive_feature_idxs[i])