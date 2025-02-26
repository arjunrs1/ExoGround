import os
import sys
import torch
from torch.utils import data 
from tensorboardX import SummaryWriter
import numpy as np 
import random 
from tqdm import tqdm
import json
import time
import math
import functools
import torch.cuda.amp as amp 
from config_egoexo4d import parse_args, set_path
from loss_egoexo4d import get_loss, get_mask_from_time, get_text_pos, visualize, save_features_to_dir
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

sys.path.append('../data/')
from data.loader_egoexo4d import EgoExo4DDataLoader
sys.path.append('../model/')
from model.exo_ground_model import ExoGroundingTransformer
from model.keystep_ground_model import GroundingModel
from model.vi_encoder import ViewInvariantEncoder, ViewInvariantMLP
sys.path.append('../')
import utils.tensorboard_utils as TB
from utils.data_utils import DataLoaderBG
from utils.train_utils import clip_gradients
from utils.utils import AverageMeter, save_checkpoint, neq_load_customized, \
calc_topk_accuracy, ProgressMeter, neq_load_customized, save_runtime_checkpoint, MovingAverage

class CurriculumDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, curriculum_epoch=0, max_epochs=None, start_frac=0.50, end_epoch_frac=0.75):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=False, seed=seed)
        self.shuffle = shuffle
        self.curriculum_epoch = curriculum_epoch
        self.max_epochs = max_epochs
        self.start_frac = start_frac
        self.end_epoch_frac = end_epoch_frac

    def __iter__(self):
        # Calculate the curriculum progress
        curriculum_progress = max(self.start_frac, min(1.0, self.start_frac + (self.curriculum_epoch / (self.max_epochs * self.end_epoch_frac)) * self.end_epoch_frac))
        num_samples = int(curriculum_progress * len(self.dataset))
        indices = list(range(len(self.dataset)))

        # Select the first num_samples based on the curriculum progress
        indices = indices[:num_samples]

        # Shuffle if required
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.tensor(indices).tolist()
            indices = torch.randperm(len(indices), generator=g).tolist()

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # Subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        self.curriculum_epoch = epoch

def get_phase_old(epoch, total_epochs, num_phases):
    # Divide the total epochs by the number of phases to determine the length of each phase
    phase_length = total_epochs // num_phases
    current_phase = epoch // phase_length
    return current_phase

def get_phase(epoch, total_epochs, num_phases, final_phase_proportion):    
    # Calculate the number of epochs for the final phase
    final_phase_length = int(total_epochs * final_phase_proportion)
    # Calculate the number of epochs for the other phases
    other_phases_length = (total_epochs - final_phase_length) // (num_phases - 1)
    # Determine the current phase
    if epoch < (total_epochs - final_phase_length):
        current_phase = epoch // other_phases_length
    else:
        current_phase = num_phases - 1  # the final phase
    
    return current_phase

def train(loader, model, optimizer, lr_scheduler, grad_scaler, device, epoch, args):
    batch_time = AverageMeter('Time',':.2f')
    data_time = AverageMeter('Data',':.2f')
    losses = AverageMeter('Loss',':.4f')
    rank = args.rank  # Assuming rank is passed in args
    if rank == 0:
        progress = ProgressMeter(
            len(loader), [batch_time, data_time, losses],
            prefix='Epoch:[{}]'.format(epoch))
    model.train()

    end = time.time()
    tic = time.time()
    optimizer.zero_grad()

    for idx, input_data in enumerate(loader):
        data_time.update(time.time() - end)

        video_seq = input_data['video_features'].to(device, non_blocking=True)
        video_padding_mask = input_data['video_padding_mask'].to(device, non_blocking=True)
        if 'audio_features' in input_data.keys():
            audio_seq = input_data['audio_features'].to(device, non_blocking=True)
            audio_padding_mask = input_data['audio_padding_mask'].to(device, non_blocking=True).bool()
        else:
            audio_seq = None
            audio_padding_mask = None
        text_embed =  input_data['narration_features'].to(device, non_blocking=True)
        text_padding_mask = input_data['narration_padding_mask'].to(device, non_blocking=True)
        if args.use_distill_nce_loss and 'ego_video_features' in input_data.keys():
            ego_seq = input_data['ego_video_features'].to(device, non_blocking=True)
        else:
            ego_seq = None
        if args.views == "multi" and 'view_available_mask' in input_data.keys():
            view_mask = input_data['view_available_mask'].to(device, non_blocking=True)
        else:
            view_mask = None

        B, T, _ = video_seq.shape

        # forward pass
        with amp.autocast():
            logits = model(video_seq, text_embed, 
                    video_padding_mask=video_padding_mask.bool(), 
                    lang_padding_mask=text_padding_mask.bool(),
                    audio_embed=audio_seq,
                    audio_padding_mask=audio_padding_mask,
                    egocentric_video_embed=ego_seq,
                    view_mask=view_mask
                    )
            if args.model in ['cotrain']:
                logits_ema = model.forward_from_ema(
                    video_seq, text_embed, 
                    video_padding_mask=video_padding_mask.bool(), 
                    lang_padding_mask=text_padding_mask.bool(),
                    audio_embed=audio_seq,
                    audio_padding_mask=audio_padding_mask,
                    egocentric_video_embed=ego_seq
                )
                logits = {**logits, **{f'ema-{k}':v for k,v in logits_ema.items()}}

            loss_dict, _ = get_loss(input_data=input_data,
                                 text_padding_mask=text_padding_mask,
                                 logits=logits, 
                                 args=args)

        loss = loss_dict['loss']
        if (not torch.isinf(loss)) and (not torch.isnan(loss)):
            losses.update(loss.item(), B)

        # backward pass
        grad_scaler.scale(loss).backward()
        if idx % args.backprop_freq == 0:
            grad_scaler.unscale_(optimizer)
            if args.clip_grad > 0:
                _ = clip_gradients(model, clip_grad=args.clip_grad)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            optimizer.zero_grad()


        # log stats
        if rank == 0 and args.iteration % 5 == 0:
            for k, v in loss_dict.items():
                value = v if k.startswith('IoU>=') else (v['mean'] if k.startswith('Rank') else v.item())
                args.train_plotter.add_data(f'train/{k}', value, args.iteration)
            args.train_plotter.add_data('train/lr', lr_scheduler.get_last_lr()[0], args.iteration)
            args.train_plotter.add_data('device/sps', 1/(time.time()-end), args.iteration)
            args.train_plotter.log_gpustat(step=args.iteration)
            args.train_plotter.writer.flush()

        if args.prof is not None:
            args.prof.step()

        batch_time.update(time.time() - end)
        if rank == 0:
            progress.display(idx)
        lr_scheduler.step(args.iteration)
        end = time.time()
        args.iteration += 1

        if rank == 0:
            # save runtime ckpt (for long-schedule training)
            if args.iteration % args.runtime_save_iter == 0:
                print('saving runtime checkpoint ...')
                state_dict = model.state_dict()
                save_dict = {
                    'epoch': epoch,
                    'state_dict': state_dict,
                    'best_acc': 1e5,
                    'optimizer': optimizer.state_dict(),
                    'iteration': args.iteration}
                save_runtime_checkpoint(save_dict, 
                    filename=os.path.join(args.model_path, 'runtime.pth.tar'))
                
    if rank == 0:
        print(f'epoch {epoch} finished, takes {time.time() - tic} seconds')
        args.train_plotter.add_data('train/total_epoch_loss', losses.avg, epoch)
    return losses.avg

@torch.no_grad()
def evaluate(loader, model, device, epoch, args):
    rank = args.rank
    model.eval()
    batch_time = AverageMeter('Time', ':.2f')
    metric_meters = {} # Dictionary to hold AverageMeters for each metric
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(loader), [batch_time, losses],
        prefix="Test: " if args.test else "Val: ")

    end = time.time()
    vis_this_epoch = False
    if args.test:
        save_list = []
    for idx, input_data in enumerate(loader):
        video_seq = input_data['video_features'].to(device, non_blocking=True)
        video_padding_mask = input_data['video_padding_mask'].to(device, non_blocking=True)
        if 'audio_features' in input_data.keys():
            audio_seq = input_data['audio_features'].to(device, non_blocking=True)
            audio_padding_mask = input_data['audio_padding_mask'].to(device, non_blocking=True).bool()
        else:
            audio_seq = None
            audio_padding_mask = None
        text_embed = input_data['narration_features'].to(device, non_blocking=True)
        text_padding_mask = input_data['narration_padding_mask'].to(device, non_blocking=True)
        if args.use_distill_nce_loss and 'ego_video_features' in input_data.keys():
            ego_seq = input_data['ego_video_features'].to(device, non_blocking=True)
        else:
            ego_seq = None
        if args.views == "multi" and 'view_available_mask' in input_data.keys():
            view_mask = input_data['view_available_mask'].to(device, non_blocking=True)
        else:
            view_mask = None

        # Forward pass
        logits = model(video_seq, text_embed,
                       video_padding_mask=video_padding_mask.bool(),
                       lang_padding_mask=text_padding_mask.bool(),
                       audio_embed=audio_seq,
                       audio_padding_mask=audio_padding_mask,
                       egocentric_video_embed=ego_seq,
                       view_mask=view_mask)

        loss_dict, ious = get_loss(input_data=input_data,
                                 text_padding_mask=text_padding_mask,
                                 logits=logits, 
                                 args=args)
    
        # Update IoU threshold metrics
        if args.model in ['grounding', 'joint']:
            for theta in args.iou_thresholds:
                key = f'IoU>={theta}'
                if key in loss_dict:
                    if key not in metric_meters:
                        metric_meters[key] = AverageMeter(key, ':.4f')
                    metric_meters[key].update(loss_dict[key], int(~text_padding_mask.sum()))

        # Update other metrics
        for key, value in loss_dict.items():
            if "Rank" in key:
                if key not in metric_meters:
                    metric_meters[key] = AverageMeter(key, ':.4f')
                metric_meters[key].update(value['mean'], value['count'])
            elif not key.startswith('IoU>='):
                if key not in metric_meters:
                    metric_meters[key] = AverageMeter(key, ':.4f')
                metric_meters[key].update(value.item(), video_seq.size(0))

        loss = loss_dict['loss']
        losses.update(loss.item(), video_seq.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.test and args.model in ['grounding', 'joint']:
            save_list.append({
                        'loss_dict': ious.cpu().detach().tolist(),
                        'metadata': {"narration": input_data['metadata']['narrations'], 
                                    "video_id": input_data['metadata']['video_id'],
                                    "cam_id": input_data['metadata']['exo_camera'],
                                    "narr_ranks": input_data['metadata']['narr_ranks']}
                    })

        if (rank == 0) and (idx % args.print_freq == 0):
            progress.display(idx)

        # Visualization call
        if rank == 0 and (args.visualize and idx % args.vis_freq == 0):# and not vis_this_epoch):
            if args.model in ['grounding', 'joint']:
                visualize(input_data, logits, args, epoch)
                vis_this_epoch = True

        if rank == 0 and args.save_features:
            if args.use_distill_nce_loss:
                output_target_features = []
                # Iterate over each sample in the batch
                for i in range(ego_seq.size(0)):
                    sample_outputs = []
                    # Iterate over each view
                    for j in range(ego_seq.size(1)):
                        view = ego_seq[i, j]
                        # Check if the view is non-zero
                        if view.abs().sum() > 0:
                            # Reshape and pass through the model
                            with torch.no_grad():
                                low_dim_target = model.module.get_low_dim_target_features(
                                    view.unsqueeze(0),  # Pass only the current view
                                    torch.zeros(view.size(0), dtype=torch.bool).unsqueeze(0)
                                )['low_dim_features']
                            sample_outputs.append(low_dim_target.squeeze(0))  # Remove batch dimension if needed
                        else:
                            sample_outputs.append(torch.zeros((args.seq_len, args.feature_dim), device=view.device))
                    # Stack the outputs for the current sample
                    sample_outputs = torch.stack(sample_outputs)
                    output_target_features.append(sample_outputs)
                output_target_features = torch.stack(output_target_features)
                #print("OUTPUT SHAPE:")
                #print(output_target_features.shape)
            else:
                output_target_features = None
            save_features_to_dir(input_data, logits, args, epoch, low_dim_target_features=output_target_features)
            print(f"Saved output features to {args.log_path}")

    if rank == 0:
        print(f' * Loss {losses.avg:.4f}')
        # Log all average values to TensorBoard
        for key, meter in metric_meters.items():
            args.train_plotter.add_data(f'val/{key}', meter.avg, epoch)
        
        if args.test and args.model in ['grounding', 'joint']:
            with open(os.path.join(args.log_path, f'test_results_epoch_{epoch}.json'), 'w') as f:
                json.dump(save_list, f)
    
    if args.test:
        return losses.avg, metric_meters
    else:
        return losses.avg

def setup(args):
    # Initialize distributed environment
    #args.distributed = int(os.environ.get('SLURM_JOB_NUM_NODES', "1")) > 1
    args.distributed = torch.distributed.is_available()
    print(f"distributed available: {args.distributed}")
    print("YES IT REACHED")
    if args.distributed:
        torch.distributed.init_process_group(backend='nccl')
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()
        local_rank = int(os.environ.get('LOCAL_RANK', args.rank % torch.cuda.device_count()))
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # CUDA setting
    if torch.cuda.is_available():
        num_gpu = torch.cuda.device_count()
        args.num_gpu = num_gpu
        if args.rank == 0:
            print('=> Effective BatchSize = %d' % args.batch_size)
    else:
        args.num_gpu = 0
        device = torch.device('cpu')
        print('=> Run with CPU')

    # general setting
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    args.iteration = 1
    args.log_path, args.model_path, args.exp_path = set_path(args)

    # tensorboard monitor in the background threads
    if args.rank == 0:
        writer_train = SummaryWriter(logdir=os.path.join(args.log_path, 'train'), flush_secs=60)
        args.train_plotter = TB.PlotterThread(writer_train)
    return device

def get_dataset(args):
    D = EgoExo4DDataLoader
    train_dataset = D(
        split="train",
        duration=args.seq_len,
        hop_length=args.seq_hop,
        use_audio=args.use_audio,
        use_keysteps=args.use_keysteps,
        views=args.views,
        use_distill_nce_loss=args.use_distill_nce_loss,
        use_center_duration=args.use_center_duration,
        multi_view_egoexo=args.multi_view_egoexo,
        num_max_views=args.num_max_views,
        randomize_narration_order=args.randomize_narration_order,
        curriculum_train=args.curriculum_train,
        sorted_curr_train=args.sorted_curr_train,
        stitched_best_exo_distill=args.stitched_best_exo_distill,
        model=args.model,
        exo_mode=args.exos,
        minimum_four_exo_takes=args.minimum_four_exo_takes,
        same_view_negative=args.same_view_negative,
        use_tf_video_features=args.use_tf_video_features,
        reverse_ranking=args.reverse_ranking,
        randomize_ranking=args.randomize_ranking)
    val_dataset = D(
        split="val",
        duration=args.seq_len,
        hop_length=args.seq_hop,
        use_audio=args.use_audio,
        use_keysteps=args.use_keysteps,
        views="exo" if args.model in ['grounding', 'joint'] else "all",
        use_distill_nce_loss=args.use_distill_nce_loss,
        use_center_duration=args.use_center_duration,
        multi_view_single_exo_inference=(args.views=="multi"),
        multi_view_egoexo=args.multi_view_egoexo,
        num_max_views=args.num_max_views,
        stitched_best_exo_distill=True if args.model in ['view_invariant'] else False, #TODO: Is this right??? Should we be fixing best_exo_distill in VI evaluation?
        model=args.model,
        randomize_narration_order=False,
        minimum_four_exo_takes=args.minimum_four_exo_takes,
        same_view_negative=args.same_view_negative,
        use_tf_video_features=args.use_tf_video_features,
        reverse_ranking=args.reverse_ranking,
        randomize_ranking=args.randomize_ranking)

    if args.views == "all" and args.curriculum_train and args.sorted_curr_train in ['sorted']:
        train_sampler = CurriculumDistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.rank, max_epochs=args.epochs, start_frac=args.start_frac, end_epoch_frac=args.end_epoch_frac)
    else:
        train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=args.world_size, rank=args.rank)

    print("Loading train dataset...")
    train_loader = DataLoaderBG(train_dataset,
        batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=train_dataset.collate_fn, pin_memory=True, drop_last=True,
        shuffle=(train_sampler is None), sampler=train_sampler, 
    )

    print("Loading val dataset...")
    val_loader = DataLoaderBG(val_dataset,
        batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=val_dataset.collate_fn, pin_memory=True, drop_last=True,
        shuffle=(val_sampler is None), sampler=val_sampler, 
    )
    return train_dataset, val_dataset, train_loader, val_loader, train_sampler

def get_test_dataset(args):
    D = EgoExo4DDataLoader
    test_dataset = D(
        split="test",
        duration=args.seq_len,
        hop_length=args.seq_hop,
        use_audio=args.use_audio,
        use_keysteps=args.use_keysteps,
        views="exo" if args.model in ['grounding', 'joint'] else "all", #fix testing on all exo views
        use_distill_nce_loss=args.use_distill_nce_loss,
        use_center_duration=args.use_center_duration,
        multi_view_single_exo_inference=(args.views=="multi"),
        multi_view_egoexo=args.multi_view_egoexo,
        num_max_views=args.num_max_views,
        stitched_best_exo_distill=True if args.model in ['view_invariant'] else False,
        model=args.model,
        randomize_narration_order=False,
        minimum_four_exo_takes=args.minimum_four_exo_takes,
        same_view_negative=args.same_view_negative,
        use_tf_video_features=args.use_tf_video_features,
        reverse_ranking=False,
        randomize_ranking=False)

    test_sampler = DistributedSampler(test_dataset, num_replicas=args.world_size, rank=args.rank)

    print("Loading test dataset...")
    test_loader = DataLoaderBG(test_dataset,
        batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=test_dataset.collate_fn, pin_memory=True, drop_last=True, #TODO: Ditto comment above
        shuffle=False, sampler=test_sampler, 
    )
    return test_loader


def optim_policy(model, args, policy='default'):
    params = []
    no_decay = ['.ln_', '.bias', '.logit_scale', '.entropy_scale']
    param_group_no_decay = []
    param_group_with_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if policy == 'default':
            if any([i in name for i in no_decay]):
                param_group_no_decay.append(param)
            else:
                param_group_with_decay.append(param)
        elif policy == 'bce':
            if 'binary_head' in name:
                if any([i in name for i in no_decay]):
                    param_group_no_decay.append(param)
                else:
                    param_group_with_decay.append(param)
            else:
                param.requires_grad = False
                continue

    params.append({'params': param_group_no_decay, 'lr': args.lr, 'weight_decay': 0.0})
    params.append({'params': param_group_with_decay, 'lr': args.lr, 'weight_decay': args.wd})
    return params


def main(args):
    device = setup(args)
    rank = args.rank
    # pre-setup: overwritting
    if args.model == 'grounding':
        pass #TODO: This does nothing right now...eventually add functionality for stage 1/stage 2 training

    #Ensure we are not including ego in backbone and distilling ego into views
    assert not (args.multi_view_egoexo and args.use_distill_nce_loss)

    if args.use_pairwise_distill_nce_loss:
        assert args.views == "multi"

    if args.use_tf_video_features:
        args.video_feature_dim = 768
        assert not args.use_distill_nce_loss

    if args.model in ['view_invariant']:
        assert args.use_distill_nce_loss

    if args.model in ['grounding']:
        assert not args.use_distill_nce_loss

    if args.only_same_view_negative:
        assert args.same_view_negative

    if args.exos != "all":
        assert not args.use_distill_nce_loss #We shouldn't be distilling for the exos experiments

    #ensure we are only including ego view in multi-view training if we are doing multi-view training
    if args.multi_view_egoexo:
        assert args.views == "multi"

    if args.test_egovlp:
        assert args.test

    args.num_max_views = 1 if not args.views == "multi" else args.num_max_views
    if args.multi_view_egoexo:
        args.num_max_views +=1 

    if not args.test:
        _, _, train_loader, val_loader, train_sampler = get_dataset(args)

    if args.test:
        #ensure no overlapping segments for full video inference
        args.seq_hop = args.seq_len 
        assert args.seq_hop == args.seq_len
         
        test_loader = get_test_dataset(args)

    

    ### Model ###
    if args.model in ['grounding']:
        if not args.use_egovlp_features:
            vi_model = ViewInvariantMLP(num_encoder_layers=args.num_encoder_layers,
                            num_decoder_layers=args.num_decoder_layers,
                            use_decoder=args.use_decoder,
                            sim=args.sim,
                            pos_enc=args.pos_enc,
                            use_text_pos_enc=args.use_text_pos_enc,
                            use_audio=args.use_audio,
                            video_embed_dim=args.video_feature_dim,
                            text_embed_dim=args.text_feature_dim,
                            audio_embed_dim=args.audio_feature_dim,
                            feature_dim=args.feature_dim,
                            use_distill_nce_loss=args.use_distill_nce_loss,
                            multi_view= args.views == "multi",
                            num_max_views=args.num_max_views,
                            use_pairwise_distill_nce_loss=args.use_pairwise_distill_nce_loss,
                            pairwise_distill_mode=args.pairwise_distill_mode
            )
            vi_checkpoint = torch.load(get_model_card(args.vi_encoder_path), map_location='cpu')
            vi_model.to(device)
            vi_model = DDP(vi_model, device_ids=[rank], find_unused_parameters=False)  # Wrap model with DDP
            vi_model_without_dp = vi_model
            try:
                vi_model_without_dp.load_state_dict(vi_checkpoint['state_dict'])
            except:
                vi_model_without_dp.load_state_dict(vi_checkpoint['state_dict'], strict=False)
                if rank == 0:
                    print('[WARNING] Non-Equal load for testing!')
            for param in vi_model.parameters():
                param.requires_grad = False
            vi_model.eval()
        else:
            vi_model = None
        model = GroundingModel(num_encoder_layers=args.num_encoder_layers,
                        num_decoder_layers=args.num_decoder_layers,
                        use_decoder=args.use_decoder,
                        sim=args.sim,
                        pos_enc=args.pos_enc,
                        use_text_pos_enc=args.use_text_pos_enc,
                        use_audio=args.use_audio,
                        video_embed_dim=args.video_feature_dim,
                        text_embed_dim=args.text_feature_dim,
                        audio_embed_dim=args.audio_feature_dim,
                        feature_dim=args.feature_dim,
                        use_distill_nce_loss=args.use_distill_nce_loss,
                        multi_view= args.views == "multi",
                        num_max_views=args.num_max_views,
                        use_pairwise_distill_nce_loss=args.use_pairwise_distill_nce_loss,
                        pairwise_distill_mode=args.pairwise_distill_mode,
                        vi_encoder=vi_model
        )
    elif args.model in ['view_invariant']:
        model = ViewInvariantMLP(num_encoder_layers=args.num_encoder_layers,
                        num_decoder_layers=args.num_decoder_layers,
                        use_decoder=args.use_decoder,
                        sim=args.sim,
                        pos_enc=args.pos_enc,
                        use_text_pos_enc=args.use_text_pos_enc,
                        use_audio=args.use_audio,
                        video_embed_dim=args.video_feature_dim,
                        text_embed_dim=args.text_feature_dim,
                        audio_embed_dim=args.audio_feature_dim,
                        feature_dim=args.feature_dim,
                        use_distill_nce_loss=args.use_distill_nce_loss,
                        multi_view= args.views == "multi",
                        num_max_views=args.num_max_views,
                        use_pairwise_distill_nce_loss=args.use_pairwise_distill_nce_loss,
                        pairwise_distill_mode=args.pairwise_distill_mode
        )
    elif args.model in ['joint']:
        model = ExoGroundingTransformer(num_encoder_layers=args.num_encoder_layers,
                        num_decoder_layers=args.num_decoder_layers,
                        use_decoder=args.use_decoder,
                        sim=args.sim,
                        pos_enc=args.pos_enc,
                        use_text_pos_enc=args.use_text_pos_enc,
                        use_audio=args.use_audio,
                        video_embed_dim=args.video_feature_dim,
                        text_embed_dim=args.text_feature_dim,
                        audio_embed_dim=args.audio_feature_dim,
                        feature_dim=args.feature_dim,
                        use_distill_nce_loss=args.use_distill_nce_loss,
                        multi_view= args.views == "multi",
                        num_max_views=args.num_max_views,
                        use_pairwise_distill_nce_loss=args.use_pairwise_distill_nce_loss,
                        pairwise_distill_mode=args.pairwise_distill_mode
        )

    model.to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=args.model in ['grounding', 'joint'])  # Wrap model with DDP
    model_without_dp = model

    ### optimizer ###
    params = optim_policy(model, args, args.optim_policy)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    if not args.test and rank == 0:
        print('\n===========Check Grad============')
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.requires_grad)
        print('=================================\n')

    ### test ###
    if args.test:
        if rank == 0:
            args.test = get_model_card(args.test)
            print(f"Loading model from {args.test} for testing...")
        checkpoint = torch.load(args.test, map_location='cpu')
        state_dict = checkpoint['state_dict']
        epoch = checkpoint['epoch']
        try:
            model_without_dp.load_state_dict(state_dict)
        except:
            model_without_dp.load_state_dict(state_dict, strict=False)
            if rank == 0:
                print('[WARNING] Non-Equal load for testing!')

        model.eval()

        if rank == 0:
            print('Start Inference ...')
        _, all_metric_meters = evaluate(test_loader, model, device, epoch, args)
        if rank == 0:
            if args.model in ['grounding', 'joint']:
                mean_iou = []
                for theta in args.iou_thresholds:
                    theta_threshold_name = f'IoU>={theta}'
                    theta_threshold_value = all_metric_meters[theta_threshold_name].avg
                    print(f"{theta_threshold_name}: {theta_threshold_value:.4f}")
                    mean_iou.append(theta_threshold_value)
                print(f"Mean IoU: {np.array(mean_iou).mean():.4f}")
                print()
                print("Per view grounding metrics:")
                print()
                for view_rank in ['0', '1', '2', '3', '4', '5', '6', 'unk', 'ego']:
                    for theta in args.iou_thresholds:
                        theta_threshold_rank_name = f'Rank {view_rank} IoU>={theta}'
                        if theta_threshold_rank_name in all_metric_meters.keys():
                            theta_threshold_rank_value = all_metric_meters[theta_threshold_rank_name].avg
                            print(f"{theta_threshold_rank_name}: {theta_threshold_rank_value:.4f}")
                    print()
            if args.model in ['view_invariant', 'joint']:
                for view_rank in ['0', '1', '2', '3', '4', '5', '6', 'unk', 'ego']:
                    for key, meter in all_metric_meters.items():
                        key_rank_pair = f"Rank {view_rank}"
                        if key_rank_pair in key:
                            stat_avg = meter.avg
                            print(f"{key}: {stat_avg:.4f}")
                    print()

        dist.barrier()
        sys.exit(0)

    ### restart ###
    best_val_loss = 1e5 
    if args.resume:
        print(f"resume from checkpoint {args.resume}")
        args.resume = get_model_card(args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')
        state_dict = checkpoint['state_dict']
        args.start_epoch = checkpoint['epoch']+1
        args.iteration = checkpoint['iteration']
        best_val_loss = checkpoint['best_acc']
        try:
            model_without_dp.load_state_dict(state_dict)
        except:
            missing_keys, unexpected_keys = model_without_dp.load_state_dict(state_dict, strict=False)
            if len(missing_keys):
                print(f'[Missing keys]:{"="*12}\n{chr(10).join(missing_keys)}\n{"="*20}')
            if len(unexpected_keys):
                print(f'[Unexpected keys]:{"="*12}\n{chr(10).join(unexpected_keys)}\n{"="*20}')
            user_input = input('[WARNING] Non-Equal load for resuming training! continue? [y/n]')
            if user_input.lower() == 'n': sys.exit()
        optimizer.load_state_dict(checkpoint['optimizer'])

    elif args.pretrain:
        print(f"pretrain from checkpoint {args.pretrain}")
        args.pretrain = get_model_card(args.pretrain)
        checkpoint = torch.load(get_model_card(args.pretrain), map_location='cpu')
        state_dict = checkpoint['state_dict']
        if args.model in ['cotrain']:
            if '_cotrain_' in args.pretrain:
                pass
            else:
                tmp_dict = {f"target.{k}": v for k,v in state_dict.items()}
                tmp_dict.update({f"online.{k}": v for k,v in state_dict.items()})
                tmp_dict.update({k: v for k,v in state_dict.items() if 'lang_model.' in k})
                state_dict = tmp_dict
        try:
            model_without_dp.load_state_dict(state_dict)
        except:
            missing_keys, unexpected_keys = model_without_dp.load_state_dict(state_dict, strict=False)
            if len(missing_keys):
                print(f'[Missing keys]:{"="*12}\n{chr(10).join(missing_keys)}\n{"="*20}')
            if len(unexpected_keys):
                print(f'[Unexpected keys]:{"="*12}\n{chr(10).join(unexpected_keys)}\n{"="*20}')
        
        if args.model in ['cotrain']:
            model_without_dp._copy_param()
            print('[TwinExoGroundingTransformer] parameter copied from online stream to target stream')

    if not args.test:
        args.decay_steps = args.epochs * len(train_loader)
    args.warmup_iterations = 1000
    def lr_schedule_fn(iteration, iter_per_epoch, args):
        if iteration < args.warmup_iterations:
            lr_multiplier = iteration / (args.warmup_iterations)
        else:
            lr_multiplier = 0.5 * \
                (1. + math.cos(math.pi * (iteration - args.warmup_iterations) / (args.epochs*iter_per_epoch - args.warmup_iterations)))
        return lr_multiplier

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, functools.partial(lr_schedule_fn, iter_per_epoch=len(train_loader), args=args)
    )
    lr_scheduler.step(args.iteration)  # for resume mode
    grad_scaler = amp.GradScaler()

    # profiler, optional
    args.prof = None
    
    print('Main loop starts')
    for epoch in range(args.start_epoch, args.epochs):
        np.random.seed(epoch)
        random.seed(epoch)

        #Curriculum learning updates
        if args.model in ['joint'] and args.curriculum_train:
            if args.sorted_curr_train in ['phased']:
                train_loader.dataset.set_phase(get_phase(epoch=epoch, total_epochs=args.epochs, num_phases=4, final_phase_proportion=args.final_phase_prop))
            elif args.views == "all" and args.sorted_curr_train in ['sorted']:
                train_sampler.set_epoch(epoch)

        if args.distributed:
            dist.barrier()
        train_loss = train(train_loader, model, optimizer, lr_scheduler, grad_scaler, device, epoch, args)
        val_loss = evaluate(val_loader, model, device, epoch, args) 

        if rank == 0 and ((epoch % args.eval_freq == 0) or (epoch == args.epochs - 1)):
            is_best = val_loss < best_val_loss
            best_val_loss = min(val_loss, best_val_loss)
            state_dict = model_without_dp.state_dict()
            save_dict = {
                'epoch': epoch,
                'state_dict': state_dict,
                'best_acc': best_val_loss,
                'optimizer': optimizer.state_dict(),
                'iteration': args.iteration}
            save_checkpoint(save_dict, is_best, args.eval_freq, 
                filename=os.path.join(args.model_path, 'epoch%d.pth.tar' % epoch), 
                keep_all=(args.model in ['cotrain']),)
    if rank == 0:
        print('Training from ep %d to ep %d finished' % (args.start_epoch, args.epochs))
    dist.destroy_process_group()
    sys.exit(0)


def get_model_card(tag):
    """allow saving ckpt shortcuts in model_card_dict. """
    model_card_dict = {}
    if tag in model_card_dict:
        print(f'getting model tag {tag}: {model_card_dict[tag]}')
    return model_card_dict.get(tag, tag)


if __name__ == "__main__":
    args = parse_args()
    args.rank = int(os.environ['RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])
    main(args)

"""
train keysteps:
torchrun --nproc_per_node=8 main_egoexo4d_distributed.py --batch_size 16 --epochs 100 --num_workers 0 --use_keysteps

test keysteps:
# torchrun --nproc_per_node=8 main_egoexo4d_distributed.py --batch_size 16 --test <PATH_TO_PTH_FILE>.tar --num_workers 0 --use_keysteps

flags:

use audio features: --use_audio

generate visualizations: --visualize

train on keysteps: --use_keysteps

use ego distill loss: --use_distill_nce_loss

train on ego view only: --views ego

train on exo views only: --views exo

train on all views: --views all

train on multi-views: --views multi

multi-view with ego: --multi_view_egoexo

pair-wise distill (all): --use_pairwise_distill_nce_loss

pair-wise distill (unmasked views only): --pairwise_distill_mode unmasked

narration order shuffling (train augmentation): --randomize_narration_order

multi-view final phase fraction: --final_phase_prop <FRAC>

curriculum train: --curriculum_train

exos (for training): --exos best, random, worst, all(default)

curr learning start fraction: --start_frac <FRAC>

curr learning end epoch perc. of max epochs: --end_epoch_frac <FRAC>

whether to train with stitched-view best exo distillation: --stitched_best_exo_distill

resume training: --resume <PATH_TO_FILE>.tar
"""