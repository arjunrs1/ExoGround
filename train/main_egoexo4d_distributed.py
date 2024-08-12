import os
import sys
import torch
from torch.utils import data 
from tensorboardX import SummaryWriter
import numpy as np 
import random 
from tqdm import tqdm
import time
import math
import functools
import torch.cuda.amp as amp 
from config_egoexo4d import parse_args, set_path
from loss_egoexo4d import get_loss, get_mask_from_time, get_text_pos, visualize
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

sys.path.append('../data/')
from data.loader_egoexo4d import EgoExo4DDataLoader
sys.path.append('../model/')
from model.exo_ground_model import ExoGroundingTransformer, TwinExoGroundingTransformer
sys.path.append('../')
import utils.tensorboard_utils as TB
from utils.data_utils import DataLoaderBG
from utils.train_utils import clip_gradients
from utils.utils import AverageMeter, save_checkpoint, neq_load_customized, \
calc_topk_accuracy, ProgressMeter, neq_load_customized, save_runtime_checkpoint

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

        B, T, _ = video_seq.shape

        # forward pass
        with amp.autocast():
            logits = model(video_seq, text_embed, 
                    video_padding_mask=video_padding_mask.bool(), 
                    lang_padding_mask=text_padding_mask.bool(),
                    audio_embed=audio_seq,
                    audio_padding_mask=audio_padding_mask,
                    egocentric_video_embed=ego_seq
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

            loss_dict = get_loss(input_data=input_data,
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

            if args.model in ['cotrain']:
                model._momentum_update() #TODO: Need to DDPify this if using cotrain

        # log stats
        if rank == 0 and args.iteration % 5 == 0:
            for k, v in loss_dict.items():
                args.train_plotter.add_data(f'train/{k}', v.item(), args.iteration)
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
    model.eval()
    batch_time = AverageMeter('Time', ':.2f')
    losses = AverageMeter('Loss', ':.4f')
    if args.test:
        iou_threshold_meters = {}
        for theta in args.iou_thresholds:
            iou_threshold_meters[f"{theta}"] = AverageMeter(f'IoU>={theta}', ':.4f')
    progress = ProgressMeter(
        len(loader), [batch_time, losses],
        prefix="Test: ")

    end = time.time()
    vis_this_epoch = False
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

        # Forward pass
        logits = model(video_seq, text_embed,
                       video_padding_mask=video_padding_mask.bool(),
                       lang_padding_mask=text_padding_mask.bool(),
                       audio_embed=audio_seq,
                       audio_padding_mask=audio_padding_mask,
                       egocentric_video_embed=ego_seq)

        loss_dict = get_loss(input_data=input_data,
                                 text_padding_mask=text_padding_mask,
                                 logits=logits, 
                                 args=args)

        loss = loss_dict['loss']
        losses.update(loss.item(), video_seq.size(0))
        if args.test:
            for theta in args.iou_thresholds:
                iou_threshold_meters[f"{theta}"].update(loss_dict[f'IoU>={theta}'], int(text_padding_mask.sum()))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % args.print_freq == 0:
            progress.display(idx)

        # Visualization call
        if args.visualize and idx % args.vis_freq == 0 and not vis_this_epoch:
            visualize(input_data, logits, args, epoch)
            vis_this_epoch = True #only vis one batch per epoch
            #TODO: Put this on tensorboard, so you can visualize during training

    print(f' * Loss {losses.avg:.4f}')

    for k, v in loss_dict.items():
        if "IoU>=" not in k:
            args.train_plotter.add_data(f'val/{k}', v.item(), epoch)
    
    if args.test:
        return losses.avg, iou_threshold_meters
    else:
        return losses.avg

def setup(args):
    # Initialize distributed environment
    args.distributed = int(os.environ.get('SLURM_JOB_NUM_NODES', "1")) > 1
    if args.distributed:
        torch.distributed.init_process_group(backend='nccl')
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()
        torch.cuda.set_device(args.rank)
        device = torch.device(f'cuda:{args.rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # CUDA setting
    if torch.cuda.is_available():
        num_gpu = torch.cuda.device_count()
        args.num_gpu = num_gpu
        if args.rank == 0:
            print('=> Effective BatchSize = %d' % args.batch_size) #TODO: Check whether this is right, or should be mult'd by num_gpu
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
        use_center_duration=args.use_center_duration)
    val_dataset = D(
        split="val",
        duration=args.seq_len,
        hop_length=args.seq_hop,
        use_audio=args.use_audio,
        use_keysteps=args.use_keysteps,
        views=args.views,
        use_distill_nce_loss=args.use_distill_nce_loss,
        use_center_duration=args.use_center_duration)

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
        collate_fn=val_dataset.collate_fn, pin_memory=True, drop_last=False,
        shuffle=(val_sampler is None), sampler=val_sampler, 
    )
    return train_dataset, val_dataset, train_loader, val_loader

def get_test_dataset(args):
    D = EgoExo4DDataLoader
    test_dataset = D(
        split="test",
        duration=args.seq_len,
        hop_length=args.seq_hop,
        use_audio=args.use_audio,
        use_keysteps=args.use_keysteps,
        views=args.views,
        use_distill_nce_loss=args.use_distill_nce_loss,
        use_center_duration=args.use_center_duration)

    test_sampler = DistributedSampler(test_dataset, num_replicas=args.world_size, rank=args.rank)

    print("Loading test dataset...")
    test_loader = DataLoaderBG(test_dataset,
        batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=test_dataset.collate_fn, pin_memory=True, drop_last=True,
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
    if args.model == 'init':
        pass #TODO: This does nothing right now...eventually add functionality for cotraining/student-teacher training

    if not args.test:
        _, _, train_loader, val_loader = get_dataset(args)

    if args.test:
        #ensure no overlapping segments for full video inference
        args.seq_hop = args.seq_len 
        assert args.seq_hop == args.seq_len
         
        test_loader = get_test_dataset(args)

    assert args.views in ["exo", "ego", "all"] # possible view settings for train/test

    ### Model ###
    if args.model in ['init']:
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
                        use_distill_nce_loss=args.use_distill_nce_loss
        )
    elif args.model in ['cotrain']:
        #deal with this later...it all needs to change
        model = TwinExoGroundingTransformer(
                        m=args.momentum_m,
                        num_encoder_layers=args.num_encoder_layers,
                        num_decoder_layers=args.num_decoder_layers,
                        language_model=args.language_model,
                        sim=args.sim,
                        pos_enc=args.pos_enc,
                        use_text_pos_enc=args.use_text_pos_enc,
                        use_alignability_head=args.use_alignability_head,
                        random_pos_start=0,
        )

    model.to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)  # Wrap model with DDP
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
        _, iou_metric_meters = evaluate(test_loader, model, device, epoch, args)
        if rank == 0:
            mean_iou = []
            for k, v in iou_metric_meters.items():
                print(f"IoU>={k}: {v.avg:.4f}")
                mean_iou.append(v.avg)
            print(f"Mean IoU: {np.array(mean_iou).mean():.4f}")

        dist.barrier()
        sys.exit(0)

    ### restart ###
    best_acc = 1e5 
    if args.resume:
        print(f"resume from checkpoint {args.resume}")
        args.resume = get_model_card(args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')
        state_dict = checkpoint['state_dict']
        args.start_epoch = checkpoint['epoch']+1
        args.iteration = checkpoint['iteration']
        best_acc = checkpoint['best_acc']
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
            # user_input = input('[WARNING] Non-Equal load for resuming training! continue? [y/n]')
            # if user_input.lower() == 'n': sys.exit()
        
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
        train_loss = train(train_loader, model, optimizer, lr_scheduler, grad_scaler, device, epoch, args)
        val_loss = evaluate(val_loader, model, device, epoch, args) 

        if rank == 0 and ((epoch % args.eval_freq == 0) or (epoch == args.epochs - 1)):
            is_best = val_loss < best_acc  #use val loss to determine is_best
            best_acc = min(val_loss, best_acc)
            state_dict = model_without_dp.state_dict()
            save_dict = {
                'epoch': epoch,
                'state_dict': state_dict,
                'best_acc': best_acc,
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
train (no audio):
torchrun --num_proc_per_node=8 main_egoexo4d.py --batch_size 16 --epochs 100 --num_workers 0 --no_audio

test:
torchrun --num_proc_per_node=8 main_egoexo4d.py --batch_size 16 --test <PATH_TO_PTH_FILE>.tar --seq_hop 64 --num_workers 0

flags:

no audio features: --no_audio

generate visualizations: --visualize

train on keysteps: --use_keysteps

use ego distill loss: --use_distill_nce_loss

train on ego view only: --views ego

train on exo views only: --views exo

train on all views: --views all

resume training: --resume <PATH_TO_FILE>.tar


cotraining:

python main.py --model cotrain --dataset htm-370k --batch_size 128 --use_text_pos_enc 0 --epochs 20 --pretrain {} --loss_threshold 0.5
"""