"""Big infrastructure_level script to accomodate parameter parsing a preparation for training/testing"""

import argparse
import os
import random
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data as data
import torch.multiprocessing as mp
# from torch.utils.tensorboard import SummaryWriter

from bdd100k_lightweight import BDD100k_DETR
#from utils import non_max_supression, mean_average_precission, intersection_over_union
from loss import MultiLoss, SegmentationLoss, DetectionLoss
from utils import Reduce_255

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import matplotlib.pyplot as plt
import tqdm
import math
import ddp
from model import get_model

def get_args_parser():


    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--debug', action='store_true', help='Slow, debug mode with some anomaly detections')
    parser.add_argument('--init_weight', action='store_true', help='Initialize weights with custom initializer')

    parser.add_argument('--root', type=str, default='./bdd100k', help='root directory for both image and labels')


    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parallelization_options = ['none', 'ddp', 'dp']
    parser.add_argument('--parallelization', type=str, choices=parallelization_options, default = 'ddp',
                        help='Choose parallelization: none, dp, ddp')
    parser.add_argument('--gpus', type=int, nargs="+", default=[0, 1, 2, 3], 
                        help='number of workers for dataloader') # this will be mostly needed for the pure data parallel.


    # Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--lr_backbone', default=1e-5, type=float)


    # Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')


    # Loss coefficients
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # Training utils
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('--cp', '-checkpoint', type=str, default='', help='path to checpoint of pretrained model')
    optimizer_options = ['adam', 'adamw', 'sgd']
    parser.add_argument('--opt', '-optimizer', type=str, choices=optimizer_options, default = 'adamw',
                    help='optimizer used for training')
    scheduler_options = ['no', 'cyclic', 'step', 'multistep', 'cosine']
    parser.add_argument('--sched', '-scheduler', type=str, choices=scheduler_options, default = 'step',
                    help='scheduler used for training')
    parser.add_argument('--lr_step', default=200, type=int)
    parser.add_argument('--sched_points', nargs='+', type=int, default=[150, 400, 1000, 2000], help='sheduler milestones list')
    parser.add_argument('--sched_gamma', type=int, default=0.2, help='gamma for learning rate scheduler')
    parser.add_argument('-seed', type=int, default=None, help='seed for reproducing behavior')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                    help='gradient clipping max norm')

    # Augmentation control:
    parser.add_argument('--bbox_size_threshold', default=None, type=float) # Ignore small bounding boxes
    parser.add_argument('--mosaic_prob', default=0.1, type=float)
    parser.add_argument('--augment_prob', default=0.5, type=float)
    parser.add_argument('--mixup_after_mosaic_prob', default=0, type=float)
    parser.add_argument('--mixup_prob', default=0.3, type=float)

    return parser


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cp', '-checkpoint', type=str, default='', help='path to checpoint of pretrained model')
    parser.add_argument('--save', type=bool, default=True, help='save model flag')

    return parser.parse_args()

# Initialize weights:
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 1)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.5)
        torch.nn.init.constant_(m.bias.data, 0)

def main():
    args = parse_arg()
    print("Parsed Args:")
    print(args)

    if args.debug:
        torch.autograd.detect_anomaly(True)
        torch.autograd.set_detect_anomaly(True, check_nan=True)

    if args.parallelization == 'ddp':
        ddp.init_distributed_mode(args)

    if "cuda" in args.device:
        torch.cuda.empty_cache()
    device = torch.device(args.device)
    print("CUDA availablilty:", torch.cuda.is_available()) 
    print(f"CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")

    if args.seed:
        seed = args.seed + ddp.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    model, loss_fn = get_model(args) ## get model somehow
    model.to(device)

    #Load model
    bare_model = model
    if args.parallelization == 'ddp':
        model = DDP(model, device_ids=[args.gpus]) # will be taken from the process ID
        bare_model = model.module
    elif args.parallelization == 'dp':
        model = torch.nn.DataParallel(model, device_ids=args.gpus)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if args.init_weights:
        model.apply(weights_init)

    #Set optimizer, loss function, and learning rate scheduler
    #We acn use weight decays for trining the backbone

    pg_default, pg_backbone_weight, pg_backbone_bias, pg_bias =  [], [], [], [], []  # optimizer parameter groups
    for name, param in bare_model.named_parameters():
        if param.requires_grad:
            if 'backbone' in name:
                if 'bias' in name:
                    pg_backbone_bias.append(param)
                elif 'weight' in name:
                    pg_backbone_weight.append(param)
            elif '.bias' in name:
                pg_bias.append(param)  # apply weight decay
            else:
                pg_default.append(param) # all else

    param_dicts = [
        {"params": pg_backbone_bias, "lr": args.lr_backbone, "weight_deacy": 0},
        {"params": pg_backbone_weight, "lr": args.lr_backbone, "weight_deacy": args.backbone_weight_decay},
        {"params": pg_bias, "weight_deacy": 0},
        {"params": pg_default}
    ]

    optimizer_map = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
        "adamw": torch.optim.AdamW
    }

    optimizer = optimizer_map[args.optimizer](param_dicts,  lr=args.lr,
                                  weight_decay=args.weight_decay)
    
    print('Optimizer groups: %g .pg_backbone_bias, %g pg_backbone_weight, %g pg_bias, %g other' % 
          (len(pg_backbone_bias), len(pg_backbone_weight), len(pg_bias), len(pg_default)))
    del pg_default, pg_backbone_weight, pg_backbone_bias, pg_bias

    if args.scheduler != "no":
        print(f"Scheduler is on: {args.scheduler}") 
    if args.scheduler == "cyclic":
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.00001, max_lr=0.01, cycle_momentum=False, step_size_up=args.lr_step_size)
    if args.scheduler == "multistep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.sched_points, gamma=args.sched_gamma)
    if args.scheduler == "cosine":
        lf = lambda x: (((1 + math.cos(x * math.pi / args.epochs)) / 2) ** 1.0) * 0.9 + 0.1  # cosine
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    if args.scheduler == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size)
    else:
        raise NotImplementedError(f"The scheduler is not supported: {args.scheduler}")

    train_dataset = BDD100k_DETR(image_set='train', args=args)
    val_dataset = BDD100k_DETR(image_set='val', args=args)

    # In case we don't want to train on the whole dataset, we limit the
    if args.train_size:
        indices_train = np.random.choice(len(train_dataset), args.train_size, replace = False)
        train_dataset = data.Subset(train_dataset, indices_train)
    if args.train_size:
        indices_val = np.random.choice(len(val_dataset), args.val_size, replace = False)
        val_dataset = data.Subset(val_dataset, indices_val)

    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)

    train_loader = data.DataLoader(dataset=train_dataset, 
                                batch_size=args.batch_size,
                                drop_last = True,
                                num_workers=args.num_workers,
                                sampler = train_sampler,
                                collate_fn = None
    )

    val_loader = data.DataLoader(dataset=val_dataset, 
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                sampler= val_sampler,
                                collate_fn = None                           # TODO: make collator

    )

    output_dir = Path(args.output_dir)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')

        del checkpoint['model']["class_embed.weight"]
        del checkpoint['model']["class_embed.bias"]
        bare_model.load_state_dict(checkpoint['model'], strict=False)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        test_stats = evaluate(model, loss_fn, val_loader, device, args.output_dir)
        if args.output_dir and ddp.is_main_process():
            torch.save(test_stats, output_dir / "eval.pth")
    
    else:

        print("Running trainig loop")

        file_log = tqdm.tqdm(total=0, position=2, bar_format='{desc}')  
        outer_tqdm = tqdm.tqdm(total=args.epochs, desc='Epochs', position=1)
        inner_tqdm = tqdm.tqdm(total=len(train_loader), desc='Batches', position=0)

        losses = []
        val_losses = []

        for epoch in tqdm.tqdm(range(args.start_epoch, args.epochs)):
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            running_loss = 0.0
            #--------------------------------------------------------------------------------------
            #Train
            train(model, loss_fn, train_loader, optimizer, device, epoch, args.clip_max_norm)

            inner_tqdm.refresh() 
            inner_tqdm.reset()

            batches_accumulated = 0
            for i, (imgs, det, seg) in enumerate(train_loader):
                
                batches_accumulated += 1
                
                imgs, seg = imgs.to(device), seg.to(device)
                det = [d.to(device) for d in det]

                det_pred, _ = model(imgs)

                loss, loss_info = loss_fn(det_pred, det)
                loss.backward()

            
                if batches_accumulated == BATCH_ACCUMULATION:
                    optimizer.step()
                    optimizer.zero_grad()
                    batches_accumulated = 0
                    running_loss += loss.item()

                # writer.add_scalar("Loss/train", running_loss, epoch)
                # print(f"[epoch: {epoch + 1}, batch: {i + 1}] loss: {loss.item()}, components: {loss_info}")

                if ((i+1) % BATCH_ACCUMULATION) == 0 and ((i+1) // BATCH_ACCUMULATION) % LOSS_COUNT_TRAIN == 0:
                    file_log.set_description_str(
                        f"[epoch: {epoch + 1}, batch: {i + 1}] loss: {running_loss / LOSS_COUNT_TRAIN}, components: {loss_info}"
                    )
                    # print(f"[epoch: {epoch + 1}, batch: {i + 1}] loss: {running_loss / LOSS_COUNT_TRAIN}, components: {loss_info}")
                    losses.append(running_loss / LOSS_COUNT_TRAIN)
                    running_loss = 0.0

                inner_tqdm.update(1)

                if SCHED_STEP == "batch":
                    scheduler.step()

            # Evaluation code
            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for i, (imgs, det, seg) in enumerate(val_loader):
                    
                    imgs, seg = imgs.to(device), seg.to(device)         # Select correct device for training
                    det = [d.to(device) for d in det]

                    det_pred, _ = model(imgs)

                    val_loss, loss_info = loss_fn(det_pred, det)

                    running_val_loss += val_loss.item()

                    if ((i+1)) % LOSS_COUNT_VAL == 0:
                        val_losses.append(running_val_loss / LOSS_COUNT_VAL)
                        file_log.set_description_str(
                            "[VALIIDATION, epoch: %d, batch: %5d] loss: %.3f" % (epoch + 1, i + 1, running_val_loss/LOSS_COUNT_VAL)
                        )
                        running_val_loss = 0.0


        outer_tqdm.update(1)
        # writer.flush()

        if SCHED_STEP == "epoch":
            scheduler.step()

    if args.save:
        print("Saving the model")
        torch.save(model.state_dict(), ROOT+f"/model_{epochs}_{args.batch}.pt")
        if USE_PARALLEL:
            torch.save(model.module.state_dict(), ROOT+f"/model_{epochs}_{args.batch}_parallel.pt")

    return losses, val_losses
    
    # untransform = transforms.Compose([
    #     transforms.Resize((720, 1280), interpolation=transforms.InterpolationMode.NEAREST),
    # ])
    # imgs = untransform(imgs)
    # seg = untransform(seg)
    # pseg = untransform(pseg)
    # torch.save(imgs, 'imgs.pt')
    # torch.save(seg, 'seg.pt')
    # torch.save(pseg, 'pseg.pt')


def train(
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        train_loader, 
        optimizer: torch.optim.Optimizer,
        device: torch.device, 
        epoch: int, 
        max_norm: float = 0
    ):
    model.train()
    loss_fn.train()
    file_log = tqdm.tqdm(total=0, position=2, bar_format='{desc}')  


    for i, (imgs, targets) in tqdm.tqdm(enumerate(train_loader)):
        imgs = imgs.to(device)
        targets = tuple(t.to(device) for t in targets)
        outputs = model(imgs)
        loss_dict, weight_dict = loss_fn(outputs, targets)
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        optimizer.zero_grad()
        loss.backward()

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()

        # reduce losses over all GPUs for logging purposes (stolen from DETR)
        loss_dict_reduced = ddp.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        if (i+1) % 10 == 0:
            file_log.set_description_str(
                f"[epoch: {epoch + 1}, batch: {i + 1}] loss: {loss_value}, components: {loss_dict_reduced_unscaled}"
            )


# def eval(
#         model: torch.nn.Module,
#         loss_fn: torch.nn.Module,
#         val_loader, 
#         device: torch.device, 
#         epoch: int, 
#         max_norm: float = 0
#     ):

#     model.eval()
#     loss_fn.eval()

#     for i, (imgs, targets) in tqdm.tqdm(enumerate(val_loader)):
#         imgs = imgs.to(device)
#         targets = tuple(t.to(device) for t in targets)

#         outputs = model(imgs)
#         loss_dict, weight_dict = loss_fn(outputs, targets)

#         # reduce losses over all GPUs for logging purposes (stolen from DETR)
#         loss_dict_reduced = ddp.reduce_dict(loss_dict)
#         loss_dict_reduced_unscaled = {f'{k}_unscaled': v
#                                       for k, v in loss_dict_reduced.items()}
#         loss_dict_reduced_scaled = {k: v * weight_dict[k]
#                                     for k, v in loss_dict_reduced.items() if k in weight_dict}
#         losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
#         loss_value = losses_reduced_scaled.item()

#         if (i+1) % 10 == 0:
#             file_log.set_description_str(
#                 f"[epoch: {epoch + 1}, batch: {i + 1}] loss: {loss_value}, components: {loss_dict_reduced_unscaled}"
#             )
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir and not os.path.exists(args.output_dir):
        os.mkdirs(args.output_dir)
    main(args)