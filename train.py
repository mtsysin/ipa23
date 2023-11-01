import argparse

import os
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data as data
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP 
from torch.distributed import init_process_group, destroy_process_group
# from torch.utils.tensorboard import SummaryWriter

from bdd100k import BDD100k, ANCHORS, BDD_100K_ROOT
#from utils import non_max_supression, mean_average_precission, intersection_over_union
from loss import MultiLoss, SegmentationLoss, DetectionLoss
from utils import Reduce_255

import matplotlib.pyplot as plt
import tqdm
import math
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('runs/prototype_lane')

torch.autograd.detect_anomaly(True)
torch.autograd.set_detect_anomaly(True, check_nan=True)

DEFAULT_BATCH = 16
BATCH_ACCUMULATION = 4
DEFAULT_EPOCHS = 99
TRAIN_SIZE = 16*4*20
VAL_SIZE = 16*4*2

LOSS_COUNT_TRAIN = 4
LOSS_COUNT_VAL = 1
ROOT = "."

USE_DDP = False             # doesn't work currently
USE_PARALLEL = True
LOAD_OLD_MODEL = False
WEIGHTS_INIT = False

OPTIMIZER = "adam"
MOMENTUM = 0.94
LEARNING_RATE = 1e-3

WEIGHT_DECAY_SWITCH_ON = False
if WEIGHT_DECAY_SWITCH_ON:
    WEIGHT_DECAY = 5e-4

INPUT_IMG_TRANSFORM = transforms.Compose([
        transforms.Resize((384, 640), interpolation=transforms.InterpolationMode.NEAREST),
        Reduce_255(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

SCHED_STEP = "epoch" # "batch", "epoch", "no"
SHUFFLE_OFF = False
SCHEDULER_PARAM = [150, 400, 1000, 2000]
SCHED_GAMMA = 0.2

DEVICE_IDS = [0, 2]

MODEL = None

torch.cuda.empty_cache()
device = torch.device(f'cuda:{DEVICE_IDS[0]}')
print(torch.cuda.is_available()) 
print(f"CUDA device: {torch.cuda.current_device()}")
print(f"CUDA device count: {torch.cuda.device_count()}")

def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda", help='devise -- cuda or cpu')
    parser.add_argument('--batch', type=int, default=DEFAULT_BATCH, help='batch size')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='number of training epochs')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers')
    parser.add_argument('--root', type=str, default=BDD_100K_ROOT, help='root directory for both image and labels')
    parser.add_argument('--cp', '-checkpoint', type=str, default='', help='path to checpoint of pretrained model')
    parser.add_argument('--sched_points', nargs='+', type=int, default=SCHEDULER_PARAM, help='sheduler milestones list')
    parser.add_argument('--sched_gamma', type=int, default=SCHED_GAMMA, help='gamma for learning rate scheduler')
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
    epochs = args.epochs

    if USE_DDP:
        pass

    #Load model
    if USE_DDP:
        model = MODEL()
        model = DDP(model, device_ids=DEVICE_IDS)
    elif USE_PARALLEL:
        model = MODEL()
        model= torch.nn.DataParallel(model, device_ids=DEVICE_IDS)
        if LOAD_OLD_MODEL:
            model.load_state_dict(torch.load(ROOT+'/out/model_100_24.pt'))
        model.to(device)
    else:
        model = MODEL().to(device)

    if WEIGHTS_INIT:
        model.apply(weights_init)

    #Set optimizer, loss function, and learning rate scheduler

    if WEIGHT_DECAY_SWITCH_ON:
        # hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in model.named_parameters():
            if v.requires_grad:
                # print(k)
                if '.bias' in k:
                    pg2.append(v)  # biases
                elif '.weight' in k and '.batchnorm' not in k:
                    pg1.append(v)  # apply weight decay
                else:
                    pg0.append(v)  # all else

        if OPTIMIZER == "adam":
            optimizer = torch.optim.Adam(
                pg0, 
                lr=LEARNING_RATE, 
                betas=(MOMENTUM, 0.999)
            )
        elif OPTIMIZER == "sgd":
            optimizer = torch.optim.SGD(
                pg0,
                lr=LEARNING_RATE,   
            )

        optimizer.add_param_group({'params': pg1, 'weight_decay': WEIGHT_DECAY})  # add pg1 with weight_decay
        optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
        print('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
        del pg0, pg1, pg2

    else:
        if OPTIMIZER == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=LEARNING_RATE, 
                betas=(MOMENTUM, 0.999)
            )
        elif OPTIMIZER == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=LEARNING_RATE,   
            )

    # Select scheduler
    if SCHED_STEP != "no":
        print("Scheduler is ON")
        # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.00001, max_lr=0.01, cycle_momentum=False, step_size_up=100)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.sched_points, gamma=args.sched_gamma)
        lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.9 + 0.1  # cosine
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)


    loss_fn = DetectionLoss()

    transform = INPUT_IMG_TRANSFORM

    #Load BDD100k Dataset
    dataset = BDD100k(root=BDD_100K_ROOT, train=True, transform=transform, anchors=ANCHORS)
    indices_train = [i for i in range (TRAIN_SIZE)]
    indices_val = [i for i in range (TRAIN_SIZE, TRAIN_SIZE + VAL_SIZE)]

    train_dataset = data.Subset(dataset, indices_train)
    val_dataset = data.Subset(dataset, indices_val)

    # val_dataset = BDD100k(root='/home/pumetu/Purdue/LaneDetection/BDD100k/', train=False, transform=transform, anchors=ANCHORS)

    train_loader = data.DataLoader(dataset=train_dataset, 
                                batch_size=args.batch,
                                num_workers=args.num_workers,
                                shuffle = False if USE_DDP or SHUFFLE_OFF else True, 
                                sampler= DistributedSampler(train_dataset) if USE_DDP else None
    )

    val_loader = data.DataLoader(dataset=val_dataset, 
                                batch_size=args.batch,
                                num_workers=args.num_workers,
                                shuffle = False if USE_DDP or SHUFFLE_OFF else True, 
                                sampler= DistributedSampler(train_dataset) if USE_DDP else None
    )


    file_log = tqdm.tqdm(total=0, position=2, bar_format='{desc}')  
    outer_tqdm = tqdm.tqdm(total=epochs, desc='Epochs', position=1)
    inner_tqdm = tqdm.tqdm(total=len(train_loader), desc='Batches', position=0)

    losses = []
    val_losses = []

    for epoch in tqdm.tqdm(range(epochs)):
        if USE_DDP:
            train_loader.sampler.set_epoch(epoch)
        running_loss = 0.0
        #--------------------------------------------------------------------------------------
        #Train
        model.train()

        inner_tqdm.refresh() 
        inner_tqdm.reset()

        batches_accumulated = 0
        for i, (imgs, det, seg) in enumerate(train_loader):
            
            batches_accumulated += 1
            
            imgs, seg = imgs.to(device), seg.to(device)
            det = [d.to(device) for d in det]

            det_pred, _ = model(imgs)

            ################# Comment out -- debug
            # max_param_value = float('-inf')
            # min_param_value = float('inf')
            # for param in model.parameters():
            #     max_param_value = max(max_param_value, torch.max(param).item())
            #     min_param_value = min(min_param_value, torch.min(param).item())
            # print("parameter value range:", min_param_value, max_param_value)
            ##############################


            
            loss, loss_info = loss_fn(det_pred, det)
            loss.backward()

            ################# Comment out
            # max_grad_value = float('-inf')
            # min_grad_value = float('inf')
            # for param in model.parameters():
            #     if param.grad is not None:
            #         max_grad_value = max(max_grad_value, torch.max(param.grad).item())
            #         min_grad_value = min(min_grad_value, torch.min(param.grad).item())
            # print("gradient value range:", min_grad_value, max_grad_value)
            ##############################

            if batches_accumulated == BATCH_ACCUMULATION:
                optimizer.step()
                ############### comment out
                # nan = False
                # for param in model.parameters():
                #     if torch.isnan(param).any():
                #         nan = True
                # if nan:
                #     print("We have nan!!!")
                #     print(torch.max(imgs).item(), torch.min(imgs).item(), torch.isnan(imgs).any())
                #     for d in det:
                #         print(torch.max(d).item(), torch.min(d).item(), torch.isnan(d).any())
                # else:
                #     print("We are good")
                ###############
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


if __name__ == '__main__':
    print("run")
    losses, val_losses = main()
    plt.plot(losses, label = "Loss")
    plt.ylabel('Loss')
    plt.xlabel(f'Processed batches * {LOSS_COUNT_TRAIN}')
    plt.legend()
    plt.savefig("./out/loss_trace.png")

    plt.clf()

    plt.plot(val_losses, label = "VAL Loss")
    plt.ylabel('Loss')
    plt.xlabel(f'Processed epochs* {LOSS_COUNT_VAL}')
    plt.legend()
    plt.savefig("./out/val_loss_trace.png")
