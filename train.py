import argparse

import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.append('../yolov4')

from model.model import YoloMulti
from bdd100k import BDD100k
#from utils import non_max_supression, mean_average_precission, intersection_over_union
from loss import MultiLoss, SegmentationLoss, DetectionLoss
from utils import SegmentationMetric

import matplotlib.pyplot as plt
import tqdm

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/prototype_lane');

device = torch.cuda.set_device(1)
print(f"CUDA device: {torch.cuda.current_device()}")
print(f"CUDA device count: {torch.cuda.device_count()}")


ANCHORS = [[(12,16),(19,36),(40,28)], [(36,75),(76,55),(72,146)], [(142,110),(192,243),(459,401)]]

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda", help='cuda or cpu')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--epoch', type=int, default=100, help='epochs')
    parser.add_argument('--num_workers', type=int, default=12, help='number of workers')
    # Fix root below
    parser.add_argument('--root', type=str, default='/data/stevenwh/bdd100k/', help='root directory for both image and labels')
    parser.add_argument('--cp', '-checkpoint', type=str, default='', help='path to checpoint of pretrained model')
    return parser.parse_args()

def main():
    args = parse_arg()

    #Load model
    model = YoloMulti().to(device)
    metric = SegmentationMetric()

    #Set optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # Can try 6e-4
    
    #loss_fn = MultiLoss()
    loss_fn = SegmentationLoss()

    transform = transforms.Compose([
        transforms.Resize((384, 640), interpolation=transforms.InterpolationMode.NEAREST),
    ])
    #Load BDD100k Dataset
    train_dataset = BDD100k(root='/data/stevenwh/bdd100k/', train=True, transform=transform, anchors=ANCHORS)
    val_dataset = BDD100k(root='/data/stevenwh/bdd100k/', train=False, transform=transform, anchors=ANCHORS)

    train_loader = data.DataLoader(dataset=train_dataset, 
                                batch_size=args.batch,
                                num_workers=args.num_workers,
                                shuffle=True)
    val_loader = data.DataLoader(dataset=val_dataset, 
                                batch_size=args.batch,
                                num_workers=args.num_workers,
                                shuffle=False)
                  
    #imgs, det, seg = next(iter(train_loader)) # First batch
    model.train()
    groundtruths = [0] * (args.epoch*10)
    counts = 0

    for epoch in tqdm.tqdm(range(args.epoch)):
        #--------------------------------------------------------------------------------------
        #Train
        
        for _ in range(4):
            imgs, _, seg = next(iter(train_loader))
            imgs, seg = imgs.to(device), seg.to(device) 
            groundtruths[counts] = (imgs, seg)
            counts += 1
             
            model.train()
            running_loss = 0

            _, pseg = model(imgs)
            
            loss = loss_fn(pseg, seg.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            writer.add_scalar("Loss/train", running_loss, epoch)

            writer.flush()
    
    # Inference on validation for evaluation
    mean_iou = [0] * 12
    iou_counts = [0] * 12
    for i in range(len(groundtruths)):
        imgs, seg = groundtruths[i]
        imgs, seg = imgs.to(device), seg.to(device)  

        _, pseg = model(imgs)
        
        iou = metric.mean_iou(seg, pseg, args.batch)
        iou[iou == np.nan] = 0
        mean_iou += iou
        iou[iou != 0] = 1
        iou_counts += iou
        
    print(np.divide(mean_iou, iou_counts))
    
    
    '''
    torch.save(model.state_dict(), 'out/model.pt')
    torch.save(model, 'out/model.pth')
    torch.save(imgs, 'out/imgs.pt')
    torch.save(seg, 'out/seg.pt')
    torch.save(pseg, 'out/pseg.pt')
    '''


if __name__ == '__main__':
    main()