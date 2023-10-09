import torch
import torchvision.transforms as transforms
import torch.utils.data as data

#from model import YoloMulti
from data_loader import BDD100k

import matplotlib.pyplot as plt

def test_dataloader(root):

    transform = transforms.Compose([transforms.Resize((448,448))])
    #train_dataset = BDD100k(root='/home/pumetu/Purdue/LaneDetection/BDD100k/', train=True, transform=transform)
    # train_loader = data.DataLoader(dataset=train_dataset, 
    #                             batch_size=4,
    #                             num_workers=4,
    #                             shuffle=True)

    val_dataset = BDD100k(root='/home/pumetu/Purdue/LaneDetection/BDD100k/', train=False, transform=transform)
    val_loader = data.DataLoader(dataset=val_dataset, 
                                batch_size=4,
                                num_workers=4,
                                shuffle=False)

    imgs, labels, lane = next(iter(val_loader))    

if __name__ == '__main__':
    test_dataloader(root='/home/pumetu/Purdue/LaneDetection/BDD100k/')