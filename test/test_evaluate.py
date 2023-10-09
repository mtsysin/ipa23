import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from model.YoloMulti import YoloMulti
import unittest
from postprocess import *
from bdd100k import *
from train import ROOT
from evaluate import DetectionMetric


device = torch.device('cuda')
print(torch.cuda.is_available()) 
print(f"CUDA device: {torch.cuda.current_device()}")
print(f"CUDA device count: {torch.cuda.device_count()}")

class TestEvaluate(unittest.TestCase):
    def test_ciou(self):
        """Compare CIOU metric to pytorch implementation"""
        metric = DetectionMetric()
        # Generate random tensor:
        torch.manual_seed(0)
        pred = torch.rand(10, 4)
        target = torch.rand(10, 4)

        pred[..., 0:1], pred[..., 2:3] = torch.min(pred[..., 0:1], pred[..., 2:3]), torch.max(pred[..., 0:1], pred[..., 2:3])
        pred[..., 1:2], pred[..., 3:4] = torch.min(pred[..., 1:2], pred[..., 3:4]), torch.max(pred[..., 1:2], pred[..., 3:4])

        target[..., 0:1], target[..., 2:3] = torch.min(target[..., 0:1], target[..., 2:3]), torch.max(target[..., 0:1], target[..., 2:3])
        target[..., 1:2], target[..., 3:4] = torch.min(target[..., 1:2], target[..., 3:4]), torch.max(target[..., 1:2], target[..., 3:4])

        print(pred)
        print(target)

        res1 = 1 - metric.box_iou(pred, target, CIoU=True).mean()
        res2 =  torchvision.ops.complete_box_iou_loss(pred, target, "mean")
        res_zero = 1 - metric.box_iou(pred, pred, CIoU=True).mean()
        print("distance", torchvision.ops.distance_box_iou_loss(pred, target, "mean"))


        print("Results should be equal:", res1, res2)
        print("Should be zero: ", res_zero)
        assert res1.item()//0.00001 == res2.item()//0.00001

        