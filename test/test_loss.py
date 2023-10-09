import unittest
import torch
from torch import nn
import numpy as np

from loss import SegmentationLoss

class LossTest(unittest.TestCase):

    def test_segmentation(self):
        pred1 = torch.tensor([
            [1, 0, 1],
            [1, 1, 1],
            [1, 0, 1]
        ])
        target1 = torch.tensor([
            [1, 0, 1],
            [1, 0, 1],
            [1, 0, 1]
        ])
        iou1 = 6 / 7

        pred2 = torch.tensor([
            [1, 0, 1],
            [0, 0, 0],
            [0, 0, 0]
        ])
        target2 = torch.tensor([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])
        iou2 = 0

        preds = torch.stack((pred1, pred2))
        targets = torch.stack((target1, target2))

        pred_batch = torch.stack((preds, preds))
        targets_batch = torch.stack((targets, targets))
        # Since we calculate the mean of the IOU loss for each sample, and both samples here are duplicates of each other,
        # mean_iou_loss will not change

        expected_loss = 1 - 6/7
        loss = SegmentationLoss()
        print(loss(pred_batch.to(torch.float32), targets_batch.to(torch.float32)))

        assert(expected_loss == loss(pred_batch.to(torch.float32), targets_batch.to(torch.float32)))
