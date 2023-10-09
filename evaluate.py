import torch

import matplotlib.pyplot as plt
import numpy as np
import math

class SegmentationMetric:
    IGNORE = -1

    def __init__(self):
        self.epsilon = 1e-6
    
    def iou(self, pred, target):
        """
        Author: Daniyaal Rasheed
        Intersectio Over Union for lane segmentation
        Args:
            pred: Binary 3D tensor, dimensions: [n_types, img_h, img_w]
            target: Matrix of integers 0...(n_lane_types - 1), dimensions: [img_h, img_w]
        Returns 
            iou: 1D tensor of iou for each lane type
        """
        # Expand target matrix into a tensor of the same size as pred
        crosswalk = (target == 0).to(torch.float32)
        double_other = (target == 1).to(torch.float32)
        double_white = (target == 2).to(torch.float32)
        double_yellow = (target == 3).to(torch.float32)
        road_curb = (target == 4).to(torch.float32)
        single_other = (target == 5).to(torch.float32)
        single_white = (target == 6).to(torch.float32)
        single_yellow = (target == 7).to(torch.float32)
        target = torch.stack([single_yellow, single_white, single_other, road_curb,double_yellow, double_white, double_other, crosswalk], dim=0)

        # Calculate intersection over union
        intersection = torch.logical_and(pred, target).to(torch.float32).sum(dim=(1,2))
        union = torch.logical_or(pred, target).to(torch.float32).sum(dim=(1,2))
        iou = intersection / (union + self.epsilon)

        # If a lane type is not present in the image, IOU will be 0.
        # It should be ignored when we compute the average IOU across all tested samples
        iou[union == 0] = SegmentationMetric.IGNORE

        return iou

#Evaluation
class DetectionMetric:
    def box_iou(self, box1, box2, xyxy=True, CIoU=False):
        """
        Author: William Stevens
                Pume Tuchinda
        Intersection Over Union
        Args:
            box1 (tensor): bounding box 1
            box2 (tensor): bounding box 2
            xyxy (bool): is format of boudning box in x1 y1 x2 y2
            CIoU (bool): if true calculate CIoU else IoU
        Returns:
            
        """
        # print("pred ", box1)
        # print("target ", box2)

        EPS = 1e-6

        if xyxy:
            box1_x1 = box1[..., 0:1]
            box1_y1 = box1[..., 1:2]
            box1_x2 = box1[..., 2:3]
            box1_y2 = box1[..., 3:4]
            box2_x1 = box2[..., 0:1]
            box2_y1 = box2[..., 1:2]
            box2_x2 = box2[..., 2:3]
            box2_y2 = box2[..., 3:4]
        else:
            box1_x1 = box1[..., 0:1] - box1[..., 2:3] / 2
            box1_y1 = box1[..., 1:2] - box1[..., 3:4] / 2
            box1_x2 = box1[..., 0:1] + box1[..., 2:3] / 2
            box1_y2 = box1[..., 1:2] + box1[..., 3:4] / 2
            box2_x1 = box2[..., 0:1] - box2[..., 2:3] / 2
            box2_y1 = box2[..., 1:2] - box2[..., 3:4] / 2
            box2_x2 = box2[..., 0:1] + box2[..., 2:3] / 2
            box2_y2 = box2[..., 1:2] + box2[..., 3:4] / 2
            
        x1 = torch.max(box1_x1, box2_x1)
        y1 = torch.max(box1_y1, box2_y1)
        x2 = torch.min(box1_x2, box2_x2)
        y2 = torch.min(box1_y2, box2_y2)

        intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
        box1_width, box1_height = box1_x2 - box1_x1, box1_y2 - box1_y1
        box2_width, box2_height = box2_x2 - box2_x1, box2_y2 - box2_y1
        union = box1_width * box1_height + box2_width * box2_height - intersection
        iou = intersection / (union + EPS)

        if CIoU:
            '''
            Complete-IOU Loss Implementation
            - Inspired by the official paper on Distance-IOU Loss (https://arxiv.org/pdf/1911.08287.pdf)
            - Combines multiple factors for bounding box regression: IOU loss, distance loss, and aspect ratio loss.
            - This results in much faster convergence than traditional IOU and generalized-IOU loss functions.
            Args:
                - preds: prediction tensor containing confidence scores for each class.
                - target: ground truth containing correct class labels.
            '''
            convex_width = torch.max(box1_x2, box2_x2) - torch.min(box1_x1, box2_x1)
            convex_height = torch.max(box1_y2, box2_y2) - torch.min(box1_y1, box2_y1)
            convex_diag_sq = convex_width**2 + convex_height**2
            center_dist_sq = (box2_x1 + box2_x2 - box1_x1 - box1_x2)**2 + (box2_y1 + box2_y2 - box1_y1 - box1_y2)**2
            dist_penalty = center_dist_sq / (convex_diag_sq + EPS) / 4 

            v = (4 / (torch.pi**2)) * torch.pow(torch.atan(box2_width / (box2_height + EPS)) - torch.atan(box1_width / (box1_height + EPS)), 2)
            with torch.no_grad():
                alpha = v / ((1 + EPS) - iou + v)
            aspect_ratio_penalty = alpha * v
            
            return iou - dist_penalty - aspect_ratio_penalty
        
        return iou
