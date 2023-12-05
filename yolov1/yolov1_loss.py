"""
Author: Vishal Urs
"""

import torch
import torch.nn as nn
from yolov1_IOU import iou as intersection_over_union


class YoloLoss(nn.Module):
    """
    YOLOv1 loss function and its calculations
    """

    def __init__(self, split_grid=7, num_boundbox=2, num_classes=20):
        """
        :param split_grid: split size 7x7
        :param num_boundbox: number of boxes output for each cell
        :param num_classes: number of classes in the dataset (BDD100k)
        """
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")  # the paper does not average the loss, only the 'sum' in reduction
        self.S = split_grid
        self.B = num_boundbox
        self.C = num_classes
        self.lambda_coord = 5
        self.lambda_noobj = 0.5

    def forward(self, predictions, target):
        """
        :param predictions: tensor with the predictions from the model
        :param target:
        :return: loss
        """
        target = target.reshape(-1,self.S, self.S, self.C + self.B*5)
        # We need to make sure the prediction is reshaped to SxSx20
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        iou_box1 = intersection_over_union(predictions[..., 0:4], target[..., 0:4])
        iou_box2 = intersection_over_union(predictions[..., 5:9], target[...,  0:4])
        ious = torch.cat([iou_box1.unsqueeze(0), iou_box2.unsqueeze(0)], dim=0)
        iou_max, bestbox = torch.max(ious, dim=0)

        existence = target[..., 4].unsqueeze(3)  # Iobj_i

        # Box coordinates part
        box_preds = existence * (
            (
                    bestbox * predictions[..., 0:4]
                    + (1 - bestbox) * predictions[..., 5:9]
            )
        )

        box_targets = existence * target[..., 5:9]

        stability_offset = 1e-6
        box_preds[..., 2:4] = torch.sign(box_preds[..., 2:4]) * torch.sqrt(torch.abs(box_preds[..., 2:4]
                                                                                     + stability_offset))

        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_preds, end_dim=2),
            torch.flatten(box_targets, end_dim=2)
        )

        # OBJECT LOSS
        prediction_box = (
            bestbox * predictions[..., 9:10] + (1 - bestbox) * predictions[..., 4:5]
        )

        obj_loss = self.mse(
            torch.flatten(existence * prediction_box),
            torch.flatten((existence * target[..., 4:5]))
        )

        # NO OBJECT LOSS
        no_obj_loss = self.mse(
            torch.flatten((1-existence) * predictions[..., 4:5], start_dim=1),
            torch.flatten((1-existence) * target[..., 4:5], start_dim=1)
        )
        no_obj_loss += self.mse(
            torch.flatten((1 - existence) * predictions[..., 9:10], start_dim=1),
            torch.flatten((1 - existence) * target[..., 4:5], start_dim=1)
        )

        # CLASS LOSS
        class_loss = self.mse(
            torch.flatten(existence * predictions[..., 10:], end_dim=-2),
            torch.flatten(existence * target[..., 10:], end_dim=-2)
        )

        loss = (
            self.lambda_coord * box_loss
            + obj_loss
            + self.lambda_noobj * no_obj_loss
            + class_loss
        )

        return loss



