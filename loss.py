import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import Tuple, List
from torch import Tensor

class BipartiteMatchingLoss(nn.Module):
    def __init__(self, class_lambda = 1.0, bbox_lambda = 1.0, iou_lambda = 1.0):
        '''
            class_loss (nn.Module): The loss function for class predictions.
            bbox_loss (nn.Module): The loss function for bounding box predictions.
        '''
        super(BipartiteMatchingLoss, self).__init__()
        self.class_lambda = class_lambda
        self.bbox_lambda = bbox_lambda
        self.iou_lambda = iou_lambda

    def forward(self, preds: Tuple[Tensor, Tensor], targets: List[Tuple[Tensor, Tensor]]):
        '''
            preds (tuple): Tuple containing predicted class logits and bounding box coordinates.
                The tuple has the following shapes: logits: [batch, num_queries, num_calsses]
                                                    boxes: [batch, num_queries, 4]
            targets (tuple): List of tuples containing target class labels and target bounding box coordinates.
                Shapes are: tuple for every image in the batch: lables: [num_target_boxes, 1]
                                                                boxes: [num_target_boxes, 4]

        Returns: Tuple of class loss and bounding box loss.
        '''
        pred, pred_bbox = preds[0], preds[1]
        targ, targ_bbox = targets[0], targets[1]
        indices = self.bipartite_matching(pred_bbox, targ_bbox)

        class_loss = self.compute_class_loss(pred, targ, indices)
        bbox_loss = self.compute_bbox_loss(pred_bbox, targ_bbox, indices)

        return class_loss, bbox_loss

    def bipartite_matching(self, preds, targets, iou_function = None, transform_bbox = lambda x: x):
        '''
            pred_bbox (Tensor): Predicted bounding boxes.
            targ_bbox (Tensor): Target bounding boxes.

        Returns:
            List of indices that indicate the associations between predicted and target boxes.

        '''

        if iou_function == None:
            iou_function = self.ciou

        batch_size, num_queries = preds[0].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = preds[0].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = preds[1].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v[0] for v in targets])
        tgt_bbox = torch.cat([v[1] for v in targets])

        # Compute the classification cost. Here we just approximate with
        # (1 - predicted prpbability of true class)
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1) # [batch_size, num_queries, num_target_bboxes]
        # Compute the giou cost betwen boxes
        cost_iou = -iou_function(transform_bbox(out_bbox), transform_bbox(tgt_bbox)) # [batch_size, num_queries, num_target_bboxes]

        # Final cost matrix
        cost_matrix = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_iou # [batch_size, num_queries, num_target_bboxes]
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        sizes = [len(v[1]) for v in targets] # length of target bbox tensors
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

        # # Compute pairwise CIoU loss
        # cost = -F.pairwise_distance(pred_bbox.unsqueeze(2), targ_bbox.unsqueeze(1), p=2)

        # # Add costs for "no object" class
        # no_object_cost = torch.zeros(batch_size, num_queries, 1).to(pred_bbox.device)
        # cost = torch.cat([cost, no_object_cost], dim=2)

        # # Linear sum assignment to determine optimal indicees
        # indices = linear_sum_assignment(cost.permute(0, 2, 1))
        # indices = [(i, indices[i]) for i in range(batch_size)]
        # return indices

    def ciou(self, boxes1, boxes2):
        '''
            boxes1 (Tensor): First set of bounding boxes.
            boxes2 (Tensor): Second set of bounding boxes.
        '''
        intersection = self.intersection(boxes1, boxes2)
        area1 = self.area(boxes1)
        area2 = self.area(boxes2)
        union = area1.unsqueeze(2) + area2.unsqueeze(1) - intersection
        iou = intersection / union
        ciou = iou - self.union(boxes1, boxes2) / union
        return ciou

    def intersection(self, boxes1, boxes2):
        '''
            boxes1 (Tensor): First set of bounding boxes.
            boxes2 (Tensor): Second set of bounding boxes.
        '''
        top_left = torch.max(boxes1[..., :2], boxes2[..., :2])
        bottom_right = torch.min(boxes1[..., 2:], boxes2[..., 2:])
        inter = torch.clamp(bottom_right - top_left, min=0)
        return inter[..., 0] * inter[..., 1]

    def area(self, boxes):
        return (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])

    def union(self, boxes1, boxes2):
        '''
            boxes1 (Tensor): First set of bounding boxes.
            boxes2 (Tensor): Second set of bounding boxes.
        '''
        intersection = self.intersection(boxes1, boxes2)
        area1 = self.area(boxes1)
        area2 = self.area(boxes2)
        union = area1 + area2 - intersection
        return union

    def compute_class_loss(self, pred, targ, indices):
        '''
            pred (Tensor): Predicted class probabilities
            targ (Tensor): Target class labels.
            indices (List): Indices for matching predicted and target boxes.
        '''
        loss_class = F.cross_entropy(pred.permute(0, 2, 1), targ, reduction='none')
        loss_class = torch.gather(loss_class, dim=2, index=indices)
        loss_class = loss_class.mean()
        return loss_class

    def compute_bbox_loss(self, pred_bbox, targ_bbox, indices):
        '''
        Compute the bounding box regression loss.

        Args:
            out_bbox (Tensor): Predicted bounding boxes.
            tgt_bbox (Tensor): Target bounding boxes.
            indices (List): Indices for matching predicted and target boxes.

        Returns:
            Bounding box regression loss.

        '''
        pred_boxes = pred_bbox[torch.arange(pred_bbox.shape[0])[:, None, None], torch.arange(pred_bbox.shape[1])[None, :, None], indices]
        loss_bbox = 1 - self.ciou(pred_boxes, targ_bbox)
        loss_bbox = loss_bbox.mean()
        return loss_bbox
