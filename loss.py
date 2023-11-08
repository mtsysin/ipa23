import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import Tuple, List
from torch import Tensor
from utils import xywh_to_xyxy

class BipartiteMatchingLoss(nn.Module):
    def __init__(self, num_classes, empty_obj_coef = 0.1, class_lambda = 1.0, bbox_lambda = 1.0, iou_lambda = 1.0):
        '''
            class_loss (nn.Module): The loss function for class predictions.
            bbox_loss (nn.Module): The loss function for bounding box predictions.
        '''
        super(BipartiteMatchingLoss, self).__init__()
        self.class_lambda = class_lambda
        self.bbox_lambda = bbox_lambda
        self.iou_lambda = iou_lambda
        self.num_classes = num_classes

        # Buffer to store the relative weight of the classes. All of them are one except for the last one
        self.register_buffer(
            'class_weights', torch.cat(
                torch.ones(self.num_classes), 
                torch.tensor([self.eos_coef])
            )
        )

        # Make a dictionary for storing the losses
        self.loss_cache = {}

    def forward(self, preds: Tuple[Tensor, Tensor], targets: List[Tuple[Tensor, Tensor]]):
        '''
            preds (tuple): Tuple containing predicted class logits and bounding box coordinates.
                The tuple has the following shapes: logits: [batch, num_queries, num_calsses]
                                                    boxes: [batch, num_queries, 4]
            targets: List of tuples containing target class labels and target bounding box coordinates.
                Shapes are: tuple for every image in the batch: lables: [num_target_boxes, 1]
                                                                boxes: [num_target_boxes, 4]

        Returns: Tuple of class loss and bounding box loss.
        '''
        # Unpack predictions
        pred, pred_bbox = preds[0], preds[1]
        targ, targ_bbox = targets[0], targets[1]

        # Get matching indices
        indices = self.bipartite_matching(pred_bbox, targ_bbox)
        # Now we need to calculate the respective losses of both bboxes and labels. Note that we'll use our
        # weights to scale down the loss of the no-obkect class.

        # Compute class and bbox losses
        class_loss = self.compute_class_loss(pred, targ, indices)
        bbox_loss = self.compute_bbox_loss(pred_bbox, targ_bbox, indices)

        return class_loss, bbox_loss

    def bipartite_matching(self, preds, targets, transform_bbox = lambda x: x):
        '''
            preds: See forward description
            targets: See forward description

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)

        '''

        batch_size, num_queries = preds[0].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = preds[0].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = preds[1].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([targ[0] for targ in targets]).squeeze(-1) # [num_targ(batch1) + num_targ(batch2) + ...] 
        tgt_bbox = torch.cat([targ[1] for targ in targets]) # [num_targ(batch1) + num_targ(batch2) + ... , 4] 

        # Compute the classification cost. Here we just approximate with
        # (1 - predicted prpbability of true class)
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        print(out_prob.shape,tgt_ids.shape )
        cost_class = -out_prob[:, tgt_ids] # [batch_size * num_queries, 1], only one class prob

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1) # [batch_size * num_queries, num_target_bboxes in all batches combined]
        # Compute the giou cost betwen boxes (will be ciou in our implementation)
        cost_iou = -box_iou(transform_bbox(out_bbox), transform_bbox(tgt_bbox), pairwise=True) # [batch_size * num_queries, num_target_bboxes]

        # Final cost matrix
        cost_matrix = self.bbox_lambda * cost_bbox + self.class_lambda * cost_class + self.iou_lambda * cost_iou # [batch_size * num_queries, num_target_bboxes]
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu() # [batch_size, num_queries, num_target_bboxes]

        sizes = [len(v[1]) for v in targets] # length of target bbox tensors (or number of bboxes per target)
        # split the sizes along the last dimension and compute Hungarian matching for each batch
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))] 
        # Return
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
    
    def apply_permutation(self, preds, targets, indices):
        '''
            preds: See forward description
            targets: See forward description
            indices: See bipartite matching description

        Returns:
            preds: Same format but in the order given by indices
            targets: Same format but in the order given by indices
        '''
        preds_ = (
            torch.gather(preds[0], index=indices) # logits
        )
        targets_ = [(
            targets[0][indices[i][1]], # labels
            targets[1][indices[i][1], :] # bboxes
        ) for i, (labels, box) in enumerate(targets)]

    def _convert_indices_permutation_to_target_mask(self, indices):
        # Essentially a list of batch numbers multiplied by the size of the target for each image
        batch_idx = torch.cat([torch.full_like(target_permutation, i) for i, (_, target_permutation) in enumerate(indices)])
        # List of corresponding selected targets for each 
        target_idx = torch.cat([target_permutation for (_, target_permutation) in indices])
        return batch_idx, target_idx
    
    def _get_precision_at_k(self, selected_logits, selected_classes):
        '''
        Takes selected logits from predictions and selected targets and returns the accuracy
        '''
        return 100 - accuracy(selected_logits, selected_classes)[0]


    def compute_class_loss(self, preds, targets, indices, store_precision_at_k = None):
        '''
            pred (Tensor): Predicted class probabilities
            targ (Tensor): Target class labels.
            indices (List): Indices for matching predicted and target boxes.
        '''
        # 1. Create a target calss tensor which will contain no_object (nclasses) class for all entries
        # Except for those which are in the matched indices. Shape is [num_preds_batch1 + num_preds_batch2 + ...]
        # We concatenate all 

        pred_logits = preds[0]
        target_all_concat = torch.cat(
            [targets[i] for i, (_, target_permutation) in enumerate(indices)]
        )
        # Make a default target with "no_class" label and shape [batch_size, num_queries]
        target_classes = torch.full(pred_logits.shape[:2], self.num_classes,
                            dtype=torch.int64, device=pred_logits.device)
        # Populate the selected queries from the matcher:
        target_indices = self._convert_indices_permutation_to_target_mask(indices)
        target_classes[target_indices] = target_all_concat

        loss_class = F.cross_entropy(pred_logits.permute(0, 2, 1), target_classes, weight=self.class_weights)

        if store_precision_at_k:
            self.loss_cache['class_error'] = self._get_precision_at_k(pred_logits[target_indices], target_all_concat)

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


def box_iou(box1, box2, xyxy=True, CIoU=True, pairwise = False):
    """
    Calculates IOU fucntionality
    If pairwise, will generate a pair from [..., M, 4] and [..., N, 4] to [..., M, N]
    Args:
        box1 (tensor): bounding box 1
        box2 (tensor): bounding box 2
        xyxy (bool): is format of boudning box in x1 y1 x2 y2
        CIoU (bool): if true calculate CIoU else IoU
    Returns:
        Tensor int the format:
        [..., iou_value]
    """

    EPS = 1e-6

    if not xyxy:
        box1 = xywh_to_xyxy(box1)
        box2 = xywh_to_xyxy(box2)

    if pairwise:
        box1 = box1[..., :, None, :] # [..., M, *N, 4]
        box2 = box2[..., None, :, :] # [..., *M, N, 4]

    box1_x1 = box1[..., 0:1]
    box1_y1 = box1[..., 1:2]
    box1_x2 = box1[..., 2:3]
    box1_y2 = box1[..., 3:4]
    box2_x1 = box2[..., 0:1]
    box2_y1 = box2[..., 1:2]
    box2_x2 = box2[..., 2:3]
    box2_y2 = box2[..., 3:4]

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
        
        iou = iou - dist_penalty - aspect_ratio_penalty
    
    return iou.squeeze(-1)