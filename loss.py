import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import Tuple, List
from torch import Tensor
from utils import xywh_to_xyxy, box_iou, accuracy

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
        indices = self.bipartite_matching(preds, targets)
        # gey number of boxes for normalizarion purposes:
        num_boxes = torch.as_tensor(
            [sum(len(t["labels"]) for t in targets)], 
            dtype=torch.float, 
            device=next(iter(preds.values())).device
        )
        # Now we need to calculate the respective losses of both bboxes and labels. Note that we'll use our
        # weights to scale down the loss of the no-obkect class.
        loss_class = self.compute_class_loss(preds, targets, indices)
        loss_bbox, loss_ciou = self.compute_bbox_loss(preds, targets, indices, num_boxes)
        losses = {
            "loss_bbox": loss_bbox, 
            "loss_class": loss_class, 
            "loss_ciou": loss_ciou
        }

        return losses

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
        # Something like 1, 1, 1, 1, 2, 2, 2, 2, 2,2 , 3, 3, 3, 3....
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
            preds: See forward descriptions
            targets: See forward descriptions
            indices: See bipartite matching description
        '''
        # 1. Create a target calss tensor which will contain no_object (nclasses) class for all entries
        # Except for those which are in the matched indices. Shape is [num_preds_batch1 + num_preds_batch2 + ...]
        # We concatenate all 

        pred_logits = preds[0]
        target_all_concat = torch.cat(
            [target_one_img[0][target_permutation] for target_one_img, (_, target_permutation) in zip(targets, indices)]
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
    

    def compute_bbox_loss(self, preds, targets, indices, num_boxes):
        '''
            preds: See forward descriptions
            targets: See forward descriptions
            indices: See bipartite matching description
        '''
        # Logic similar to the class loss. We only need to take care of bothe the l1 loss
        # as wel as the CIOU loss.

        # Get selected bboxes:
        idx = self._convert_indices_permutation_to_target_mask(indices)
        # Get pred bboxes for all queries
        pred_bboxes = preds[1][idx]
        # Get all target bboxes, shape [num_preds_batch1 + num_preds_batch2 + ..., 4]
        target_bboxes_all_concat = torch.cat(
            [target_one_img[0][target_permutation] for target_one_img, (_, target_permutation) in zip(targets, indices)],
            dim=0
        )
        # Convert boxes to the indeed format:
        loss_bbox = F.l1_loss(pred_bboxes, target_bboxes_all_concat, reduction='none') / num_boxes

        loss_ciou = 1 - box_iou(
            pred_bboxes,
            target_bboxes_all_concat,
            xyxy = False,
            CIoU = True,
            pairwise = False
        )
        return loss_bbox, loss_ciou
