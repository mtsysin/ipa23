import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from utils import box_cxcywh_to_xyxy, generalized_box_iou

'''
Compute the matching assignment between the targets and the predictions of the network

Targets do not include empty objects. So there are more predictions than targets.
In this case, we do a one-to-one matching of the best predictions, leaving the others unmatched.
'''
class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()
        self.cost_class = cost_class  # Cost weight for classification
        self.cost_bbox = cost_bbox    # Cost weight for bounding box distance
        self.cost_giou = cost_giou    # Cost weight for generalized IoU
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "All costs can't be 0"  # Ensure at least one cost is non-zero

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # Softmax of predicted logits to get class probabilities
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        
        # Flatten predicted bounding boxes
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Concatenate target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Calculate classification cost based on probabilities
        cost_class = -out_prob[:, tgt_ids]

        # Calculate bounding box distance cost using L1 distance (p=1)
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Calculate generalized IoU cost
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Combine costs with respective weights
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()  # Reshape to (batch_size, num_queries, num_targets)

        # Split the cost matrix based on target sizes and perform Hungarian matching for each batch separately
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        
        # Return matched indices as tensors
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)
