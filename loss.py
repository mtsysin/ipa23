import torch
import torch.nn as nn
import torch.nn.functional as F
from bdd100k import ANCHORS, GRID_SCALES, H, W
import torchvision
from evaluate import DetectionMetric
from utils import DetectionUtils

utils = DetectionUtils()

class MultiLoss(nn.Module):
    '''
    Author: Pume Tuchinda
    Multi Task Loss Function
        - Acts entirely as just a combination of the two losses for the two tasks we are doing
    '''
    def __init__(self, alpha_det=1, alpha_lane=1):
        super().__init__()
        self.det_loss = DetectionLoss()
        self.lane_loss = SegmentationLoss()
        self.alpha_det = alpha_det
        self.alpha_lane = alpha_lane

    def forward(self, dets, seg, det_targets, lane_targets, drivable_targets):
        lanes = seg[... , 0:9]
        drivable = seg[... , 9:12]
        det_loss = self.det_loss(dets, det_targets)    
        lane_loss = self.lane_loss(lanes, lane_targets)
        drive_loss = self.lane_loss(drivable, drivable_targets)

        return self.alpha_det * det_loss + self.alpha_lane * lane_loss + self.alpha_lane * drive_loss


class DetectionLoss(nn.Module):
    '''
    Author: William Stevens
            Pume Tuchinda
    Detection Loss Function:
        1. Distance IOU Loss (size and location of bounding box)
            - Loss related to IOU of bounding box
            - Loss related to center placement of bounding box
            - Loss related to aspect ratio of bounding box
        2. Focal Loss (classification of object)
            - Classification loss
    Args:
        batch_size (int): number of batches
        n_classes (int): number of classes in dataset      
        alpha_class (float): loss parameter for classification
        alpha_box (float):  loss parameter for bounding box 
        alpha_obj (float): loss parameter for object score
    '''
    def __init__(self, n_classes=13, alpha_class=1., alpha_box=1., alpha_obj=1., anchors=ANCHORS, grid_scales=GRID_SCALES, size=[H, W]):
        super().__init__()
        self.metric = DetectionMetric()

        self.C = n_classes
        self.alpha_class = alpha_class
        self.alpha_box = alpha_box
        self.alpha_obj = alpha_obj

        # Scale anchors to the grid size
        self.anchors = torch.tensor(anchors)
        self.grid_scales = torch.tensor(grid_scales)
        self.size = torch.tensor(size)
        cell_sizes = (self.size / self.grid_scales).repeat_interleave(3, dim=0).view(3, 3, 2)
        self.anchors = self.anchors / cell_sizes

        self.loss = nn.MSELoss()
        self.loss2 = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        '''
        Forward pass of Detection Loss
        Args:
            preds (tensor): [det_head1, det_head2, det_head3]
                - each det_head is a detection from the different scales where we are detection bbox
                - det_head shape: (batch_size, n_anchors, scale, scale, num_classes + 5)
            target (tensor): Groundtruth, size : (batch_size, n_anchors, scale, scale, n_classes + 5)
        Returns:
            loss (tensor): loss value
        '''
        ciou_loss, obj_loss, noobj_loss, class_loss = 0, 0, 0, 0
        for i, pred in enumerate(preds):                                                        # For each scale
            target = targets[i].to(pred.device)

            Iobj = target[..., self.C] == 1
            Inoobj = Iobj == 0

            # batch_size, n_anchors, gy, gx, n_outputs = pred.shape
            # gridy, gridx = torch.meshgrid([torch.arange(gy), torch.arange(gx)], indexing='ij')
            with torch.no_grad():    
                anchor = self.anchors[i].view(1, 3, 1, 1, 2).to(pred.device)          # Get the anchor boxes corresponding to the chosen scale, view: (Batch, anchor index, sy, sx, box dimension)

            # In target, the bounding box x and y coordiantes and w and h are scaled by the size of the cell!!!!
            # target[..., self.C+3:self.C+5] = torch.log(1e-6 + target[..., self.C+3:self.C+5] / anchor)

            bbox_detection = torch.cat(
                (
                    pred[..., self.C+1:self.C+3][Iobj].sigmoid(),
                    (pred[..., self.C+3:self.C+5] + torch.log(anchor))[Iobj].exp()
                ),
                dim = -1
            )
            # print("maximum:", torch.max(pred[..., self.C+3:self.C+5]))

            # print("pred.shape", pred.shape)
            # print("target.shape", target.shape)

            # print("shape of OBJ part", pred[..., self.C+1:self.C+5][Iobj].shape)
            # print("sample of OBJ part", pred[..., self.C+1:self.C+5][Iobj][:2])


            iou = self.metric.box_iou(bbox_detection, target[..., self.C+1:self.C+5][Iobj], xyxy=False, CIoU=True).mean()
            ciou_loss += 1 - iou
            # ciou_loss += self.loss(pred[..., self.C+1:self.C+5][Iobj], target[..., self.C+1:self.C+5][Iobj])

            # print('prediction:' ,pred[..., self.C+1:self.C+5][Iobj])
            # print('target: ', target[..., self.C+1:self.C+5][Iobj])

            # obj_loss += self._focal_loss(pred[..., self.C:self.C+1], target[..., self.C:self.C+1])

            obj_loss += self._focal_loss(pred[..., self.C:self.C+1][Iobj], target[..., self.C:self.C+1][Iobj])
            noobj_loss += self._focal_loss(pred[..., self.C:self.C+1][Inoobj], target[..., self.C:self.C+1][Inoobj])
            class_loss += self._focal_loss(pred[..., :self.C][Iobj], target[..., :self.C][Iobj])

        return self.alpha_box * ciou_loss + self.alpha_class * class_loss + self.alpha_obj * (obj_loss + noobj_loss), (ciou_loss.item(), obj_loss.item(), noobj_loss.item(), class_loss.item())
    
    def _focal_loss(self, preds, targets, alpha=0.25, gamma=2, reduction="mean"):
        '''
        Focal Loss Implementation
        Author: William Stevens
        - Inspired by the official paper on Focal Loss (https://arxiv.org/abs/1708.02002)
        - Enhancement to cross entropy loss which improves classification accuracy casued by class imbalances.
        Args:
            - preds (tensor): prediction tensor containing confidence scores for each class.
            - target (tensor): ground truth containing correct class labels.
            - alpha (tensor): class weights to represent the class imbalance.
            - gamma (int): Focal term. Constant, tunable exponent applied to the modulating factor which amplifies
            loss emphasis on difficult learning tasks that result in misclassification.
        '''
        p = torch.sigmoid(preds)
        ce_loss = F.binary_cross_entropy_with_logits(preds, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss *= alpha_t 

        if reduction == "mean":
            return loss.mean() 
        else:
            return loss.sum() 

class SegmentationLoss(nn.Module):
    '''
    Segmentation Loss Function:
        1. Dice Binary Cross Entropy Loss
        2. IOU Loss 
            - only used for lane due to the thiness of the lane lines
    Author: Daniyaal Rasheed
            Pume Tuchinda
    Args:
        - preds (tensor): prediction tensor containing probaility of whether a pixel is a type of lane or not
        - target (tensor): ground truth containing correct classification.
    Returns:
        - IOU loss (scalar): Summation of IOU for every lane in the batch
    '''
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-6

    def _focal_loss(self, pred, target):
        num_classes = pred.size(1)
        target = target.squeeze(dim=1)
        loss = 0

        for cls in range(num_classes):
            targetClass = target[:, cls, ...]
            predClass = pred[:, cls, ...]

            logpt = F.binary_cross_entropy_with_logits(predClass, targetClass, reduction='none')
            pt = torch.exp(-logpt)
            focal = (1 - pt).pow(2)

            classLoss = focal * logpt
            loss += classLoss.mean()
        
        return loss

    # def _dice_loss(self, pred, target):
    #     batchSize, numClasses = pred.size(0), pred.size(1)
    #     target = target.view(batchSize, numClasses, -1)
    #     pred = pred.view(batchSize, numClasses, -1)

    #     target = F.one_hot(target.long(), numClasses).permute(0, 2, 1)
    
    #     intersection = torch.sum(pred * target, dim=(0,2))
    #     cardinality = torch.sum(pred + target, dim=(0,2))

    #     diceScore = (2 * intersection) / cardinality.clamp(1e-6)

    #     loss = 1 - diceScore
    #     mask = target.sum((0,2)) > 0
    #     loss * mask.to(loss.dtype)

    #     return loss.mean()

    def forward(self, pred, target):
        focalLoss = self._focal_loss(pred, target)
        #diceLoss = self._dice_loss(pred, target)

        return focalLoss# + diceLoss
