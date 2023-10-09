import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import torch
import torch.utils.data as data
from torchvision.io import read_image

from utils import DetectionUtils

ANCHORS = [[(12,16),(19,36),(40,28)], [(36,75),(76,55),(72,146)], [(142,110),(192,243),(459,401)]]
# GRID_SCALES = [(12, 20), (24, 40), (48, 80)]
GRID_SCALES = [(48, 80), (24, 40), (12, 20)]
H, W = 720, 1280
BDD_100K_ROOT = "bdd100k/"
CLASS_DICT = {
            'pedestrian' :      0,
            'rider' :           1,
            'car' :             2,
            'truck' :           3, 
            'bus' :             4, 
            'train' :           5, 
            'motorcycle' :      6,
            'bicycle' :         7,
            'traffic light' :   8,
            'traffic sign' :    9,
            'other vehicle':    10,
            'trailer':          11,
            'other person':     12,
        }
REVERSE_CLASS_DICT = {value: key for key, value in CLASS_DICT.items()}
THRESH = 10         # Threshold for object area

class BDD100k(data.DataLoader):

    def __init__(self, root, train=True, transform=None, S=GRID_SCALES, anchors=ANCHORS):
        self.root = root 
        self.train = train
        self.transform = transform
        self.utils = DetectionUtils()

        self.detect = pd.read_json(self.root + 'labels/det_20/det_train.json') if train else pd.read_json(self.root + 'labels/det_20/det_val.json')
        self.detect.dropna(axis=0, subset=['labels'], inplace=True)

        self.class_dict = CLASS_DICT
        
        self.S = S
        self.C = len(self.class_dict)

        assert len(anchors) == len(S), "Anchors and scale prediction not matching"

        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.n_anchors = self.anchors.shape[0]
        self.n_anchors_scale = self.n_anchors // 3
        self.ignore_iou_thresh = 0.5

        #Initialize paths:
        self.img_path = self.root + 'images/100k/train/' if self.train else self.root + 'images/100k/val/'
        self.lane_path = self.root + 'labels/lane/masks/train/' if self.train else self.root + 'labels/lane/masks/val/'

        
    def __len__(self):
        return len(self.detect.index)

    def _iou_anchors(self, box, anchor):
        '''
        IOU for anchors boxes.
        Assuming the two boxes have the same center.
        '''
        intersection = torch.min(box[...,0], anchor[...,0]) * torch.min(box[...,1], anchor[...,1])
        union = (box[...,0] * box[...,1] + anchor[...,0] * anchor[...,1]) - intersection

        return intersection / union
  
    def __getitem__(self, index):
        target = self.detect.iloc[index]

        img = read_image(self.img_path + target['name']) 

        img = img.type(torch.float32)
        _, height, width = img.shape

        if self.transform:
            img = self.transform(img)

        #--------------------------------------------------------------------------------------------------------------------
        # Bounding Boxes

        annotations = target['labels']
        bboxes = []


        # Load image information (class and bounding box); convert bbox format (to x_c, y_c, w, h), keep them unscaled.
        for obj in annotations:
            obj_class = self.class_dict[obj['category']]
            bbox = list(obj['box2d'].values())
            bbox = self.utils.xyxy_to_xywh(bbox) 
            if bbox[2] >= THRESH and bbox[3] >= THRESH:
                box_tensor = torch.Tensor(([obj_class] + bbox.tolist()))
                bboxes.append(box_tensor) 

        label = [torch.zeros(self.n_anchors_scale, Sy, Sx, self.C + 5) for Sy, Sx in self.S]  # array with n_anchors_scale (=3) tensors for each scale 

        for bbox in bboxes:
            obj_class, x, y, w, h = bbox.tolist()
            x, w = x / width, w / width                                         # Normalizing by the whole image size
            y, h = y / height, h / height
            obj_class = int(obj_class)
            anchors_iou = self._iou_anchors(bbox[..., 3:5], self.anchors)       # Calculate IOU between unscaled anchors and unscaled bbox
            anchor_idx = torch.argmax(anchors_iou, dim=0)                       # Find anchor box index with highest IOU with the box
                                                                                # Calculated over a list of all anchors for all scales
            anchor_exist = [False] * 3

            scale_idx = torch.div(anchor_idx, self.n_anchors_scale, rounding_mode='floor')      # Find a scale to which the chosen anchor belongs

            anchor = anchor_idx % self.n_anchors_scale                                          # Find the index on the chosen anchor within its scale

            Sy, Sx = self.S[scale_idx]                                                          # Get the output grid size for the chosen scale

            i, j = int(Sy * y), int(Sx * x)                                                     # Find the indices of a cell to which the bbox center belongs
                                                                                                # Remember, x and y here are already scaled by the total size of the image

            exist = label[scale_idx][anchor, i, j, self.C]                                      # Check if we already have some object associated with this scale, anchor, and cell
                                                                                                # Checking objectness score which has index self.C in our last-dimension vector

            if not exist and not anchor_exist[scale_idx]:                                       # If there is nothing in this position
                label[scale_idx][anchor, i, j, self.C] = 1                                      # Now, we put "1" in the objectness position
                x_, y_ = Sx * x - j, Sy * y - i                                                 # Scale x, y, w, h to the size of the cell       
                w_, h_ = Sx * w, Sy * h
                bbox_ = torch.tensor([x_, y_, w_, h_])
                label[scale_idx][anchor, i, j, self.C+1:self.C+5] = bbox_
                label[scale_idx][anchor, i, j, obj_class] = 1
                anchor_exist[scale_idx] = True
            elif not exist and anchors_iou[anchor_idx] > self.ignore_iou_thresh:
                label[scale_idx][anchor, i, j, self.C] = -1

        # #--------------------------------------------------------------------------------------------------------------------
        # #Lane Mask
        # lane_name = os.path.splitext(target['name'])[0] + '.png'
        # lane_path2 = self.lane_path + lane_name

        # lane_mask = read_image(self.lane_path + lane_name)
        

        # #Binary
        # if self.transform:
        #     lane_mask = self.transform(lane_mask)                                               # Transform the lane mask in the same way we transform our image
        # lane_mask[lane_mask == 255.] = 1
        # lane_mask = torch.where((lane_mask==0)|(lane_mask==1), lane_mask^1, lane_mask)

        # #--------------------------------------------------------------------------------------------------------------------
        # # Multi Class
        # mask_image = torch.where(lane_mask != 255, 1, 0)
        # category_image = torch.bitwise_and(lane_mask, 0x7) * mask_image + (mask_image - 1)
        # crosswalk = (category_image == 0).to(torch.float32)
        # double_other = (category_image == 1).to(torch.float32)
        # double_white = (category_image == 2).to(torch.float32)
        # double_yellow = (category_image == 3).to(torch.float32)
        # road_curb = (category_image == 4).to(torch.float32)
        # single_other = (category_image == 5).to(torch.float32)
        # single_white = (category_image == 6).to(torch.float32)
        # single_yellow = (category_image == 7).to(torch.float32)
        # lane_background = (category_image == 8).to(torch.float32)
        # lane = torch.stack([lane_background, single_yellow, single_white, single_other, road_curb,double_yellow, double_white, double_other, crosswalk], dim=0)
        # # #--------------------------------------------------------------------------------------------------------------------

        # #--------------------------------------------------------------------------------------------------------------------
        # #Drivable Area
        # drive_path = self.root + 'labels/drivable/masks/train/' if self.train else self.root + 'labels/drivable/masks/val/'
        # drive_name = os.path.splitext(target['name'])[0] + '.png'
        # drive_path2 = drive_path + drive_name
        # drive_mask = read_image(drive_path + drive_name)

        # if self.transform:
        #     drive_mask = self.transform(drive_mask)[0]
        # direct_mask = torch.where(drive_mask == 0, 1, 0)
        # alternative_mask = torch.where(drive_mask == 1, 1, 0)
        # drive_background = torch.where(drive_mask == 2, 1, 0)
        # drivable = torch.stack([drive_background, direct_mask, alternative_mask], dim= 0)
        # #--------------------------------------------------------------------------------------------------------------------

        # seg = self._build_seg_target(lane_path2, drive_path2)

        return img, label, 4


    def _build_seg_target(self, lane_path, drivable_path):
        '''Build groundtruth for  segmentation
        Note: This combines the lanes and drivable masks into one
        Args:
            lane_path (str): path to lane binary mask
            drivable_path (str): path to drivable binary mask
        '''
        lane = cv2.imread(lane_path)[..., 0]
        drivable = cv2.imread(drivable_path)[..., 0]
        lanes = np.bitwise_and(lane, 0b111)
        lane_mask, drivable_mask = [], []
        for i in range(9):
            lane_mask.append(np.where(lanes==i, 1, 0))
            if i in range(3):
                drivable_mask.append(np.where(drivable==i, 1, 0))
        lane_mask, drivable_mask = np.stack(lane_mask), np.stack(drivable_mask)
        lane_mask = torch.tensor(lane_mask)
        drivable_mask = torch.tensor(drivable_mask)
        if self.transform:
            lane_mask, drivable_mask = self.transform(lane_mask), self.transform(drivable_mask)
        mask = torch.cat((lane_mask, drivable_mask), axis=0)
        return mask
