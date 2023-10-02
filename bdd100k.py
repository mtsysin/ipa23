import os
import cv2
import numpy as np

import pandas as pd

import torch
import torch.utils.data as data
from torchvision.io import read_image

from utils import DetectionUtils

ANCHORS = [[(12,16),(19,36),(40,28)], [(36,75),(76,55),(72,146)], [(142,110),(192,243),(459,401)]]

class BDD100k(data.DataLoader):
    '''
    Author: Pume Tuchinda
    Data Loader for BDD100k Dataset
        - Object Bounding Box
        - Lane Mask
        - Drivable Area Mask
    '''

    def __init__(self, root, train=True, transform=None, S=[(12, 20), (24, 40), (48, 80)], anchors=ANCHORS):
        self.root = root 
        self.train = train
        self.transform = transform
        self.utils = DetectionUtils()

        self.detect = pd.read_json(self.root + 'labels/det_20/det_train.json') if train else pd.read_json(self.root + 'labels/det_20/det_val.json')
        self.detect.dropna(axis=0, subset=['labels'], inplace=True)

        self.class_dict = {
            'pedestrian' : 1,
            'rider' : 2,
            'car' : 3,
            'truck' : 4, 
            'bus' : 5, 
            'train' : 6, 
            'motorcycle' : 7,
            'bicycle' : 8,
            'traffic light' : 9,
            'traffic sign' : 10,
            'other vehicle': 11,
            'trailer': 12,
            'other person': 13,
        }
        self.S = S
        self.C = len(self.class_dict)

        assert len(anchors) == len(S), "Anchors and scale prediction not matching"

        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.n_anchors = self.anchors.shape[0]
        self.n_anchors_scale = self.n_anchors // 3
        self.ignore_iou_thresh = 0.5
        self.utils = DetectionUtils()
        
    def __len__(self):
        return len(self.detect.index)

    def _iou_anchors(self, box, anchor):
        '''
        IOU for anchors boxes
        '''
        intersection = torch.min(box[...,0], anchor[...,0]) * torch.min(box[...,1], anchor[...,1])
        union = (box[...,0] * box[...,1] + anchor[...,0] * anchor[...,1]) - intersection

        return intersection / union
  
    def __getitem__(self, index):
        target = self.detect.iloc[index]

        img_path = self.root + 'images/100k/train/' if self.train else self.root + 'images/100k/val/'
        img = read_image(img_path + target['name']) 
        img = img.type(torch.float32)
        _, height, width = img.shape

        if self.transform:
            img = self.transform(img)
        #--------------------------------------------------------------------------------------------------------------------
        #Bounding Boxes
        label = [torch.zeros(3, Sx, Sy, self.C + 5) for Sx, Sy in self.S]   

        annotations = target['labels']
        bboxes = []
        for obj in annotations:
            obj_class = self.class_dict[obj['category']]
            bbox = list(obj['box2d'].values())
            bbox = self.utils.xyxy_to_xywh(bbox) 
            box_tensor = torch.Tensor(([obj_class] + bbox.tolist()))
            bboxes.append(box_tensor) 

        label = [torch.zeros(self.n_anchors_scale, Sy, Sx, self.C + 5) for Sy, Sx in self.S]

        for bbox in bboxes:
            obj_class, x, y, w, h = bbox.tolist()
            x, w = x / width, w / width
            y, h = y / height, h / height
            obj_class = int(obj_class)
            anchors_iou = self._iou_anchors(bbox[..., 3:5], self.anchors)
            anchor_idx = torch.argmax(anchors_iou, dim=0)
            anchor_exist = [False] * 3

            scale_idx = torch.div(anchor_idx, self.n_anchors_scale, rounding_mode='floor')
            anchor = anchor_idx % self.n_anchors_scale
            Sy, Sx = self.S[scale_idx]
            i, j = int(Sy * y), int(Sx * x)
            exist = label[scale_idx][anchor, i, j, self.C]
            if not exist and not anchor_exist[scale_idx]: 
                label[scale_idx][anchor, i, j, self.C] = 1
                x_, y_ = Sx * x - j, Sy * y - i 
                w_, h_ = Sx * w, Sy * h
                bbox_ = torch.tensor([x_, y_, w_, h_])
                label[scale_idx][anchor, i, j, self.C+1:self.C+5] = bbox_
                label[scale_idx][anchor, i, j, obj_class] = 1
                anchor_exist[scale_idx] = True
            elif not exist and anchors_iou[anchor_idx] > self.ignore_iou_thresh:
                label[scale_idx][anchor, i, j, self.C] = -1

        #--------------------------------------------------------------------------------------------------------------------
        #Lane Mask
        lane_path = self.root + 'labels/lane/masks/train/' if self.train else self.root + 'labels/lane/masks/val/'
        lane_name = os.path.splitext(target['name'])[0] + '.png'
        lane_path2 = lane_path + lane_name

        lane_mask = read_image(lane_path + lane_name)
        

        #Binary
        if self.transform:
            lane_mask = self.transform(lane_mask)
        lane_mask[lane_mask == 255.] = 1
        lane_mask = torch.where((lane_mask==0)|(lane_mask==1), lane_mask^1, lane_mask)

        #--------------------------------------------------------------------------------------------------------------------
        # Multi Class
        mask_image = torch.where(lane_mask != 255, 1, 0)
        category_image = torch.bitwise_and(lane_mask, 0x7) * mask_image + (mask_image - 1)
        crosswalk = (category_image == 0).to(torch.float32)
        double_other = (category_image == 1).to(torch.float32)
        double_white = (category_image == 2).to(torch.float32)
        double_yellow = (category_image == 3).to(torch.float32)
        road_curb = (category_image == 4).to(torch.float32)
        single_other = (category_image == 5).to(torch.float32)
        single_white = (category_image == 6).to(torch.float32)
        single_yellow = (category_image == 7).to(torch.float32)
        lane_background = (category_image == 8).to(torch.float32)
        lane = torch.stack([lane_background, single_yellow, single_white, single_other, road_curb,double_yellow, double_white, double_other, crosswalk], dim=0)
        # #--------------------------------------------------------------------------------------------------------------------

        #--------------------------------------------------------------------------------------------------------------------
        #Drivable Area
        drive_path = self.root + 'labels/drivable/masks/train/' if self.train else self.root + 'labels/drivable/masks/val/'
        drive_name = os.path.splitext(target['name'])[0] + '.png'
        drive_path2 = drive_path + drive_name
        drive_mask = read_image(drive_path + drive_name)

        if self.transform:
            drive_mask = self.transform(drive_mask)[0]
        direct_mask = torch.where(drive_mask == 0, 1, 0)
        alternative_mask = torch.where(drive_mask == 1, 1, 0)
        drive_background = torch.where(drive_mask == 2, 1, 0)
        drivable = torch.stack([drive_background, direct_mask, alternative_mask], dim= 0)
        #--------------------------------------------------------------------------------------------------------------------

        seg = self._build_seg_target(lane_path2, drive_path2)
        return img / 255., label, seg


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
