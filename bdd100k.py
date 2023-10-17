import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import torch
import torch.utils.data as data
from torchvision.io import read_image
import torchvision.transforms.v2 as transforms
from torchvision import tv_tensors
import random
from augmentation import mosaic_augmentation, cls_bbox_augment, mixup_augmentation

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
BBOX_SIZE_THRESHOLD = None         # Threshold for object area
MOSAIC_PROB = 1.0
AUGMENT_PROB = 1.0
MIXUP_AFTER_MOS = 0
MIXUP_PROB = 1.0


class BDD100k(data.DataLoader):

    def __init__(self, 
            root, 
            dataset_type = 'train', 
            transform=None, 
            S=GRID_SCALES, 
            anchors=ANCHORS,
            target = 'yolo'
        ):
        self.root = root 
        self.dataset_type = dataset_type
        self.transform = transform
        self.utils = DetectionUtils()
        self.target = target
        

        self.detect = pd.read_json(self.root + 'labels/det_20/det_train.json') if self.dataset_type == 'train' else pd.read_json(self.root + 'labels/det_20/det_val.json')
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
        self.img_path = self.root + 'images/100k/train/' if self.dataset_type == 'train' else self.root + 'images/100k/val/'
        self.lane_path = self.root + 'labels/lane/masks/train/' if self.dataset_type == 'train' else self.root + 'labels/lane/masks/val/'

        
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
    
    def load_cls_bboxes(self, index, enforce_type = None):
        # Load image information (class and bounding box); convert bbox format (to x_c, y_c, w, h), keep them unscaled.
        annotations = self.detect.iloc[index]['labels']
        bboxes = []
        for obj in annotations:
            obj_class = self.class_dict[obj['category']]
            bbox = list(obj['box2d'].values())
            _, _, w, h = self.utils.xyxy_to_xywh(bbox) 
            if BBOX_SIZE_THRESHOLD and (w < BBOX_SIZE_THRESHOLD or h < BBOX_SIZE_THRESHOLD):
                continue
            box_tensor = torch.Tensor(([obj_class] + bbox))
            bboxes.append(box_tensor) 
        return torch.stack(bboxes).to(enforce_type) if enforce_type else torch.stack(bboxes)
    
    def load_img_and_bboxes(self, index, augmentations = None, enforce_bbox_type = None):
        img = read_image(self.img_path + self.detect.iloc[index]['name'])
        bboxes = self.load_cls_bboxes(index, enforce_type=enforce_bbox_type)
        
        _, height, width = img.shape
        # print(img.shape)
        # print(img[:, 2:4, 5:7])
        # print(bboxes)


        augmentation_scheme = self._get_augmentatiion_scheme((height, width))

        if random.random() < AUGMENT_PROB:
            img, bboxes = self.cls_bbox_augment(augmentation_scheme, img, bboxes)

        return img, bboxes

    def _get_augmentatiion_scheme(self, size):
        return transforms.Compose([
            transforms.RandomPerspective(distortion_scale=0.5, p=1, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomResizedCrop(size=size, antialias=True),
            transforms.RandomHorizontalFlip(p=1),
            transforms.ClampBoundingBoxes(),
            transforms.SanitizeBoundingBoxes(),
            transforms.RandomPhotometricDistort(),
        ])
    
    def cls_bbox_augment(self, augmentation, image, bboxes):
        return cls_bbox_augment(augmentation, image, bboxes)
    
    def load_mosaic(self, base_index):
        # get random indices to build a mosaic augmentation:
        idxs = [base_index]
        while len(idxs) < 4:
            candidate = random.randint(0, self.__len__() - 1)
            if candidate not in idxs:
                idxs.append(candidate)

        images, bboxes_l = zip(*[list(self.load_img_and_bboxes(i)) for i in idxs])

        return mosaic_augmentation(images, bboxes_l)

    def __getitem__(self, index):
       
        if random.random() < MOSAIC_PROB:       # Run mosaic
            img, bboxes = self.load_mosaic(index)
            if random.random() < MIXUP_AFTER_MOS:       # Run mixup with mosaics
                img_2, bboxes_2 = self.load_mosaic(random.randint(0, self.__len__() - 1))
                img, bboxes = mixup_augmentation(img, bboxes, img_2, bboxes_2)
        elif random.random() < MIXUP_PROB:                                   # Just run simple augmentations
            # Choose another index:
            idx_2 = index
            while idx_2 == index:
                idx_2 = random.randint(0, self.__len__() - 1)
            img_1, bboxes_1 = self.load_img_and_bboxes(index)   
            img_2, bboxes_2 = self.load_img_and_bboxes(idx_2)   
            img, bboxes = mixup_augmentation(img_1, bboxes_1, img_2, bboxes_2)
        else:
            img, bboxes = self.load_img_and_bboxes(index) 

        bboxes = self.utils.xyxy_to_xywh(bboxes)    # Convert to XYWH, since transforms use different verino of XYWH and we'll do all calculations in XYXY

        # print("Generated bboxes: ", bboxes)
        # print(img[:, 2:4, 5:7])

        if self.transform:
            img = self.transform(img)

        if self.target == 'yolo':
            return self._build_deteciton_target_yolov4(img, bboxes)
        else:
            raise NotImplementedError("Only works for yolo for now")


    def _build_deteciton_target_yolov4(self, img, bboxes):

        label = [torch.zeros(self.n_anchors_scale, Sy, Sx, self.C + 5) for Sy, Sx in self.S]  # array with n_anchors_scale (=3) tensors for each scale 

        height, width = H, W                                            # Get the shape of the input image
        
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

        return img, label, None

