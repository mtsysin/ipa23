import os

import pandas as pd

import torch
import torch.utils.data as data
from torchvision.io import read_image
from PIL import Image
from pathlib import Path

from torchvision.transforms import v2 as transforms
from torchvision import tv_tensors
from augmentation import mosaic_augmentation, cls_bbox_augment, mixup_augmentation
import random

CLASS_DICT = {
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

class BDD100k_DETR(data.DataLoader):
    def __init__(self, args, image_set = "train"):
        self.root = Path(args.root) 
        self.image_set = image_set
        self.detect = pd.read_json(self.root / f'labels/det_20/det_{image_set}.json')
        self.detect.dropna(axis=0, subset=['labels'], inplace=True)
        self.class_dict = CLASS_DICT
        self.bbox_size_threshold = args.bbox_size_threshold
        self.mosaic_prob = args.mosaic_prob
        self.augment_prob = args.augment_prob
        self.mixup_after_mosaic_prob = args.mixup_after_mosaic_prob
        self.mixup_prob = args.mixup_prob

        self.base_transform = transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.detect.index)
    
    def load_cls_bboxes(self, index, enforced_type = None):
        # Load image information (class and bounding box); convert bbox format (to x_c, y_c, w, h), keep them unscaled.
        annotations = self.detect.iloc[index]['labels']
        bboxes = []
        classes = []
        for obj in annotations:
            obj_class = self.class_dict[obj['category']]
            bbox = list(obj['box2d'].values())
            _, _, w, h = self.utils.xyxy_to_xywh(bbox) 
            if self.bbox_size_threshold and (w < self.bbox_size_threshold or h < self.bbox_size_threshold):
                continue
            bbox_tensor = torch.Tensor(bbox)
            cls_tensor = torch.Tensor([obj_class])
            bboxes.append(bbox_tensor)
            classes.append(cls_tensor)
        return (torch.stack(bboxes).to(enforced_type) if enforced_type else torch.stack(bboxes),
                torch.stack(classes).to(enforced_type) if enforced_type else torch.stack(classes))
    
    def load_img_and_bboxes(self, index, enforced_bbox_type = None):
        img = read_image(self.img_path + self.detect.iloc[index]['name'])
        bboxes, classes = self.load_cls_bboxes(index, enforced_type=enforced_bbox_type)
        
        _, height, width = img.shape

        augmentation_scheme = self._get_augmentation_scheme((height, width))

        if random.random() < self.augment_prob:
            img, bboxes, classes = cls_bbox_augment(augmentation_scheme, img, bboxes, classes)
        else:
            img, bboxes, classes = self.base_transform(img, bboxes, classes)

        return img, bboxes, classes
    
    def _get_augmentation_scheme(self, size):
        if self.image_set == "train":
            return transforms.Compose([
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomResizedCrop(size=size, antialias=True),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ClampBoundingBoxes(),
                transforms.SanitizeBoundingBoxes(),
                transforms.RandomPhotometricDistort(),
                self.base_transform
            ])
        else:
            return self.base_transform

    def load_mosaic(self, base_index):
        # get random indices to build a mosaic augmentation:
        idxs = [base_index]
        while len(idxs) < 4:
            candidate = random.randint(0, self.__len__() - 1)
            if candidate not in idxs:
                idxs.append(candidate)

        images, bboxes_l, classes_l = zip(*[list(self.load_img_and_bboxes(i)) for i in idxs])

        return mosaic_augmentation(images, bboxes_l, classes_l)

    def __getitem__(self, index):
        target = self.detect.iloc[index]

        if random.random() < self.mosaic_prob:               # Run mosaic
            img, bboxes, classes = self.load_mosaic(index)
            if random.random() < self.mixup_after_mosaic_prob:       # Run mixup with mosaics
                img_2, bboxes_2, classes_2 = self.load_mosaic(random.randint(0, self.__len__() - 1))
                img, bboxes = mixup_augmentation(img, bboxes, classes, img_2, bboxes_2, classes_2)
        elif random.random() < self.mixup_prob / (1 - self.mosaic_prob):      # Just run simple augmentations
            # Choose another index:
            idx_2 = index
            while idx_2 == index:
                idx_2 = random.randint(0, self.__len__() - 1)
            img_1, bboxes_1, classes_1 = self.load_img_and_bboxes(index)   
            img_2, bboxes_2, classes_2 = self.load_img_and_bboxes(idx_2)   
            img, bboxes = mixup_augmentation(img_1, bboxes_1, classes_1, img_2, bboxes_2, classes_2)
        else:
            img, bboxes, classes = self.load_img_and_bboxes(index)

        return img, (bboxes, classes)
