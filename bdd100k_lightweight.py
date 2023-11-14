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
BBOX_SIZE_THRESHOLD = None         # Threshold for object area
MOSAIC_PROB = 1.0
AUGMENT_PROB = 1.0
MIXUP_AFTER_MOS = 0
MIXUP_PROB = 1.0


class BDD100k(data.DataLoader):
    def __init__(self, root: Path, image_set = "train"):
        self.root = root 
        self.image_set = image_set
        self.detect = pd.read_json(self.root / 'labels/det_20/det_train.json') if self.image_set == "train" else pd.read_json(self.root / 'labels/det_20/det_val.json')
        self.detect.dropna(axis=0, subset=['labels'], inplace=True)
        self.class_dict = CLASS_DICT

        self.base_transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True)
        ])
        
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
        if self.image_set == "train":
            return transforms.Compose([
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomResizedCrop(size=size, antialias=True),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ClampBoundingBoxes(),
                transforms.SanitizeBoundingBoxes(),
                transforms.RandomPhotometricDistort(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

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
        target = self.detect.iloc[index]

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

        img_path = os.path.join(self.root, f'images/100k/{self.image_set}')
        image = Image.open(os.path.join(img_path, target['name'])).convert("RGB")
        image = self.normalize_transform(image)

        annotations = target['labels']
        bboxes = []
        labels = []
        for obj in annotations:
            bboxes.append(list(obj['box2d'].values()))      # xyxy bbox
            labels.append(self.class_dict[obj['category']]) # category

        res = self._get_augmentatiion_scheme(image.shape[-2:])({
            "image": image,
            "boxes": tv_tensors.BoundingBoxes(
                torch.tensor(bboxes), 
                format='XYXY', 
                canvas_size=image.shape[-2:]
            ),            
            "labels": labels
        })
        return res
