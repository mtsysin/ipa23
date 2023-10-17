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
from evaluate import DetectionMetric
import random
import torchvision.transforms.v2 as transforms
from torchvision import tv_tensors


def cls_bbox_augment(augmentation, image, bboxes, format = "xyxy"):
    res = augmentation({
        "image": image,
        "boxes": tv_tensors.BoundingBoxes(
            bboxes[..., 1:5], 
            format=format.upper(), 
            canvas_size=image.shape[-2:]
        ),            
        "labels": bboxes[..., 0:1]
    })
    return res["image"], torch.cat((res["labels"], res["boxes"]), dim=-1)


def mixup_augmentation(im1, labels1, im2, labels2, ratio = 32.0):
    """
    implements mixup augmentation by adding two images with a coefficient lambda
    Labels are just stacked togeter
    """
    lamd = np.random.beta(ratio, ratio)  # mixup ratio, see paper https://arxiv.org/pdf/1710.09412.pdf
    im = (im1 * lamd + im2 * (1 - lamd)).to(torch.uint8)
    labels = torch.cat((labels1, labels2), -2)
    return im, labels


def mosaic_augmentation(images, bboxes_l, bbox_format = 'xyxy', center_point_coef = 2/5):
    """Accepts 4 images and creates a mosaic augmented image for it"""
    assert (len(images) == 4) and (len(bboxes_l) == 4)
    for image in images[1:]:
        assert image.shape == images[0].shape
    
    _, h, w = images[0].shape

    # Choose a point in the middle to base our tiles on. top left image excludes this point.
    
    center_y, center_x = np.random.randint(int(h * center_point_coef), int(h * (1-center_point_coef))), np.random.randint(int(w * center_point_coef), int(w * (1-center_point_coef)))
    
    def transform_builder_(center_y, center_x):
        return transforms.Compose([
            transforms.Resize((center_y, center_x), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.ClampBoundingBoxes(),
            transforms.SanitizeBoundingBoxes(min_size=10)
        ])

    # Resize images accordingly
    image_top_left, bboxes_top_left = cls_bbox_augment(transform_builder_(center_y, center_x), images[0], bboxes_l[0])
    image_top_right, bboxes_top_right = cls_bbox_augment(transform_builder_(center_y, w - center_x), images[1], bboxes_l[1])
    image_bottom_right, bboxes_bottom_right = cls_bbox_augment(transform_builder_(h - center_y, w - center_x), images[2], bboxes_l[2])
    image_bottom_left, bboxes_bottom_left = cls_bbox_augment(transform_builder_(h - center_y, center_x), images[3], bboxes_l[3])

    # Format bboxes accordingly
    def build_shifter(shift_y, shift_x):
        if bbox_format == 'xywh':
            s = [0, shift_x, shift_y, 0, 0]
        else:
            s = [0, shift_x, shift_y, shift_x, shift_y]
        return torch.tensor(s).view(1, 5)
    
    bboxes_top_left += build_shifter(0, 0)
    bboxes_top_right += build_shifter(0, center_x)
    bboxes_bottom_left += build_shifter(center_y, 0)
    bboxes_bottom_right += build_shifter(center_y, center_x)

    image_mosaic = torch.cat(
        (torch.cat((image_top_left, image_top_right), -1), torch.cat((image_bottom_left, image_bottom_right), -1)),
        dim= -2
    )
    bboxes_mosaic = torch.cat([bboxes_top_left, bboxes_top_right, bboxes_bottom_right, bboxes_bottom_left], dim = -2)

    return image_mosaic, bboxes_mosaic


def cutmix_augmentation(image, bboxes, image_2, bboxes_2, ratio = 32.0):
    assert image.shape == image_2.shape
    h, w = image.shape[-2:]
    lamd = np.random.beta(ratio, ratio)  # See paper for box generation
    rx, ry = np.random.uniform(0, 1, 2) * np.array([w, h])
    rw, rh = np.array([rx, ry]) * np.sqrt(1 - lamd)

    bbox_xyxy = np.clip([rx - rw // 2, ry - rh // 2, rx + rw // 2, ry + rh // 2], [0, 0, 0, 0], [w, h, w, h])
    x1, y1, x2, y2 = bbox_xyxy

    image[:, x1:x2, y1:y2] = image_2[:, x1:x2, y1:y2]


def cutout_augmentation(image, bboxes, scales = None):
    """Make cutouts"""
    _, h, w = image.shape

    # Choose a set of scales on which to apply cutout
    if scales is None:
        scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  

    for s in scales:
        mask_h = random.randint(1, int(h * s))  # create random masks
        mask_w = random.randint(1, int(w * s))

        # box
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)

        # apply random color mask
        image[:, ymin:ymax, xmin:xmax] = torch.randint(64, 191, (3, 1, 1))

        # return unobscured labels
        if len(bboxes) and s > 0.03:
            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            ioa = DetectionMetric.box_iou(box, DetectionUtils.xywh_to_xyxy(bboxes[:, 1:5]))  # intersection over area
            bboxes = bboxes[ioa < 0.60]  # remove >60% obscured labels

    