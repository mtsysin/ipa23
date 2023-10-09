'''
YOLOP Model Implementation using YOLOv4 for Object Detection and U-Net for Lane/Drivable Area Segmentation
- Author: William Stevens (Revised original YOLOv3 model.py to work for YOLOv4 optimizations)
- Credit: Pume Tuchinda and Mingyu Kim
Inspired by the official YOLOP, YOLOv4, and U-Net papers:
(https://arxiv.org/pdf/2108.11250.pdf, https://arxiv.org/abs/2004.10934v1, https://arxiv.org/pdf/1505.04597.pdf)
'''

import torch.nn as nn
from typing import List

from .PANet import ConvBlock, PANet
from .CSPDarknet import CSPDarknet

class DetectionHead(nn.Module):
    '''
    YOLOv4 Detection Head
    Author: Pume Tuchinda, Mingyu Kim
    '''
    def __init__(self, num_classes, num_anchors):
        '''
        Define instance variables and the three decoder convolutions
        '''
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_output = 5 + num_classes

        self.dec_n5 = nn.Conv2d(in_channels=1024, out_channels=num_anchors*self.num_output, kernel_size=1)
        self.dec_n4 = nn.Conv2d(in_channels=512, out_channels=num_anchors*self.num_output, kernel_size=1)
        self.dec_n3 = nn.Conv2d(in_channels=256, out_channels=num_anchors*self.num_output, kernel_size=1)

    def forward(self, n3, n4, n5):
        '''
        Decode each stage's detection to produce 3 detections for each stage
        '''
        bs, _, j5, i5 = n5.shape
        d5 = self.dec_n5(n5).view(bs, self.num_anchors, j5, i5, self.num_output)

        bs, _, j4, i4 = n4.shape
        d4 = self.dec_n4(n4).view(bs, self.num_anchors, j4, i4, self.num_output)

        bs, _, j3, i3 = n3.shape
        d3 = self.dec_n3(n3).view(bs, self.num_anchors, j3, i3, self.num_output)

        return [d3, d4, d5]
        


class SegmentationHead(nn.Module):
    '''
    U-Net Combined Segmentation Head for Lane Segmentation and Drivable Area Segmentation
    Author: William Stevens
    Credit: Pume Tuchinda
    '''
    def __init__(self, num_classes):
        '''
        Initialize SegmentationHead object: Combined U-Net Implementation for Lane Segmentation and Drivable Area Segmentation
        Define function calls to be used in forward pass:
        - Define upsampling operation (Mode: Bilinear)
        - Define F3: 3x3 convolution of F4
        - Define F2: 3x3 convolution of F3
        - Define F1: 3x3 convolution of F2
        - Define F0: 3x3 convolution of F1 (output channels will be number of classes)
        '''
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv_f3 = ConvBlock(in_channels=256, out_channels=128)
        self.conv_f2 = ConvBlock(in_channels=128, out_channels=64)
        self.conv_f1 = ConvBlock(in_channels=64, out_channels=32)
        self.conv_f0 = ConvBlock(in_channels=32, out_channels=num_classes)


    def forward(self, n3):
        '''
        Forward pass:
        Use function calls defined in object initialization to carry out desired forward pass:
        - 3x3 convolution of inputted N3 to get F3
        - 2x2 Up-sampling operation of F3
        - 3x3 convolution of F3 to get F2
        - 2x2 Up-sampling operation of F2
        - 3x3 convolution of F2 to get F1
        - 2x2 Up-sampling operation of F1
        - 3x3 convolution of F1 to get F0

        Return F0 as the segmentation maps for all segmentation classes
        '''
        f3 = self.conv_f3(n3)
        f3 = self.upsample(f3)

        f2 = self.conv_f2(f3)
        f2 = self.upsample(f2)

        f1 = self.conv_f1(f2)
        f1 = self.upsample(f1)

        f0 = self.conv_f0(f1)

        return f0

class YoloMulti(nn.Module):
    '''
    YoloMulti Model Implementation to achieve Multi-Task Object Detection and Lane/Drivable Area Segmentation
    (YOLOv4 for Object Detection and U-Net for Lane/Drivable Segmentation)
    - Author: William Steven
    - Credit: Pume Tuchinda
    '''
    def __init__(self, obj_classes=13, lane_classes=9, drivable_classes=3, num_anchors=3):
        '''
        Initialize YoloMulti object: Multi-Task Object Detection and Lane/Drivable Area Segmentation
        Define function calls to be used in forward pass:
        - Define segmenation classes (lane classes + drivable area classes)

        - Define feature extractor: CSPDarknet
        - Define feature aggregator: PANet
        - Define detection head: classes = obj classes, num anchors = 3
        - Define segmentation head: classes = seg classes (lane classes + drivable area classes)
        '''
        super().__init__()
        self.seg_classes = lane_classes + drivable_classes

        self.extractor = CSPDarknet()
        self.aggregator = PANet()
        self.det_head = DetectionHead(num_classes=obj_classes, num_anchors=num_anchors)
        self.seg_head = SegmentationHead(num_classes=self.seg_classes)

    def forward(self, x):
        '''
        Forward pass:
        Use function calls defined in object initialization to carry out desired forward pass:
        - Get c1, c2, c3, c4, c5 from feature extractor CSPDarknet (input x)
        - Get n3, n4, n5 from feature aggregator PANet (input c3, c4, c5)
        - Get segmentation from seg_head (input n3)
        - Get detection from det_head (input n3, n4, n5)

        Return detection and segmentation maps for input image x
        '''
        _, _, c3, c4, c5 = self.extractor(x)
        n3, n4, n5 = self.aggregator(c3, c4, c5)
        seg = self.seg_head(n3)
        det = self.det_head(n3, n4, n5)

        return (det, seg)
