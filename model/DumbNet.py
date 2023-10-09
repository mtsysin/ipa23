import torch.nn as nn
from typing import List
import torch


from .PANet import ConvBlock, PANet
from .CSPDarknet import CSPDarknet

"""Simple neural net for testing"""

class Mish(nn.Module):
    def __init__(self):
        '''
        Initialize Mish object
        - Obtain parent class's information, functions, and fields using super().__init__() of nn.Module
        '''
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size=3, stride=1):
        '''
        Initialize ConvBlock object
        - Obtain parent class's information, functions, and fields using super().__init__() of nn.Module
        - Define function calls using the object initialization parameters
        '''
        super(ConvBlock, self).__init__()
        padding=((filter_size-1)//2)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=filter_size, padding=padding, stride=stride)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.mish = Mish()

    def forward(self, x):
        '''
        Override of forward pass inherited from parent class nn.Module
        - Use function calls defined in object initialization to carry out desired forward pass
        '''
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.mish(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        '''
        Author: William Stevens
        Initialize ResBlock object
        - Obtain parent class's information, functions, and fields using super().__init__() of nn.Module
        - Define variables and function calls using the object initialization parameters
        Args:
            conv1x1 (tuple): 
            conv3x3 (tuple): 
        '''
        super().__init__()
        self.firstConvBlock = ConvBlock(in_ch, out_ch, filter_size=1)
        self.secondConvBlock = ConvBlock(out_ch, in_ch, filter_size=3)

    def forward(self, x):
        '''
        Override of forward pass inherited from parent class nn.Module
        - Use function calls defined in object initialization to carry out desired forward pass
        '''
        input = x
        x = self.firstConvBlock(x)
        x = self.secondConvBlock(x)
        return x + input


class DumbNet(nn.Module):
    '''
    YOLOv4 Detection Head
    Author: Pume Tuchinda, Mingyu Kim
    '''
    def __init__(self, num_classes = 13, num_anchors = 3):
        '''
        Define instance variables and the three decoder convolutions
        '''
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_output = 5 + num_classes

        self.conv1 = ConvBlock(in_channels=3, out_channels=16, filter_size=7, stride = 2)
        self.conv2 = ConvBlock(in_channels=16, out_channels=32, filter_size=3, stride = 2)
        self.conv3 = ConvBlock(in_channels=32, out_channels=64, filter_size=3, stride = 2)
        self.conv4 = ConvBlock(in_channels=64, out_channels=128, filter_size=3, stride = 2)
        self.conv5 = ConvBlock(in_channels=128, out_channels=256, filter_size=3, stride = 2)
        # self.conv6 = ConvBlock(in_channels=256, out_channels=512, filter_size=3, stride = 2)

        self.out1 = nn.Conv2d(in_channels=64, out_channels=self.num_output * self.num_anchors, kernel_size=1)
        self.out2 = nn.Conv2d(in_channels=128, out_channels=self.num_output * self.num_anchors, kernel_size=1)
        self.out3 = nn.Conv2d(in_channels=256, out_channels=self.num_output * self.num_anchors, kernel_size=1)

    def forward(self, x):
        '''
        Decode each stage's detection to produce 3 detections for each stage
        '''
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)


        out1 = self.out1(x)
        b, _, j, i = out1.shape
        out1 = out1.view(b, self.num_anchors, j, i, self.num_output)

        x = self.conv4(x)

        out2 = self.out2(x)
        b, _, j, i = out2.shape
        out2 = out2.view(b, self.num_anchors, j, i, self.num_output)

        x = self.conv5(x)

        out3 = self.out3(x)
        b, _, j, i = out3.shape
        out3 = out3.view(b, self.num_anchors, j, i, self.num_output)
        
        return (out1, out2, out3), out3
    