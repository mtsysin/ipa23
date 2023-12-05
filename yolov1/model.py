import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
# in channels, out channels, kernel, stride, padding
yolo_structure = iter([(3, 64, 7, 2, 3),
                  "pool",
                   (64, 192, 3, 1, 1),
                 "pool",
                 (192,128, 1, 1, 0),
                 (128,256, 3, 1, 1),
                 (256,256, 1, 1, 0),
                 (256,512, 3, 1, 1),
                 "pool",
#                   repeat 4 times
                 (512, 256, 1, 1, 0),
                 (256, 512,3,1,1),
#                   
                 (512, 512, 1, 1,0),
                 (512, 1024, 3, 1, 1),
                 "pool",
#                  repeat twice
                 (1024, 512, 1, 1, 0),
                 (512, 1024, 3, 1, 1),
#                   
                 (1024, 1024, 3, 1, 1),
                 (1024, 1024, 3, 2, 1),
                 (1024, 1024, 3, 1, 1),
                 (1024, 1024, 3, 1, 1)
                 ])

class YoloV1(nn.Module):
    def __init__(self, split_grids=7, num_gridbboxes=2, num_classes=20):
        super(YoloV1, self).__init__();
        self.architecture = nn.Sequential();
        self.S = split_grids;
        self.B = num_gridbboxes;
        self.C = num_classes;
#         create architecture
        pool_counter = 0;
        for element in yolo_structure:
            if(element == "pool"):
                self.architecture.append(nn.MaxPool2d(kernel_size=2, stride=2));
                pool_counter += 1;
            else:
                if(pool_counter == 3):
#                     repeat 4 times
                    next_element = next(yolo_structure);
                    for i in range(0, 4):
                        self.architecture.append(nn.Conv2d(*element));
                        self.architecture.append(nn.LeakyReLU(0.1));
                        self.architecture.append(nn.Conv2d(*next_element));
                        self.architecture.append(nn.LeakyReLU(0.1));
                    pool_counter += 1;
                elif(pool_counter == 5):
#                     repeat 2 times
                    next_element = next(yolo_structure);
                    for i in range(0, 2):
                        self.architecture.append(nn.Conv2d(*element));
                        self.architecture.append(nn.LeakyReLU(0.1));
                        self.architecture.append(nn.Conv2d(*next_element));
                        self.architecture.append(nn.LeakyReLU(0.1));
                    pool_counter += 1;
                else:
                    self.architecture.append(nn.Conv2d(*element));
                    self.architecture.append(nn.LeakyReLU(0.1));
        # fully connected layer
        # 7x7x1024 --> 4096
        self.architecture.append(nn.Flatten(start_dim=1))
        self.architecture.append(nn.Linear(7*7*1024, 4096));
        self.architecture.append(nn.LeakyReLU(0.1));
        # 4096 --> 7x7x30 (S X S (5*B + C))
        self.architecture.append(nn.Linear(4096, (5*self.B + self.C)*self.S**2));
        # self.architecture.append(nn.Conv2d(1024, (5*self.B + self.C), 1, 1,0))
    def forward(self, x):
        x = self.architecture(x).view(-1,self.S, self.S, 5*self.B + self.C);
        return x;
