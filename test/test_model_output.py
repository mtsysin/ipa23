import torch
from model.YoloMulti import YoloMulti
from model.CSPDarknet import CSPDarknet, Mish
from model.DumbNet import DumbNet
import unittest
from torchinfo import summary
import matplotlib.pyplot as plt

class TestModel(unittest.TestCase):
    def test_model_ouptut(self):
        torch.cuda.empty_cache()
        device = torch.device('cuda')
        BATCH = 8
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))
        model = YoloMulti().to(device)
        print(next(model.parameters()).device)
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))

        summary(model, input_size=(BATCH, 3, 384, 640), depth=6)

        x = torch.randn(BATCH, 3, 384, 640)
        x = x.to(device)

        det, seg = model(x)
        print(f'Detection scale 1 {det[0].shape}')
        print(f'Detection scale 2 {det[1].shape}')
        print(f'Detection scale 3 {det[2].shape}')
        print(f'Lane {seg.shape}')

    def test_mish(self):
        x = torch.arange(-5, 5, 0.01)
        y = Mish()(x)
        plt.plot(x, y)
        plt.savefig("./out/test_mish.png")


    def test_dumb_net(self):
        torch.cuda.empty_cache()
        device = torch.device('cuda')
        BATCH = 24
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))
        model = DumbNet(13, 3).to(device)
        print(next(model.parameters()).device)
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))

        summary(model, input_size=(BATCH, 3, 384, 640), depth=6)

        x = torch.randn(BATCH, 3, 384, 640)
        x = x.to(device)

        det, seg = model(x)
        print(f'Detection scale 1 {det[0].shape}')
        print(f'Detection scale 2 {det[1].shape}')
        print(f'Detection scale 3 {det[2].shape}')
        # print(f'Lane {seg.shape}')