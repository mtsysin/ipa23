# Object Detection with DETR - a minimal implementation

from PIL import Image
import requests
import matplotlib.pyplot as plt

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms.v2 as T
torch.set_grad_enabled(False)
from main import get_args_parser # Path to original DETR
import argparse
from models import build_model # Path to original DETR
import numpy as np
import random

CLASSES = [
    "N/A",
    'pedestrian',
    'rider',
    'car' ,
    'truck' , 
    'bus' , 
    'train' , 
    'motorcycle',
    'bicycle',
    'traffic light' ,
    'traffic sign',
    'other vehicle',
    'trailer',
    'other person'
]

class DETRdemo(nn.Module):
    """
    Demo DETR implementation.

    Demo implementation of DETR in minimal number of lines, with the
    following differences wrt DETR in the paper:
    * learned positional encoding (instead of sine)
    * positional encoding is passed at input (instead of attention)
    * fc bbox predictor (instead of MLP)
    The model achieves ~40 AP on COCO val5k and runs at ~28 FPS on Tesla V100.
    Only batch size 1 supported.
    """
    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = resnet50()
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)

        # construct positional encodings
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        # propagate through the transformer
        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1)).transpose(0, 1)
        
        # finally project transformer outputs to class labels and bounding boxes
        return {'pred_logits': self.linear_class(h), 
                'pred_boxes': self.linear_bbox(h).sigmoid()}

transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def eval(model, image_path, device):
    if isinstance(image_path, list):
        for im in image_path:
            eval(model, im, device)
        return

    im = Image.open(image_path)
    im = im.convert('RGB')
    print(im)

    scores, boxes = detect(im, model, transform, device)
    plot_results(im, scores, boxes, image_path.split('/')[-1])


def run(args):

    print(args)

    device = torch.device(args.device)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, _, _ = build_model(args)
    model.to(device)

    checkpoint = torch.load(args.resume, map_location='cpu')

    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()

    eval(model, [
                "/data/mtsysin/ipa/detr/out/purdue.png",
                #  "/data/mtsysin/ipa/LaneDetection_F23/bdd100k/images/100k/test/cabc9045-cd422b81.jpg"
                "/data/mtsysin/ipa/LaneDetection_F23/bdd100k/images/100k/val/b1ebfc3c-740ec84a.jpg",
                "/data/mtsysin/ipa/LaneDetection_F23/bdd100k/images/100k/val/b2a8e8b4-a4e93829.jpg",
                "/data/mtsysin/ipa/LaneDetection_F23/bdd100k/images/100k/val/b2bee3e1-80c787bd.jpg",
                "/data/mtsysin/ipa/LaneDetection_F23/bdd100k/images/100k/val/b2f4a409-80dacf25.jpg",

                "/data/mtsysin/ipa/LaneDetection_F23/bdd100k/images/100k/train/b05dc32c-3bec8c3e.jpg",
                "/data/mtsysin/ipa/LaneDetection_F23/bdd100k/images/100k/train/aff7a87c-2384f2ae.jpg",
                "/data/mtsysin/ipa/LaneDetection_F23/bdd100k/images/100k/train/afcfe157-d1708127.jpg",
                "/data/mtsysin/ipa/LaneDetection_F23/bdd100k/images/100k/train/afae225d-25e1c655.jpg",
                "/data/mtsysin/ipa/LaneDetection_F23/bdd100k/images/100k/train/af8863c7-86a30311.jpg",


                 ], device)


# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    out_bbox = bbox_adjust(out_bbox)
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device="cuda:0")
    return b

def detect(im, model, transform, device, keep_conf = 0.98):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0).to(device)

    assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'model only supports images up to 1600 pixels on each side'

    # Run the model
    outputs = model(img)

    # keep only predictions with certain confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > keep_conf

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    return probas[keep], bboxes_scaled

def bbox_adjust(boxes: torch.Tensor, matrix: torch.Tensor = torch.tensor(
        [
            [1, 0, 0.05, 0],
            [0, 1, 0, 0],
            [0, 0, .9, 0],
            [0, 0, 0, 1]
        ]
)):
    # x y w h
    matrix = matrix.to(boxes.device)
    return torch.einsum('bi,xi->bx', boxes, matrix)

def plot_results(pil_img, prob, boxes, file_name, show_confidence = False):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f"{CLASSES[cl]}" + (f": {p[cl]:0.2f}" if show_confidence else "")
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.savefig(f'/data/mtsysin/ipa/detr/out/{file_name}.png', bbox_inches='tight')
    

if __name__=="__main__":
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    run(args)