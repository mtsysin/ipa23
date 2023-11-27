from loss import BipartiteMatchingLoss
import torch
import torch.nn as nn
import numpy as np
from loss import box_iou
import random
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--test', '-t', default='none', type=str)
    return parser


def test_matching():
    batch_size = 16
    num_queries = 70
    num_classes = 13

    targets = []
    preds = []
    for _ in range(batch_size):
        s = random.randint(5, 10)
        targets.append((
            torch.randint(0, num_classes, (s, 1)),
            torch.rand(s, 4),
        ))
    preds = (
        torch.rand(batch_size, num_queries, num_classes), # logits
        torch.rand(batch_size, num_queries, 4), # bbox
    )

    hm = BipartiteMatchingLoss()
    out = hm.bipartite_matching(preds, targets)

    print(len(out))

    for el in out:
        print(el)

def test_simple():
    a = torch.rand(3, 5, 4)
    print(a)
    idx = torch.tensor([3, 2, 1, 4,0])
    print(a[:, idx, :])

def test_iou():
    B = 16
    M, N  = 5, 7
    b1 = torch.rand(M, 4)
    b2 = torch.rand(N, 4)

    print("b1, b2 shape ", b1.shape, b2.shape)
    out = box_iou(b1, b2, pairwise=True)
    print("out.shape: ", out.shape)


def test_pos_encoding():
    from misc import NestedTensor
    from model.positional_embedding import PositionEmbeddingSine, PositionEmbeddingLearned
    B = 16
    H, W,  = 720, 1080
    src = torch.rand(B, 3, H, W)
    mask = torch.rand(B, H, W) > 0.5 ## Make some boolean array
    x = NestedTensor(src, mask)

    embedding_sine = PositionEmbeddingSine(
        num_pos_feats=512/2, 
        temperature=10000, 
        normalize=False, 
        scale=None
    )

    pos = embedding_sine(x)

    print("Output of the positional encoding layer: ", pos.shape)

def test_transformer():
    from model.model import Transformer
    from model.positional_embedding import PositionEmbeddingSine

    B = 16
    
    d_model = 512
    transformer = Transformer(
        d_model=d_model,
        dropout=0.1,
        nhead=8,
        dim_feedforward=2048,
        num_encoder_layers=6,
        num_decoder_layers=6,
        normalize_before=False,
        return_intermediate_dec=True,
    )
    H, W,  = 17, 23
    num_queries = 90
    hidden_dim = d_model
    src = torch.rand(B, hidden_dim, H, W)
    mask = torch.rand(B, H, W)
    query_embed = nn.Embedding(num_queries, hidden_dim)
    pos_embed = torch.rand(B, hidden_dim, H, W)
    
    hs = transformer(src, mask, query_embed.weight, pos_embed)[0]

    print("Output of the transformer (Should be\
          num_intermediate, batch, num_queries, d_model): ", hs.shape)
    
def test_backbone():
    from model.feature_extractor import build_backbone
    from misc import NestedTensor

    B = 16
    H, W = 720, 1080
    args = argparse.Namespace()
    args.lr_backbone = 0.001
    args.masks = False
    args.dilation = False
    args.backbone = 'resnet50'
    args.hidden_dim = 512
    args.position_embedding = 'sine'

    backbone = build_backbone(args)

    src = torch.rand(B, 3, H, W)
    mask = torch.rand(B, H, W) > 0.5 ## Make some boolean array
    x = NestedTensor(src, mask)

    out = backbone(x)

    print("Shapes of the intermediate outputs and positional encodings is equal to ", [o.tensors.shape for o in out[0]], 
          [o.mask.shape for o in out[0]],  [o.shape for o in out[1]])

def test_detr():
    from model.model import build_detr
    from misc import NestedTensor

    B = 16
    H, W = 720, 1080
    args = argparse.Namespace()
    args.lr_backbone = 0.001
    args.masks = False
    args.dilation = False
    args.backbone = 'resnet50'
    args.hidden_dim = 512
    args.position_embedding = 'sine'
    args.num_classes = 13
    args.num_queries = 90
    args.aux_loss = False
    args.pre_norm = False
    args.enc_layers = 6
    args.dec_layers = 6
    args.dim_feedforward = 2048
    args.dropout = 0.1
    args.nheads = 8

    detr = build_detr(args)

    src = torch.rand(B, 3, H, W)
    mask = torch.rand(B, H, W) > 0.5 ## Make some boolean array
    x = NestedTensor(src, mask)

    out = detr(x)
    print("Shapes of the bboxes and logits of the DETR are equal to ", out["pred_logits"].shape, out['pred_boxes'].shape)

def test_loss():
    from loss import BipartiteMatchingLoss

    # Create loss
    num_classes = 13
    loss = BipartiteMatchingLoss(num_classes)
    batch_size = 16
    num_queries = 70
    num_classes = 13

    targets = []
    for _ in range(batch_size):
        s = random.randint(5, 10)
        targets.append((
            torch.randint(0, num_classes, (s, 1)),
            torch.rand(s, 4),
        ))
    preds = (
        torch.rand(batch_size, num_queries, num_classes+1), # logits
        torch.rand(batch_size, num_queries, 4), # bbox
    )

    out = loss(preds, targets)

    print("Output of the loss: ", out)

if __name__== "__main__":
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    locals()[f"test_{args.test}"]()
