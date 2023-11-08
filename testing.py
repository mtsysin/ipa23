from loss import BipartiteMatchingLoss
import torch
import numpy as np
from loss import box_iou
import random

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


if __name__== "__main__":
    test_matching()
    test_simple()