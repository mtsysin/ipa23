from loss import BipartiteMatchingLoss
import torch
import numpy as np

def test_matching():
    batch_size = 16
    num_queries = 70
    num_classes = 13

    outputs = (
        torch.rand(batch_size, num_queries, num_classes),
        "pred_boxes": torch.rand(batch_size, num_queries, 4),
    )

    targets = [{
        "labels": torch.randint(0, num_classes, [s]),
        "boxes": torch.rand(s, 4),
    } for s in [np.random.randint(40, 60) for _ in range(batch_size)]]


    hm = BipartiteMatchingLoss()

    out = hm.forward(outputs, targets)

    print(len(out))

    for el in out:
        print(el)


if __name__== "__main__":
    test_matching()