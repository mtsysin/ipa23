import torch
from torch import Tensor
from typing import Optional, List
import torchvision

"""
Nested Tensor class
Source: facebookresearch/detr
"""
class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)
    
"""The main idea behing nested tensors is to store images of different side otgether
Essentially, we can store a list of tensors which are all """

def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    '''
    This funciton is a building block of the batch collator. It pads the images 
    with zeros to the maximum shape determined by the funciotn _max_by_axis()
    The output is a nested tensor with shape (batch, c, max_x, max_y) and a mask of 
    size (batch, max_x, max_y) 
    '''
    if tensor_list[0].ndim == 3:
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)

def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

def collate_fn(batch):
    '''
    Collator funciotn for the sampler. We cannot directly generate a batch of images,
    Thus we us a custom collator that create a nested tenor with padded images stacked together
    as well as the masks of the same size which tell us which portion of the image should be ignored
    '''
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)