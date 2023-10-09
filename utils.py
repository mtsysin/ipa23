import torch
import numpy as np

class DetectionUtils:
    '''
    Author: Pume Tuchinda
    '''
    def xyxy_to_xywh(self, bbox):
        '''
        Converts bounding box of format x1, y1, x2, y2 to x, y, w, h
        Args:
            bbox: bounding box with format x1, y1, x2, y2
        Return:
            bbox_: bounding box with format x, y, w, h if norm is False else the coordinates are normalized to the height and width of the image
        '''
        bbox_ = bbox.clone() if isinstance(bbox, torch.Tensor) else np.copy(bbox)
        bbox_[0] = (bbox[0] + bbox[2]) / 2
        bbox_[1] = (bbox[1] + bbox[3]) / 2
        bbox_[2] = bbox[2] - bbox[0]
        bbox_[3] = bbox[3] - bbox[1]

        return bbox_

    def xywh_to_xyxy(self, bbox):
        '''
        Converts bounding box of format x, y, w, h to x1, y1, x2, y2
        Args:
            bbox: bounding box with format x, y, w, h
        Return:
            bbox_: bounding box with format x1, y2, x2, y2
        '''
        bbox_ = bbox.clone() if isinstance(bbox, torch.Tensor) else np.copy(bbox)
        bbox_[:, 0] = (bbox[:, 0] - bbox[:, 2] / 2) 
        bbox_[:, 1] = (bbox[:, 1] - bbox[:, 3] / 2)
        bbox_[:, 2] = (bbox[:, 0] + bbox[:, 2] / 2) 
        bbox_[:, 3] = (bbox[:, 1] + bbox[:, 3] / 2)

        return bbox_
    
class Reduce_255():
    def __init__(self) -> None:
        pass
    def __call__(self, img):
        return img/255.