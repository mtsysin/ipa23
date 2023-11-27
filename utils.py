import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from scipy.interpolate import RectBivariateSpline
    
def xyxy_to_xywh(x):
    '''
    Convert x1, y1, x2, y2 format to xywh format
    Args:
    - x: tuple of two bounding box coordinates (top left, bottom right corners)
    '''
    if isinstance(x, tuple) or isinstance(x, list):
        x1, y1, x2, y2 = x
        return (x1 + x2) / 2,  (y1 + y2) / 2, x2-x1, y2-y1
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y

def xywh_to_xyxy(x):
    '''
    Convert x1, y1, w, h format to xyxy format
    Args:
    - x: tuple of bounding box coordinates (top left point, width, height)
    '''
    if isinstance(x, tuple) or isinstance(x, list):
        x, y, w, h = x
        w/=2
        h/=2
        return x - w,  y - h, x + w,  y + h
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
    return y

def norm_bbox(x, height, width):
    '''
    Normalize bounding box coordinates to given height and width
    Args:
    - x: tuple of bounding box coordinates (top left point, width, height)
    - height: height normalizer
    - width: width normalizer
    '''
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] /= width
    y[..., 1] /= height
    y[..., 2] /= width
    y[..., 3] /= height
    return y

def unnorm_bbox(x, height, width):
    '''
    Un-normalize bounding box coordinates away from given height and width normalizers
    Args:
    - x: tuple of bounding box coordinates (top left point, width, height)
    - height: height normalizer
    - width: width normalizer
    '''
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] *= width
    y[..., 1] *= height
    y[..., 2] *= width
    y[..., 3] *= height


def get_image_patches(img_path, patch_size=16):
    # Load the input image
    img = Image.open(img_path)

    # Convert the image to a NumPy matrix
    np_img = np.array(img)

    # Get the dimensions of the image
    height, width, _ = np_img.shape

    # Calculate the number of patches in each dimension
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size

    # Initialize an empty list to store the patches
    patches = []

    # Iterate over the image and extract patches
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            # Define the coordinates for the current patch
            h_start, h_end = i * patch_size, (i + 1) * patch_size
            w_start, w_end = j * patch_size, (j + 1) * patch_size
            
            # Extract the patch
            patch = np_img[h_start:h_end, w_start:w_end, :]
            
            # Flatten the patch to a 1D vector and append to the list
            patch_flat = patch.reshape(-1)
            patches.append(patch_flat)

    # Convert the list of patches to a NumPy array
    patches_np = np.array(patches)

    # Convert the NumPy array to a PyTorch tensor
    return torch.from_numpy(patches_np)


def bicubic_interpolation(image, scale_factor):
    # Perform bicubic spline interpolation on an RGB image.
    # Param: image; Input image (H x W x C) (C=3 always)
    # Param: scale_factor;: Scaling factor for the interpolation.
    # Returns: numpy.ndarray: Interpolated image.

    # Get the input image dimensions
    height, width, channels = image.shape

    # Create grid for the original and interpolated images
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    x_interp = np.linspace(0, width - 1, int(width * scale_factor))
    y_interp = np.linspace(0, height - 1, int(height * scale_factor))

    # Initialize an empty interpolated image
    interpolated_image = np.zeros((int(height * scale_factor), int(width * scale_factor), channels))

    # Perform bicubic spline interpolation for each channel
    for c in range(channels):
        # Create a 2D interpolation function for the current channel
        interp_func = RectBivariateSpline(y, x, image[:, :, c], kx=3, ky=3)

        # Perform interpolation
        interpolated_image[:, :, c] = interp_func(y_interp, x_interp, grid=True)

    # Clip pixel values to be in the valid range [0, 255]
    interpolated_image = np.clip(interpolated_image, 0, 255)

    # Convert the interpolated image to uint8 format
    interpolated_image = interpolated_image.astype(np.uint8)

    return interpolated_image

def box_iou(box1, box2, xyxy=True, CIoU=True, pairwise = False):
    """
    Calculates IOU fucntionality
    If pairwise, will generate a pair from [..., M, 4] and [..., N, 4] to [..., M, N]
    Args:
        box1 (tensor): bounding box 1
        box2 (tensor): bounding box 2
        xyxy (bool): is format of boudning box in x1 y1 x2 y2
        CIoU (bool): if true calculate CIoU else IoU
    Returns:
        Tensor int the format:
        [..., iou_value]
    """

    EPS = 1e-6

    if not xyxy:
        box1 = xywh_to_xyxy(box1)
        box2 = xywh_to_xyxy(box2)

    if pairwise:
        box1 = box1[..., :, None, :] # [..., M, *N, 4]
        box2 = box2[..., None, :, :] # [..., *M, N, 4]

    box1_x1 = box1[..., 0:1]
    box1_y1 = box1[..., 1:2]
    box1_x2 = box1[..., 2:3]
    box1_y2 = box1[..., 3:4]
    box2_x1 = box2[..., 0:1]
    box2_y1 = box2[..., 1:2]
    box2_x2 = box2[..., 2:3]
    box2_y2 = box2[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_width, box1_height = box1_x2 - box1_x1, box1_y2 - box1_y1
    box2_width, box2_height = box2_x2 - box2_x1, box2_y2 - box2_y1
    union = box1_width * box1_height + box2_width * box2_height - intersection
    iou = intersection / (union + EPS)

    if CIoU:
        '''
        Complete-IOU Loss Implementation
        - Inspired by the official paper on Distance-IOU Loss (https://arxiv.org/pdf/1911.08287.pdf)
        - Combines multiple factors for bounding box regression: IOU loss, distance loss, and aspect ratio loss.
        - This results in much faster convergence than traditional IOU and generalized-IOU loss functions.
        Args:
            - preds: prediction tensor containing confidence scores for each class.
            - target: ground truth containing correct class labels.
        '''
        convex_width = torch.max(box1_x2, box2_x2) - torch.min(box1_x1, box2_x1)
        convex_height = torch.max(box1_y2, box2_y2) - torch.min(box1_y1, box2_y1)
        convex_diag_sq = convex_width**2 + convex_height**2
        center_dist_sq = (box2_x1 + box2_x2 - box1_x1 - box1_x2)**2 + (box2_y1 + box2_y2 - box1_y1 - box1_y2)**2
        dist_penalty = center_dist_sq / (convex_diag_sq + EPS) / 4 

        v = (4 / (torch.pi**2)) * torch.pow(torch.atan(box2_width / (box2_height + EPS)) - torch.atan(box1_width / (box1_height + EPS)), 2)
        with torch.no_grad():
            alpha = v / ((1 + EPS) - iou + v)
        aspect_ratio_penalty = alpha * v
        
        iou = iou - dist_penalty - aspect_ratio_penalty
    
    return iou.squeeze(-1)

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res