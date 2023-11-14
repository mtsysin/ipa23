import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from scipy.interpolate import RectBivariateSpline
import torch.distributed as dist

def bbox_iou(pred:torch.Tensor, target: torch.Tensor, format:str = 'xyxy'):
    '''
    Caculate IOU between 2 boxes
    Params::
    - pred: predictions tensor
    - target: target tensor
    - format: xyxy or xywh
    '''
    assert pred.shape[0] == target.shape[0]
    pred = pred.view(-1, 4)
    target = target.view(-1, 4)
    
def xyxy_to_xywh(x):
    '''
    Convert x1, y1, x2, y2 format to xywh format
    Args:
    - x: tuple of two bounding box coordinates (top left, bottom right corners)
    '''
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y

def xywh_to_xyxy(x):
    '''
    Convert x1, y1, w, h format to xyxt format
    Args:
    - x: tuple of bounding box coordinates (top left point, width, height)
    '''
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


def scaled_dotprod_attention(Q, K, V):
    # Assert query and key matrices have same dimensionality
    assert(Q.size(), K.size())

    # Compute dot product of query matrix and transposition of key matrix
    QK_T = torch.dot(Q, torch.transpose(K, 0, 1))
    d_k = K.size()

    # Divide dot product of Q and K^T with the square root of their dimension
    # to factor down dot products and stabilize training 
    dotprod_matrix = torch.div(QK_T, np.sqrt(d_k))

    # Compute probability distribution of dot products using softmax function
    softMax = nn.SoftMax(dim=1)
    attention_dist = softMax(dotprod_matrix)

    # Compute weighted sum of attentions by taking product of attention distribution with values vector
    attention_weighted_sum = torch.dot(attention_dist, V)
    return attention_weighted_sum

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

from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0
