import torch
from torch import nn
from utils import DetectionUtils
from torchvision.utils import draw_bounding_boxes
from bdd100k import ANCHORS, CLASS_DICT, REVERSE_CLASS_DICT, W, H, GRID_SCALES
from evaluate import DetectionMetric
'''
Author: Pume Tuchinda
'''

detection_metric = DetectionMetric()

def draw_bbox_raw(img, dets):
    """
    Draws bboxes on an image. Accepts a single image with NO batch dimension present
    """
    
    def unnorm_bbox(bbox, width, height, i, j, sx, sy):
        '''
        Unormalize the bounding box predictions based on the yolo predictions
        '''
        bbox[:,0] = width * (bbox[:,0] + j) / sx
        bbox[:,1] = height * (bbox[:,1] + i) / sy
        bbox[:,2] = width * (bbox[:,2] / sx)
        bbox[:,3] = height * (bbox[:,3] / sy)

        return bbox

    num_to_class = REVERSE_CLASS_DICT

    utils = DetectionUtils()

    img = img.to(torch.uint8)
    _, height, width = img.shape

    for det in dets:
        scale_pred = det[0]
        sy, sx = scale_pred.shape[1], scale_pred.shape[2]
        for i in range(sy):
            for j in range(sx):
                conf = scale_pred[:, i, j, 13]
                if conf.any() == 1.:
                    scale = torch.argmax(conf)
                    pred = scale_pred[scale]
                    _, pclass = torch.max(pred[..., :13], dim=2)
                    class_pred = num_to_class[pclass[i, j].item()]
                    label = [class_pred]
                    bbox = pred[i, j, 14:19].clone().unsqueeze(0)
                    bbox = unnorm_bbox(bbox, width, height, i ,j, sx, sy)
                    bbox = utils.xywh_to_xyxy(bbox)
                    img = draw_bounding_boxes(img, bbox, width=3, labels=label, colors=(100, 250, 150)) 

    return img

def draw_bbox(imgs, dets):
    """
    Draws bboxes on a batch of images
    Inout batch of images and batch of detected bboxes
    """

    num_to_class = REVERSE_CLASS_DICT

    utils = DetectionUtils()

    imgs = imgs.to(torch.uint8)

    for i, (img, det) in enumerate(zip(imgs, dets)):
        labels = [num_to_class[index.item()] for index in det[..., 0]]
        bboxes = utils.xywh_to_xyxy(det[..., 2:])
        imgs[i, ...] = draw_bounding_boxes(img, bboxes, width=3, labels=labels, colors=(100, 250, 150)) 

    return imgs

def nms(bboxes_batch, iou_threshold, conf_threshold):
    bboxes_batch_after_nms = []
    for bboxes in bboxes_batch:

        mask = bboxes[:, 1] >= conf_threshold
        bboxes = bboxes[mask]
        # print("shape of bboxes after confidence threshold", bboxes.shape)

        # Sort the boxes by their confidence scores in descending order
        confidences = bboxes[:, 1]
        indices = torch.argsort(confidences, descending=True)
        bboxes = bboxes[indices]

        bboxes_after_nms = []

        while len(bboxes) > 0:                                                   # While we still have bboxes
            # print("shape of bboxes in nms", bboxes.shape)
            chosen_box = bboxes[0]

            mask = detection_metric.box_iou(chosen_box[None ,2:], bboxes[... ,2:], xyxy=False).squeeze(-1) < iou_threshold
            bboxes = bboxes[mask]

            bboxes_after_nms.append(chosen_box)

        if bboxes_after_nms:
            bboxes_after_nms = torch.stack(bboxes_after_nms)
        else:
            raise ValueError("No bounding boxes detected!!!")
        bboxes_batch_after_nms.append(bboxes_after_nms)

    return bboxes_batch_after_nms

def process_prediction(prediction, img_height, img_width, anchor, C=13, true_prediction=True):
    """
    Takes a batch of predictions and converts them to a tensor of all bounding boxes with shape
    (batch, # of bounding boxes, 6 [class, confidence, x, y, w, h])
    true_prediction -- helper parameter allowing to apply the processing function to already scaled predictions (for example can apply to true labels)
    """
    batch_size, n_anchors, gy, gx, n_outputs = prediction.shape                             # Get dimensions of prediction
    # prediction = prediction[0].clone().detach().cpu()                                       # Use predictions for the first batch
    gridy, gridx = torch.meshgrid([torch.arange(gy), torch.arange(gx)], indexing='ij')

    if true_prediction:
        confidence = prediction[..., C].sigmoid()                                               # clamp the confidence between 0 and 1 so we can represent it as percentage
    else:
        confidence = prediction[..., C]

    # something = prediction[..., C+3:C+5] > 1.069e+05
    something = confidence > 0.8
    print("SOMETHING_PREDICTION", prediction[something].shape, prediction[something][..., C+3:C+5])

    if true_prediction:
        bbox_detection = torch.cat(                                                             # Convert predictions to the format where everything is represented as a fraction of a cell size
            (
                prediction[..., C+1:C+3].sigmoid(),
                (prediction[..., C+3:C+5] + torch.log(anchor)).exp()
            ),
            dim = -1
        )
        # prediction[..., C+1:C+3] = prediction[..., C+1:C+3].sigmoid()                           
        # prediction[..., C+3:C+5] = prediction[..., C+3:C+5].exp() * anchor
    else:
        bbox_detection = prediction[..., C+1:C+5]


    predicted_class = torch.argmax(prediction[..., :C], dim=-1)
    x = (bbox_detection[..., 0] + gridx) / gx * img_width                                     # Convert x and y to acual pixel values on image
    y = (bbox_detection[..., 1] + gridy) / gy * img_height
    width = bbox_detection[..., 2] / gx * img_width                                           # Do the same for w and h
    height = bbox_detection[..., 3] / gy * img_height
    detection = torch.stack((predicted_class, confidence, x, y, width, height), dim=-1).view(batch_size, -1, 6)

    return detection   
    
def get_bboxes(predictions, iou_threshold, conf_threshold, true_prediction=True):
    """
    Takes predicted values (3 scales with each scale batched)
    Outputs list of tensors: for each batch output a tensor of accepted bounding boxes
    """
    # Prepare anchors:
    if true_prediction:
        anchors = torch.tensor(ANCHORS)
        # print(anchors)
        grid_scales = torch.tensor(GRID_SCALES)
        # print(grid_scales)
        size = torch.tensor([H, W])
        # print(size)
        cell_sizes = (size / grid_scales).repeat_interleave(3, dim=0).view(3, 3, 2)
        # print(cell_sizes)
        anchors = anchors / cell_sizes
        print(anchors)
    else:
        anchors = ANCHORS

    detections = []
    for i, prediction in enumerate(predictions):                                                # For all prediction scales: i - number of scale, prediction - prediction for the corresponding scale
        anchor = torch.tensor(anchors[i]).view(1, 3, 1, 1, 2)                                   # Get the anchor boxes corresponding to the chosen scale, view: (Batch, anchor index, sy, sx, box dimension)
        print(f"anchor {i}", anchor)
        detections.append(process_prediction(prediction, 384, 640, anchor, true_prediction=true_prediction))    # Process detection
    detection = torch.cat(tuple(detections), dim=1)
    print("detection.shape", detection.shape)
    something = detection[..., 4] > 1.069e+05
    print("SOMETHING", something, something.sum())
    print(detection[0][2:7, ...])
    nms_b = nms(detection, iou_threshold, conf_threshold)                            # Perform non-maximum suppression on the resulting list of bounding boxes

    return nms_b