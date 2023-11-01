"""A simple tool to convert bdd labels to coco format for the use in pre-trained models

BDD Model:

- name: string
- url: string
- videoName: string (optional)
- attributes:
    - weather: "rainy|snowy|clear|overcast|undefined|partly cloudy|foggy"
    - scene: "tunnel|residential|parking lot|undefined|city street|gas stations|highway|"
    - timeofday: "daytime|night|dawn/dusk|undefined"
- intrinsics
    - focal: [x, y]
    - center: [x, y]
    - nearClip:
- extrinsics
    - location
    - rotation
- timestamp: int64 (epoch time ms)
- frameIndex: int (optional, frame index in this video)
- labels [ ]:
    - id: int32
    - category: string (classification)
    - manualShape: boolean (whether the shape of the label is created or modified manually)
    - manualAttributes: boolean (whether the attribute of the label is created or modified manually)
    - score: float (the confidence or some other ways of measuring the quality of the label.)
    - attributes:
        - occluded: boolean
        - truncated: boolean
        - trafficLightColor: "red|green|yellow|none"
        - areaType: "direct | alternative" (for driving area)
        - laneDirection: "parallel|vertical" (for lanes)
        - laneStyle: "solid | dashed" (for lanes)
        - laneTypes: (for lanes)
    - box2d:
       - x1: float
       - y1: float
       - x2: float
       - y2: float
   - box3d:
       - alpha: (observation angle if there is a 2D view)
       - orientation: (3D orientation of the bounding box, used for 3D point cloud annotation)
       - location: (3D point, x, y, z, center of the box)
       - dimension: (3D point, height, width, length)
   - poly2d: an array of objects, with the structure
       - vertices: [][]float (list of 2-tuples [x, y])
       - types: string (each character corresponds to the type of the vertex with the same index in vertices. ‘L’ for vertex and ‘C’ for control point of a bezier curve.
       - closed: boolean (closed for polygon and otherwise for path)


COCO model:

{
    "info": {
        "description": "COCO 2017 Dataset",
        "url": "http://cocodataset.org",
        "version": "1.0",
        "year": 2017,
        "contributor": "COCO Consortium",
        "date_created": "2017/09/01"
    },
    "licenses": [
        {
            "url": "http://creativecommons.org/licenses/by/2.0/",
            "id": 4,
            "name": "Attribution License"
        }
    ],
    "images": [
        {
            "id": 242287,
            "license": 4,
            "coco_url": "http://images.cocodataset.org/val2017/xxxxxxxxxxxx.jpg",
            "flickr_url": "http://farm3.staticflickr.com/2626/xxxxxxxxxxxx.jpg",
            "width": 426,
            "height": 640,
            "file_name": "xxxxxxxxx.jpg",
            "date_captured": "2013-11-15 02:41:42"
        },
        {
            "id": 245915,
            "license": 4,
            "coco_url": "http://images.cocodataset.org/val2017/nnnnnnnnnnnn.jpg",
            "flickr_url": "http://farm1.staticflickr.com/88/xxxxxxxxxxxx.jpg",
            "width": 640,
            "height": 480,
            "file_name": "nnnnnnnnnn.jpg",
            "date_captured": "2013-11-18 02:53:27"
        }
    ],
    "annotations": [
        {
            "id": 125686,
            "category_id": 0,
            "iscrowd": 0,
            "segmentation": [
                [
                    164.81,
                    417.51,......167.55,
                    410.64
                ]
            ],
            "image_id": 242287,
            "area": 42061.80340000001,
            "bbox": [
                19.23,
                383.18,
                314.5,
                244.46
            ]
        },
        {
            "id": 1409619,
            "category_id": 0,
            "iscrowd": 0,
            "segmentation": [
                [
                    376.81,
                    238.8,........382.74,
                    241.17
                ]
            ],
            "image_id": 245915,
            "area": 3556.2197000000015,
            "bbox": [
                399,
                251,
                155,
                101
            ]
        },
        {
            "id": 1410165,
            "category_id": 1,
            "iscrowd": 0,
            "segmentation": [
                [
                    486.34,
                    239.01,..........495.95,
                    244.39
                ]
            ],
            "image_id": 245915,
            "area": 1775.8932499999994,
            "bbox": [
                86,
                65,
                220,
                334
            ]
        }
    ],
    "categories": [
        {
            "supercategory": "speaker",
            "id": 0,
            "name": "echo"
        },
        {
            "supercategory": "speaker",
            "id": 1,
            "name": "echo dot"
        }
    ]
}


"""


import pandas as pd
import json
import os
import tqdm

CLASS_DICT = {
    'pedestrian' : 1,
    'rider' : 2,
    'car' : 3,
    'truck' : 4, 
    'bus' : 5, 
    'train' : 6, 
    'motorcycle' : 7,
    'bicycle' : 8,
    'traffic light' : 9,
    'traffic sign' : 10,
    'other vehicle': 11,
    'trailer': 12,
    'other person': 13,
}

class Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(Encoder, self).default(obj)

class BDD2COCO():
    def __init__(self, bdd_root, class_dict = CLASS_DICT, convert_det = True, convert_seg = False) -> None:
        assert convert_det or convert_seg == True, "Provide at least one conversiton"

        self.coconvert_det = convert_det
        self.convert_seg = convert_seg
        self.root = bdd_root
        self.class_dict = class_dict

    def generate_info_dict(self):
        return  {
                "description": "BDD100k Dataset",
                "url": "",
                "version": "1.0",
                "year": 2017,
                "contributor": "",
                "date_created": ""
            }
    
    def generate_licenses_list(self):
        return [
            {
                "url": "http://creativecommons.org/licenses/by/2.0/",
                "id": 4,
                "name": "Attribution License"
            }
        ]
    
    def generate_image_entry(self, id: int, filename: str, timestamp):
        return   {
            "id": id,
            "license": 4,
            "coco_url": "",
            "flickr_url": "",
            "width": 1080,
            "height": 720,
            "file_name": filename,
            "date_captured": timestamp # "2013-11-15 02:41:42"
        }
    
    def generate_annotation_entry(self, id: int, category_id: int, image_id: int, bbox: None, area = None, segmentation = None):
        x = {
            "id": id,
            "category_id": category_id,
            "iscrowd": 0,
            "image_id": image_id,
        }
        if bbox:
            x["bbox"] = bbox
        if area:
            x["area"] = area
        if segmentation:
            x["segmentation"] = segmentation
        return x

    def generate_categories(self):
        categories = []
        for name, id in self.class_dict.items():
            categories.append({
                "supercategory": "none",
                "id": id,
                "name": name
            })
        return categories

    @staticmethod
    def xyxy2xywh_coco(bbox_bdd):
        return [bbox_bdd['x1'], bbox_bdd['y1'], bbox_bdd['x2'] - bbox_bdd['x1'], bbox_bdd['y2'] - bbox_bdd['y1']]

    def generate(self, image_set = "train", save_path = '/data/mtsysin/ipa/LaneDetection_F23/bdd100k/labels', segmentation_to_bbox = True):
        
        detect = pd.read_json(os.path.join(self.root, f'labels/det_20/det_{image_set}.json'))
        detect.dropna(axis=0, subset=['labels'], inplace=True)

        images = []
        annotations = []

        for idx in tqdm.tqdm(range(len(detect.index))):
            target = detect.iloc[idx]
            # add image
            images.append(
                self.generate_image_entry(
                    id = idx,
                    filename=target['name'],
                    timestamp=target['timestamp']
                )
            )

            for label in target['labels']:
                bbox = label['box2d']
                bbox_coco = self.xyxy2xywh_coco(bbox)
                x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                if segmentation_to_bbox:
                    seg_coco = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                else:
                    raise NotImplementedError("Segmentation not supported fully yet")
                area_coco = (x2 - x1) * (y2 - y1)
                annotations.append(
                    self.generate_annotation_entry(
                        id = label['id'],
                        category_id=self.class_dict[label['category']],
                        image_id=idx,
                        bbox = bbox_coco,
                        segmentation=seg_coco,
                        area=area_coco
                    )
                )

        # build annotation file
        data = {
            "info": self.generate_info_dict(),
            "licenses": self.generate_licenses_list(),
            "images": images,
            "annotations": annotations,
            "categories": self.generate_categories()
        }


        save_path = os.path.join(save_path, "labels_coco")
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        with open(os.path.join(save_path, f"{image_set}.json"), 'w') as fp:
            json.dump(data, fp, cls=Encoder)


import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import requests
from pycocotools.coco import COCO


def test_coco_format(
        idx = 5,
        coco_annotation_file_path = "/data/mtsysin/ipa/LaneDetection_F23/bdd100k/labels/labels_coco",
        img_path = "/data/mtsysin/ipa/LaneDetection_F23/bdd100k/images/100k",
        image_set = "val"
    ):

    coco_annotation = COCO(annotation_file=os.path.join(coco_annotation_file_path, f"{image_set}.json"))

    # Category IDs.
    cat_ids = coco_annotation.getCatIds()
    print(f"Number of Unique Categories: {len(cat_ids)}")
    print("Category IDs:")
    print(cat_ids)  # The IDs are not necessarily consecutive.

    # All categories.
    cats = coco_annotation.loadCats(cat_ids)
    cat_names = [cat["name"] for cat in cats]
    print("Categories Names:")
    print(cat_names)

    # Category ID -> Category Name.
    query_id = cat_ids[0]
    query_annotation = coco_annotation.loadCats([query_id])[0]
    query_name = query_annotation["name"]
    query_supercategory = query_annotation["supercategory"]
    print("Category ID -> Category Name:")
    print(
        f"Category ID: {query_id}, Category Name: {query_name}, Supercategory: {query_supercategory}"
    )

    # Category Name -> Category ID.
    query_name = cat_names[2]
    query_id = coco_annotation.getCatIds(catNms=[query_name])[0]
    print("Category Name -> ID:")
    print(f"Category Name: {query_name}, Category ID: {query_id}")

    # Get the ID of all the images containing the object of the category.
    img_ids = coco_annotation.getImgIds(catIds=[query_id])
    print(f"Number of Images Containing {query_name}: {len(img_ids)}")

    # Pick one image.
    img_id = img_ids[idx]
    img_info = coco_annotation.loadImgs([img_id])[0]
    img_file_name = img_info["file_name"]
    img_url = img_info["coco_url"]
    print(
        f"Image ID: {img_id}, File Name: {img_file_name}, Image URL: {img_url}"
    )

    # Get all the annotations for the specified image.
    ann_ids = coco_annotation.getAnnIds(imgIds=[img_id], iscrowd=None)
    anns = coco_annotation.loadAnns(ann_ids)
    print(f"Annotations for Image ID {img_id}:")
    print(anns)

    # Open image.
    im = Image.open(os.path.join(img_path, image_set, img_file_name)).convert("RGB")

    # Save image and its labeled version.
    plt.axis("off")
    plt.imshow(np.asarray(im))
    plt.savefig(f"{img_id}.jpg", bbox_inches="tight", pad_inches=0)
    # Plot segmentation and bounding box.
    coco_annotation.showAnns(anns, draw_bbox=True)
    plt.savefig(f"{img_id}_annotated.jpg", bbox_inches="tight", pad_inches=0)

    return


if __name__ == "__main__":
    gen = BDD2COCO(bdd_root="/data/mtsysin/ipa/LaneDetection_F23/bdd100k")
    gen.generate(image_set="val")
    test_coco_format(10) # Test image output 
