import torch
from torch.utils.data import DataLoader
import unittest
import torchvision.transforms as transforms
import torch.utils.data as data
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

from train import BDD_100K_ROOT

from bdd100k import *


class TestBDD100k(unittest.TestCase):
    def test_transform(self):
        img = read_image(BDD_100K_ROOT + 'images/100k/train/' + "730585a8-d57b052f.jpg") 
        img = img.type(torch.float32)

        print(1, img)
        img = transforms.Resize((384, 640), interpolation=transforms.InterpolationMode.NEAREST)(img)
        print(2, img)
        img = img/255
        img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
        print(3, img)

        pass

    def test_single_process_dataloader(self):
        train_dataset = BDD100k(root=BDD_100K_ROOT, dataset_type = 'train')
        self._check_dataloader(train_dataset, num_workers=0)     
        test_dataset = BDD100k(root=BDD_100K_ROOT, dataset_type = 'val')
        self._check_dataloader(test_dataset, num_workers=0)

    def test_multi_process_dataloader(self):
        train_dataset = BDD100k(root=BDD_100K_ROOT, train=True)
        self._check_dataloader(train_dataset, num_workers=2)
        test_dataset = BDD100k(root=BDD_100K_ROOT, train=False)
        self._check_dataloader(test_dataset, num_workers=2)

    def test_dataloader_1(self):
        #Test Dataloader 1
        '''
        test = TestBDD100k()
        test.test_multi_process_dataloader()
        '''
        #Test Dataloader 2

        print("Beginning dataloader tests...")

        transform = transforms.Compose([
            transforms.Resize((384, 640), interpolation=transforms.InterpolationMode.NEAREST),
        ])
        #Load BDD100k Dataset
        train_dataset = BDD100k(root='/data/stevenwh/bdd100k/', train=True, transform=transform, anchors=ANCHORS)
        val_dataset = BDD100k(root='/data/stevenwh/bdd100k/', train=False, transform=transform, anchors=ANCHORS)

        train_start = time.time()

        train_loader = data.DataLoader(dataset=train_dataset,
                                    batch_size=2,
                                    num_workers=12, # start at 12, go up to 20 to see which is faster
                                    shuffle=True)   # 12 WAS FASTER. And doing 20 gave a warning
        train_end = time.time()
        print("Training: ", (train_end - train_start))

        val_start = time.time()
        val_loader = data.DataLoader(dataset=val_dataset, 
                                    batch_size=2,
                                    num_workers=12,
                                    shuffle=False)
        val_end = time.time()
        print("Validation: ", (val_end - val_start))

        imgs, dets, lanes, drives = next(iter(val_loader))

    def test_dataset_scaling_and_reversion(self, indices, *args, **kwargs):
        """
        Dataset testing: get example outputs of the dataset
        Reverse-engineer the output of the dataset and get origianl images"
        """
        # Create dataset
        dataset = BDD100k(
            root = BDD_100K_ROOT, 
            *args,
            **kwargs
        )

        C = len(CLASS_DICT)

        # Generate sample outputs of the dataset
        for idx in indices:
            # Get index from the dataset
            image, label, _ = dataset[idx]
            _, img_size_y, img_size_x = image.size()

            image = image.permute(1, 2, 0)

            # OpenCV complains that the image is not contiguous after the permute. Force it. 
            # image = np.uint8(image)
            image = np.ascontiguousarray(image, dtype=np.uint8)

            print("Image type: ", type(image))
            print("Image size: ", image.shape)
            print("Image range: ", image.min(), image.max())
            print("Image snippet: ", image[4:6, 3:8, :])
            print("Label size: ", [l.size() for l in label])
            
            for scale_idx, l in enumerate(label):
                print(f"###########\nSCALE {scale_idx}\n###########")

                # Find indices and bboxes where there is an image on the current scale:
                Iobj_i = l[..., C].bool()

                selected_igms = l[Iobj_i]
                selected_igms_positions = Iobj_i.nonzero(as_tuple=False)

                print("Selected yolo vectors: ", selected_igms)
                print("Selected yolo vectors positions: ", selected_igms_positions)

                Sy, Sx = GRID_SCALES[scale_idx]                                                          # Get the output grid size for the chosen scale

                # Show image and corresponding bounding boxes:
                for yolo_vector, position in zip(selected_igms, selected_igms_positions):
                    # get bbox values and convert them to scalars
                    x, y, w, h = yolo_vector[C+1:C+5].tolist()
                    class_vector = yolo_vector[:C].tolist()
                    class_index = class_vector.index(1.0)
                    anchor_idx, y_idx, x_idx = position.tolist()

                    # Select correct dimensions
                    print("old", x, y, w, h, x_idx, img_size_x, Sx)
                    x = int((x + x_idx) * img_size_x / Sx)
                    y = int((y + y_idx) * img_size_y / Sy)
                    w = int(w * img_size_x / Sx)
                    h = int(h * img_size_y / Sy)

                    print("new", x, y, w, h)
                    image = cv2.rectangle(image, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (36, 255, 12), 2) 
                    image = cv2.putText(image, REVERSE_CLASS_DICT[class_index], (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX , 0.3, (36, 255, 12), 2)


            cv2.imwrite(f'out/test_dataset_scaling_and_reversion_{idx}.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


    def _check_dataloader(self, data, num_workers):
        """This function only tests that the loading process throws no error"""
        print("Initializing data loader...")
        loader = DataLoader(data, batch_size=4, num_workers=num_workers)
        print("Loading images...")
        for _ in loader:
            pass
        print("Success")

    def test_target(self):
        dataset = BDD100k(root=BDD_100K_ROOT, train=False)
        loader = DataLoader(dataset, batch_size=4)
        for img, target in loader:
            pass





