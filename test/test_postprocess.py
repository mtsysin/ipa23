import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from model.YoloMulti import YoloMulti
import unittest
from postprocess import *
from bdd100k import *
from train import ROOT, USE_PARALLEL, INPUT_IMG_TRANSFORM, USE_DDP, SHUFFLE_OFF
from loss import DetectionLoss

device = torch.device('cuda:0')
print(torch.cuda.is_available()) 
print(f"CUDA device: {torch.cuda.current_device()}")
print(f"CUDA device count: {torch.cuda.device_count()}")

class TestPostprocess(unittest.TestCase):
    def test_postprocess_random(self):
        """Try to run this for random prediction"""
        pred1 = torch.rand(16, 3, 10, 20, 18)
        pred2 = torch.rand(16, 3, 10, 20, 18)
        pred3 = torch.rand(16, 3, 10, 20, 18)
        pred = [pred1, pred2, pred3]

        nms_b = get_bboxes(pred, 0.7, 0.7)

    def test_postprocess_on_dataset_output(self):
        """Try to run this for sample prediction"""
        index = 3
        # Create dataset
        dataset = BDD100k(
            root = BDD_100K_ROOT,
            transform = transforms.Compose([
                transforms.Resize((384, 640), interpolation=transforms.InterpolationMode.NEAREST)
            ])
        )
        loader = DataLoader(dataset, batch_size=4)

        C = len(CLASS_DICT)
        iterator = iter(loader)
        for _ in range(4):
            image, label, _ = next(iterator)

        for l in label:
            assert not torch.isnan(l).any()
        assert not torch.isnan(image).any()


        print(f"label[{0}].size()", label[0].size())
        nms_b = get_bboxes(label, 0.7, 0.7, true_prediction=False) # Get bboxes for a the first image in a btach

        print(len(nms_b))
        print(nms_b[index].size())
        print(nms_b[index])

        print(image[index, 2:20, 3:30])

        image = draw_bbox(image, dets=nms_b)
        image = (image/255.)
        torchvision.utils.save_image(image[index], "out/test_postprocess_on_dataset_output.png")

    def test_postprocess_on_simple_pretrained_model(self):

        index = 6

        """Try to run this for sample prediction"""

        
        dataset = BDD100k(root=BDD_100K_ROOT, train=True, transform=INPUT_IMG_TRANSFORM, anchors=ANCHORS)
        indices = [i for i in range (24*30)]
        dataset = data.Subset(dataset, indices)

        # val_dataset = BDD100k(root='/home/pumetu/Purdue/LaneDetection/BDD100k/', train=False, transform=transform, anchors=ANCHORS)

        loader = data.DataLoader(dataset=dataset, 
                                    batch_size=24,
                                    num_workers=2,
                                    shuffle = False if USE_DDP or SHUFFLE_OFF else True, 
                                    sampler= None
        )

        C = len(CLASS_DICT)
        iterator = iter(loader)
        for _ in range(21):
            image, det, _ = next(iterator)

        model = YoloMulti()
        if USE_PARALLEL:
            # weights = torch.load(ROOT+'/model.pt', map_location='cpu')
            # model.load_state_dict(weights, strict=False)
            model= torch.nn.DataParallel(model, device_ids=[0, 1, 2])
            model.load_state_dict(torch.load(ROOT+'/model.pt'))

            model.to(device)
        # model.load_state_dict(torch.load(ROOT+'/model.pt'))
        # model.eval()

        image = image.to(device)


        label, _ = model(image)
        label = [d.to(device) for d in label]
        det = [d.to(device) for d in det]

        loss_fn = DetectionLoss()

        loss = loss_fn(label, det)

        print("loss::::::", loss)

        label = [l.cpu() for l in label]
        det = [l.cpu() for l in det]

        image = image.cpu()

        print(f"label[0].size()", label[0].size())
        nms_b = get_bboxes(label, 0.2, 0.6, true_prediction=True) # Get bboxes for a the first image in a btach
        # nms_b_true = get_bboxes(det, 0.5, 0.8, true_prediction=False) # Get bboxes for a the first image in a btach


        print(f"len(nms_b)", len(nms_b))
        print(f"nms_b[{index}].size()", nms_b[index].size())
        print(f"nms_b[{index}]", nms_b[index])

        # print(f"len(nms_b_true)", len(nms_b_true))
        # print(f"nms_b_true[{index}].size()", nms_b_true[index].size())
        # print(f"nms_b_treu[{index}]", nms_b_true[index])

        # print(image[0, 2:20, 3:30])
        image = (image + 1.) / 2. * 255.
        image = draw_bbox(image, dets=nms_b)
        image = image / 255.
        torchvision.utils.save_image(image[index], "out/test_postprocess_on_simple_pretrained_model.png")

