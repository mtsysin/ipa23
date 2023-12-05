from model import YoloV1
import pathlib
import torch
from VOC import VocDataset
import cv2 as cv
import numpy as np

# load model
PATH = pathlib.Path(".") / "weights_loss305" / "state.pt";
model = YoloV1();
model.load_state_dict(torch.load(str(PATH.absolute())));

# *****
#  we assume 448 by 448 img
# split in 7 grids
# and 2 bboxes
def visualize_detect(idx):


    if(isinstance(idx, np.ndarray)):
        img_448 = cv.resize(idx, (448,448),interpolation=cv.INTER_LINEAR);
    else:
        # get image
        dataset = VocDataset();
        img_448, target = dataset.__getitem__(idx);
    # send image through the model
    detection = model(img_448.view(-1,*img_448.shape)).view(7,7,30);

    img = cv.imread(str(dataset.paths[idx]));
    img_448 = cv.resize(img,(448,448), interpolation=cv.INTER_LINEAR);
    for S1, S2 in dataset.grids[idx]:
        print(S1, S2);
        if(S1 != -1 and S2 != -1):
            # bounding box 1
            x1, y1 = (detection[S1][S2][:2] + torch.tensor([S1, S2]))*64 - detection[S1][S2][2:4]*224;
            x2, y2 = (detection[S1][S2][:2] + torch.tensor([S1, S2]))*64 + detection[S1][S2][2:4]*224;
            # print(x1,y1,x2,y2)
            cv.rectangle(img_448,(int(x1), int(y1)),(int(x2),int(y2)),color=(0,0,255),thickness=1);
            # check the target to see if there is another box in the grid
            # if(target[S1][S2][9] != 0):
            #     x2, y2 = (detection[S1][S2][:2] + torch.tensor([S1, S2]))*64 + detection[S1][S2][2:4]*448;
    # 7x7x30
    cv.imwrite(f"prediction{idx}.jpg", img_448);




