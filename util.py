import torchvision.transforms as trns
from PIL import Image
import numpy as np

import dataset

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
import cv2
import matplotlib.pyplot as plt
import torch
class conversion():
    def __init__(self):
       print('conversion activated')

    def show_tensor_to_PIL(self, src):
        return trns.ToPILImage()(src).convert("RGB")

    def tensor_to_cv2(self, src , invert=True):
        if invert==True:
            return cv2.cvtColor(np.invert(np.asarray(self.show_tensor_to_PIL(src))),cv2.COLOR_RGB2BGR)
        else :
            return cv2.cvtColor(np.asarray(self.show_tensor_to_PIL(src)),cv2.COLOR_RGB2BGR)
    def cv2_to_tensor(self,src):
        trans = trns.Compose([
            trns.Resize([224,224]),
            trns.ToTensor(),
            #dataset.valid_normalized()
        ])
        out = Image.fromarray(cv2.cvtColor(src,cv2.COLOR_BGR2RGB))
        out = torch.unsqueeze(trans(out),0)
        return out
#conversion = conversion()

