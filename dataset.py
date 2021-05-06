import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, dataloader
from torchvision import transforms, utils
import glob
from PIL import Image
import pandas as pd 
import os

class asian_face_dataset():
    def __init__(self,path_dataset="./tarball-lite/AFAD-Lite/*/*/*.jpg",
                path_sample = "./capture/*",transform=None):
        self.transform = transform
        self.asian_face = sorted(glob.glob(path_dataset))
        gt = sorted(glob.glob(path_sample))
        self.duplicate = list()
        for i in range(500):
            self.duplicate+=gt
        self.full_image = self.asian_face + self.duplicate
        self.full_label =np.concatenate((np.zeros(len(self.asian_face)),np.ones(50000)))
    def __len__(self):
       return len(self.full_label)
    def __getitem__(self , index):
        image = Image.open(self.full_image[index])
        label = self.full_label[index]
        if self.transform:
            image = self.transform(image)
        sample = {'image' :  image,"label" :label}
        return sample

"""
path = "./tarball-lite/AFAD-Lite/*/*/*.jpg"
asian_face = sorted(glob.glob(path))
gt = sorted(glob.glob('./capture/*'))

duplicate = list()
for i in range(500):
    duplicate+=gt

full_image = asian_face + duplicate

full_label =np.concatenate((np.zeros(len(asian_face)),np.ones(50000)))   
print(len(full_image))
print(len(full_label))
"""
trns = transforms.Compose([ transforms.Resize((224,224)),
                            transforms.ToTensor()])
data = asian_face_dataset(transform=trns)

#print(transforms.ToPILImage()(data[0]['image']).show())

#train_loader = DataLoader(data,batch_size=bs,shuffle=True,num_workers=os.cpu_count())
"""
for i ,sample in enumerate(train_loader):
    image = sample['image']
    label = sample['label']
"""
def get_train_loader(batch_size=16):
    return DataLoader(data,batch_size=batch_size,shuffle=True,num_workers=os.cpu_count())