import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, dataloader
from torchvision import transforms, utils
import glob
from PIL import Image
import pandas as pd 
import os
import math
class asian_face_dataset():
    def __init__(self,path_dataset="./tarball-lite/AFAD-Lite/*/*/*.jpg",
                path_sample = "./capture/*",transform=None,train=True):
        self.transform = transform

        self.asian_face = sorted(glob.glob(path_dataset))
        gt = sorted(glob.glob(path_sample))
        
        train_dataset_len = math.ceil(0.7*len(self.asian_face))
        train_sample_len = math.ceil(0.7*len(gt))
        if train==True:
            self.asian_face = self.asian_face[0:train_dataset_len]
            gt = gt[0:train_sample_len]
        else:
            self.asian_face = self.asian_face[train_dataset_len:]
            gt = gt[train_sample_len:]
        self.duplicate = list()
        for i in range(500):
            self.duplicate+=gt
        self.full_image = self.asian_face + self.duplicate
        self.full_label =np.concatenate((np.zeros(len(self.asian_face)),np.ones(len(self.full_image)-len(self.asian_face))))
    def __len__(self):
       return len(self.full_label)
    def __getitem__(self , index):
        image = Image.open(self.full_image[index])
        label = self.full_label[index]
        if self.transform:
            image = self.transform(image)
        sample = {'image' :  image,"label" :label}
        return sample

"
trns = transforms.Compose([ transforms.Resize((224,224)),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.RandomRotation(30),
                            transforms.RandomVerticalFlip(p=0.5),
                            #transforms.ColorJitter(),
                            transforms.ToTensor(),
                            
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trns_test = transforms.Compose([ transforms.Resize((224,224)),
                                transforms.ToTensor()])
train_set = asian_face_dataset(transform=trns,train=True)
test_set = asian_face_dataset(transform=trns_test,train=False)

def get_train_loader(batch_size=16):
    return DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=os.cpu_count()) , DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=os.cpu_count())