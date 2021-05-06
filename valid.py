import numpy as np
import cv2
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, dataloader
from torchvision import transforms, utils
import glob
from PIL import Image
import pandas as pd 
import os
import dataset
import torchvision.models as models
import torch.optim as optim
from util import conversion


model = models.resnet50(pretrained=True)
model.fc = nn.Linear(2048,2)
model.load_state_dict(torch.load("./checkpoint.pth"))

model = model.cuda()

conversion = conversion()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
while(True):
    ret ,frame = cap.read()
    if ret == True:
        gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.2, 3)
        for (x,y,w,h) in faces:
            img_ori = frame.copy()
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            crop = img_ori[y:y+h, x:x+w]
            crop = conversion.cv2_to_tensor(cv2.resize(crop,(224,224))).cuda()
            output = model(crop)
            print(print(f"output : {output}"))
        cv2.imshow('src' , frame)
        cv2.imshow('gray' , gray)
        if cv2.waitKey(1) % 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()

