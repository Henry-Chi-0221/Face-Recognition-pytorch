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
num_epoch = 1
lr = 0.001
train_loader = dataset.get_train_loader(batch_size = 16)
num_classes = 2


resnet = models.resnet50(pretrained=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters() , lr = lr)
resnet.fc = nn.Linear(2048,num_classes)
resnet = resnet.cuda()
print(resnet)

for epoch in range(num_epoch):
    running_loss = 0.0
    for i ,sample in enumerate(train_loader):
        image = sample['image'].cuda()
        label = sample['label'].to(torch.long).cuda()
        optimizer.zero_grad()

        output = resnet(image)

        loss = criterion(output,label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10== 9:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.6f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0
            torch.save(resnet.state_dict(),"./checkpoint.pth")



