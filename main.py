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
num_epoch = 2
lr = 0.0005
bs = 4
train_loader , test_loader = dataset.get_train_loader(batch_size = bs)
num_classes = 2



resnet = models.resnet18(pretrained=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters() , lr = lr)
resnet.fc = nn.Linear(512,num_classes)
resnet = resnet.cuda()
def train(epoch):
    epoch =epoch
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


def test():
    net = models.resnet18(pretrained=True)
    net.fc = nn.Linear(512,num_classes)
    net = net.cuda()
    net.load_state_dict(torch.load("./checkpoint.pth"))
    correct = 0
    total = 0
    with torch.no_grad():
        for i ,sample in enumerate(test_loader):
            image , label = sample["image"].cuda() , sample["label"].cuda()
            output = net(image)
            _,predicted = torch.max(output.data, 1)
            total+= label.size(0)
            correct += (predicted==label).sum().item()
            #print(label)
            if i%100 == 0:
                print(f"{i}/{32816/bs}")
    print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

for epoch in range(num_epoch):
    resnet.train()
    train(epoch)
    resnet.eval()
    test()