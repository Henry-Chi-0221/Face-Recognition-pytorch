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

    def get_mpl_colormap(self, cmap_name):
        cmap = plt.get_cmap(cmap_name)
        sm = plt.cm.ScalarMappable(cmap=cmap)
        color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:,2::-1]
        return color_range.reshape(256, 1, 3)

    def gray_to_cmap(self, src,m_type="plasma"):
        return cv2.applyColorMap(src,self.get_mpl_colormap(plt.get_cmap(m_type)))

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

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    #print(gt)
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred)**2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def compute_means_std(n_channels=3):
    from time import time
    import torch
    before = time()
    n_channels = n_channels

    train_set = dataset.depth_dataset_train(train=True,transform=dataset.train_transform)
    val_set   = dataset.depth_dataset_train(train=False,transform=dataset.val_transform)
    train_loader = DataLoader(
        dataset=train_set,
        num_workers=os.cpu_count()
    )
    val_loader = DataLoader(
        dataset=val_set,
        num_workers=os.cpu_count()
    )
    import sys

    mean = torch.zeros(3,1)
    std = torch.zeros(3,1)
    if n_channels ==1 :
        mean = torch.zeros(1)
        std = torch.zeros(1)
    length = len(train_set)
    for i ,sample in enumerate(train_loader):
        inputs_l,inputs_r,targets_l,targets_r = sample
        inputs = [inputs_l,inputs_r]
        if n_channels == 1 :
            inputs = [targets_l,targets_r]
        for j in range(n_channels):
            for k in range(1):
                mean[j] += inputs[k][:,j,:,:].mean()
                std[j] += inputs[k][:,j,:,:].std()
        print(f"{i+1} / {length}")



    mean.div_(length)
    std.div_(length)
    print(f"mean : {mean}")
    print(f"std  : {std} ")
    print("time elapsed: ", time()-before)


#compute_means_std(3)
#compute_means_std(1)

def test_dataloader():
    conversion = conversion()
    train_set = dataset.depth_dataset_train(train=True,transform=dataset.train_transform)
    val_set   = dataset.depth_dataset_train(train=False,transform=dataset.val_transform)
    train_loader = DataLoader(
        dataset=train_set,
        num_workers=os.cpu_count(),
    )
    val_loader = DataLoader(
        dataset=val_set,
        num_workers=os.cpu_count()
    )
    for i,sample in enumerate(train_loader):
        raw_l , _,_,_ = sample
        conversion.show_tensor_to_PIL(raw_l[0]).show()
        break


#test_dataloader()

def autoremove():
    import glob
    ls = ['input' , 'output','target']
    for i in ls:
        path =glob.glob(os.path.join("./tmp" , i,"*") )
        for j in path:
            os.remove(j)
        
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


