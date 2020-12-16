#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torchvision.transforms as transforms
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import albumentations as albu
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
import pandas as pd
from albumentations import Resize, Normalize, Compose, VerticalFlip
from albumentations.pytorch import ToTensor as albu_ToTensor


# In[3]:


DATA_DIR = Path('../train_data/seg_100/seg_100/')


# In[4]:


# helper function for data visualization
def visualize(FIGSIZE = (16,5), **images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize = FIGSIZE)
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

def visualize_tensor(FIGSIZE = (16,5), **images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize = FIGSIZE)
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(np.transpose(image, (1,2,0)).squeeze())
    plt.show()


# In[5]:


class SegDataset(Dataset):
    def __init__(self, DATA_DIR):
        self.all_files = os.listdir(DATA_DIR)  

        self.image_file_names = [x for x in self.all_files if x[-4:] == '.bmp']
        self.image_file_names.sort()

        self.mask_file_names = [x for x in self.all_files if x[-4:] == '.png']
        self.mask_file_names.sort()
        
        self.image_file_paths = [os.path.join(DATA_DIR, x) for x in self.image_file_names]
        self.mask_file_paths = [os.path.join(DATA_DIR, x) for x in self.mask_file_names]

        #assert len(self.image_file_paths) == len(self.mask_file_paths)
        print(len(self.image_file_paths), len(self.mask_file_paths))
    def __getitem__(self, i):
        image = cv2.imread(self.image_file_paths[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # the link
        mask = cv2.imread(self.mask_file_paths[i], 0)
        #print(mask)
        SIZE = (384,384)
        
        image = cv2.resize(image, SIZE)
        mask = cv2.resize(mask, SIZE)
        
        #print(image)
        
        mean=(0.485, 0.456, 0.406)
        std=(0.229, 0.224, 0.225)
                
        trans_1 = transforms.Compose([

            transforms.ToTensor(),
            transforms.Normalize(mean = mean, std = std),

        ])
        
        trans_2 = transforms.Compose([
            transforms.ToTensor()
        ])

        image = trans_1(image)
        mask = trans_2(mask)
        mask[mask>0] = 1


        return image, mask
    def __len__(self):
        return len(self.image_file_names)


# In[6]:


total_set = SegDataset(DATA_DIR)
#print(train_dataset.all_files)

#train_dataset, valid_dataset = torch.utils.data.random_split(total_set, [2100, 342])

train_dataset = total_set
valid_dataset = total_set


# In[7]:



test1 = total_set.image_file_paths
test2 = total_set.mask_file_paths

# for i in range(0,len(test1)):
#     if test1[i].split('_')[3] != test2[i].split('_')[3]:
#         print(test1[i].split('_')[3], test2[i].split('_')[3])
    
D = pd.DataFrame({"image":test1, "mask":test2})
D.to_csv("image_and_mask.csv")


# In[8]:


# Check dimensions starts
X, Y = train_dataset[2209]
print(X.shape)
print(Y.shape)


# In[9]:


print(X)
print(X.max())
print(Y)
print(Y.max())
def convert_back(X):
    return (X - X.min())/(X.max() - X.min())
visualize_tensor(X =convert_back(X), Y = Y)


# In[10]:


train_loader = DataLoader(train_dataset, batch_size=40, shuffle=True, num_workers=2)
valid_loader = DataLoader(valid_dataset, batch_size=10, shuffle=False, num_workers=2)

#X, Y = train_loader.dataset[0]
X_batch, Y_batch = next(iter(train_loader))
print(X_batch.shape)
print(Y_batch.shape)
# Check dimensions ends


# In[11]:


model = torch.load("./best_model_CRN_para_eff5.pth")


# In[12]:


import torch.nn as nn
from collections import OrderedDict
from torchvision import models

new_state_dict = OrderedDict()

old_state_dict = model.state_dict()
print(old_state_dict)


# In[13]:


for k, v in old_state_dict.items():
        if k[:7] == 'module.': # remove module.
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v 


# In[14]:


model_2 = model


# In[15]:


model_2.load_state_dict(new_state_dict)


# In[16]:


torch.cuda.reset_accumulated_memory_stats()


# In[17]:


try:
    del INPUT
except:
    print('There is no INPUT')
X_batch, Y_batch = next(iter(train_loader))
print(X_batch.shape)
print(Y_batch.shape)


# In[18]:


INPUT = X_batch[0:5]
INPUT = INPUT.cuda()
Output = model_2(INPUT)


# In[19]:


for i in range(5):
#     i = 0
    X_Origin = X_batch[i].detach().cpu()
    X = Output[i]
    X2 = X.detach().cpu()
    Y2 = Y_batch[i].detach().cpu()

    print(Y2.max())
    visualize_tensor(Origin = (X_Origin - X_Origin.min())/(X_Origin.max()-X_Origin.min()),Predicted_Area = X2, Real_Area = Y2)
    plt.show()


# In[20]:


import time

t1 = time.time()
INPUT = X_batch[0:5]
t2 = time.time()
print(t2-t1)

t1 = time.time()
INPUT = INPUT.cuda()
t2 = time.time()
print(t2-t1)

t1 = time.time()
Output = model_2(INPUT)
t2 = time.time()
print(t2-t1)


# In[21]:


import time
t1 = time.time()
time.sleep(1)
t2 = time.time()

print(t2-t1)


# In[ ]:





# In[ ]:




