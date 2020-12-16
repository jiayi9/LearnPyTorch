#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


DATA_DIR = Path('../train_data/seg_100/seg_100/')
# x_train_dir = DATA_DIR/'train'
# y_train_dir = DATA_DIR/'train_mask'
# x_valid_dir = DATA_DIR/'val'
# y_valid_dir = DATA_DIR/'val_mask'
# x_test_dir = DATA_DIR/'test'
# y_test_dir = DATA_DIR/'test_mask'
# print('\n', x_train_dir, '\n', y_train_dir, '\n',x_valid_dir, '\n',y_valid_dir, '\n',x_test_dir, '\n',y_test_dir)
#print(os.listdir(DATA_DIR))


# In[3]:


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


# In[4]:


# class SegDataset(Dataset):
#     def __init__(self, images_dir, masks_dir):
#         self.ids = os.listdir(images_dir)        
#         self.image_file_paths = [os.path.join(images_dir, image_id) for image_id in self.ids]
#         self.mask_file_paths = [os.path.join(masks_dir, image_id) for image_id in self.ids]
#     def __getitem__(self, i):
#         image = cv2.imread(self.image_file_paths[i])
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         # the link
#         mask = cv2.imread(self.mask_file_paths[i].replace(".bmp","_label.png"), 0)
        
#         SIZE = (380,380)
#         image = cv2.resize(image, SIZE)
#         mask = cv2.resize(mask, SIZE)
#         trans = transforms.Compose([transforms.ToTensor()])
#         image = trans(image)
#         mask = trans(mask)
        
#         return image, mask
#     def __len__(self):
#         return len(self.ids)


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
        
        #print(image)
        
        #image = Normalize(mean, std)(image)
        
        mask = trans_2(mask)
        mask[mask>0] = 1
        #mask = np.transpose(mask, (2,0,1))
        #print(mask.shape)
        #mask = np.expand_dims(mask,0)
        #print(mask.shape)
        #mask = torch.Tensor(mask)
        #print(mask.shape)

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



test1 = total_set.image_file_paths[0:10]
test2 = total_set.mask_file_paths[0:10]

for i in range(10):
    print(test1[i], test2[i])


# In[8]:


# Check dimensions starts
X, Y = train_dataset[0]
print(X.shape)
print(Y.shape)


# In[9]:


print(X)
print(X.max())
print(Y)
print(Y.max())
visualize_tensor(X = X, Y = Y)


# In[10]:


train_loader = DataLoader(train_dataset, batch_size=40, shuffle=True, num_workers=2)
valid_loader = DataLoader(valid_dataset, batch_size=10, shuffle=False, num_workers=2)

#X, Y = train_loader.dataset[0]
X_batch, Y_batch = next(iter(train_loader))
print(X_batch.shape)
print(Y_batch.shape)
# Check dimensions ends


# In[ ]:





# In[11]:


print(X.max())


# In[12]:


print(Y.max())


# In[ ]:





# In[13]:


# ENCODER = 'se_resnext50_32x4d'
# ENCODER_WEIGHTS = 'imagenet'
# CLASSES = ['flake']
# ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multicalss segmentation
# DEVICE = 'cuda'


# In[14]:


#EfficientNet B5
ENCODER = 'efficientnet-b4'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['flake']
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multicalss segmentation
DEVICE = 'cuda'


# In[15]:


# create segmentation model with pretrained encoder
# model = smp.FPN(
#     encoder_name=ENCODER, 
#     encoder_weights=ENCODER_WEIGHTS, 
#     classes=len(CLASSES), 
#     activation=ACTIVATION,
# )

model = smp.Unet(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
)


# In[16]:


preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
print(preprocessing_fn)


# In[17]:



OUTPUT = model(X_batch)



# In[18]:


OUTPUT.shape


# In[19]:


visualize_tensor(X = OUTPUT[0].detach().numpy())


# In[20]:


#print(OUTPUT[0].detach().numpy())


# In[21]:


model_para = nn.DataParallel(model)


# In[22]:



loss = smp.utils.losses.DiceLoss()

metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.0001),
])


# In[23]:



train_epoch = smp.utils.train.TrainEpoch(
    model_para, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model_para, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)


# In[24]:



max_score = 0

DICE_LOSS_TRAIN = []
IOU_SCORE_TRAIN = []
DICE_LOSS_VAL = []
IOU_SCORE_VAL = []

for i in range(0, 100):
    
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)
    
    DICE_LOSS_TRAIN.append(train_logs['dice_loss'])
    IOU_SCORE_TRAIN.append(train_logs['iou_score'])
    DICE_LOSS_VAL.append(valid_logs['dice_loss'])
    IOU_SCORE_VAL.append(valid_logs['iou_score'])
    
    # do something (save model, change lr, etc.)
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, './best_model_CRN_para_eff5.pth')
        print('Model saved!')
        
    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')


# In[25]:


import pandas as pd
REPORT = pd.DataFrame({
        'DICE_LOSS_TRAIN':DICE_LOSS_TRAIN,
        'IOU_SCORE_TRAIN':IOU_SCORE_TRAIN,
        'DICE_LOSS_VAL':DICE_LOSS_VAL,
        'IOU_SCORE_VAL':IOU_SCORE_VAL
})
       
print(REPORT)
REPORT.to_csv("seg_progress_report_crn_0_06.csv")
        


# In[26]:


import pandas as pd
REPORT = pd.read_csv("seg_progress_report_crn.csv")
plt.plot(REPORT.DICE_LOSS_TRAIN)
plt.plot(REPORT.DICE_LOSS_VAL)


# In[ ]:


REPORT


# In[ ]:


128/5


# In[ ]:





# In[ ]:




