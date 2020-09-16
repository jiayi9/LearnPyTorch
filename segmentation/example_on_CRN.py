
import torchvision.transforms as transforms
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import albumentations as albu

from torch.utils.data import DataLoader
from torch.utils.data import Dataset 


# list all files under a folder recursively
def list_files_recur(path, format = '.BMP'):
    file_paths = []
    file_names = []
    for r, d, f in os.walk(path):
        for file in f:
            if format in file or format.lower() in file:
                file_paths.append(os.path.join(r, file))
                file_names.append(file)
    return([file_paths, file_names])

# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()
    
def visualize_2(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        if len(image.shape) == 3:
            plt.imshow(np.transpose(image,(1,2,0)))
        else:
            plt.imshow(image)
    plt.show()

DATA_DIR = 'C:/PyTorch/lujiayi/CRN_flake'

x_train_dir = os.path.join(DATA_DIR, 'train')
y_train_dir = os.path.join(DATA_DIR, 'train_mask')

x_valid_dir = os.path.join(DATA_DIR, 'val')
y_valid_dir = os.path.join(DATA_DIR, 'val_mask')

x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'test_mask')

print('\n', x_train_dir, '\n', y_train_dir, '\n',x_valid_dir, '\n',y_valid_dir, '\n',x_test_dir, '\n',y_test_dir)

train_images_fps = list_files_recur(x_train_dir)[0]
train_masks_fps = list_files_recur(y_train_dir,'png')[0]

print(len(train_images_fps), len(train_masks_fps))



#    CLASSES = ['sky', 'building', 'pole', 'road', 'pavement', 
#               'tree', 'signsymbol', 'fence', 'car', 'pedestrian', 
#               'bicyclist', 'unlabelled']
#    classes = ['sky']
#    class_values = [CLASSES.index(cls.lower()) for cls in classes]

class_values = [128]

image = cv2.imread(train_images_fps[0])
# BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print(image.shape)

mask = cv2.imread(train_masks_fps[0], 0)
print(mask.shape)
print(np.transpose(np.unique(mask, return_counts = True)))

mask = cv2.imread(train_masks_fps[0])[:,:,2]
print(mask.shape)
print(np.transpose(np.unique(mask, return_counts = True)))

plt.imshow(mask)

masks = [(mask == v) for v in class_values]

print(masks[0].shape)

mask = np.stack(masks, axis=-1).astype('float')

print(mask.shape)

np.stack([np.array([[1,1],[1,1]]),np.array([[1,1],[1,1]])])
np.stack([np.array([1]),np.array([1])])
np.stack([np.array([[1,1],[1,1]]),np.array([[1,1],[1,1]])], axis = -1)
np.stack([np.array([[1,1],[1,1]]),np.array([[1,1],[1,1]])], axis = 0)
np.stack([np.array([[1,1],[1,1]]),np.array([[1,1],[1,1]])], axis = 2)

plt.imshow(image)
plt.imshow(mask.squeeze())


class SegDataset(Dataset): 
    def __init__(self, images_dir, masks_dir, augmentation=None, preprocessing=None):
        self.ids = os.listdir(images_dir)
        self.images_fps = list_files_recur(images_dir)[0]
        self.masks_fps = list_files_recur(masks_dir,'png')[0]
        self.class_values = [128]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i])[:,:,2]
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
                # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
#        image = cv2.resize(image,(320,320)) 
#        mask = cv2.resize(mask,(320,320))
#        mask = np.expand_dims(mask,2)
#        
#        image = np.transpose(image, (2, 0,1))
#        mask = np.transpose(mask, (2, 0,1))
        
        return image, mask
        
    def __len__(self):
        return len(self.ids)

dataset = SegDataset(x_train_dir, y_train_dir)



image, mask = dataset[4] # get some sample
visualize(image = image, mask = mask.squeeze())
print(image.shape, mask.shape)

# (320, 320, 3) (320, 320, 1)




#transforms = [albu.HorizontalFlip(p=1)]
#
#get_aug = albu.Compose(transforms)
#
#train_dataset = SegDataset(x_train_dir, y_train_dir, 
#                           augmentation = get_aug,
#                           preprocessing = Composed_transform
#                           )
#
#
#
#image, mask = train_dataset[0] 
#
#print(image.shape, mask.shape)
#
#visualize_2(image = image, mask = mask.squeeze())



#preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
#
#
#_transform = [
#        albu.Lambda(image=preprocessing_fn),
#        albu.Lambda(image=to_tensor, mask=to_tensor),
#]
#
#Composed_transform = albu.Compose(_transform)








import albumentations as albu

def get_training_augmentation():
    train_transform = [

        #albu.HorizontalFlip(p=0.5),

        #albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        albu.RandomCrop(height=320, width=320, always_apply=True),

        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(384, 480),
        albu.Resize(320,320)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)





import torch
import numpy as np
import segmentation_models_pytorch as smp

ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['car']
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multicalss segmentation
DEVICE = 'cuda'

# create segmentation model with pretrained encoder
model = smp.FPN(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)




train_dataset = SegDataset(x_train_dir, y_train_dir, 
                           augmentation=get_training_augmentation(),
                           preprocessing=get_preprocessing(preprocessing_fn)
                           )

valid_dataset = SegDataset(x_valid_dir, y_valid_dir, 
                           augmentation=get_training_augmentation(),
                           preprocessing=get_preprocessing(preprocessing_fn)
                           )

image, mask = train_dataset[0]

#print(image)
#visualize(image = image, mask = mask.squeeze())


train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

loss = smp.utils.losses.DiceLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.0001),
])

train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)

# train model for 40 epochs

max_score = 0

DICE_LOSS_TRAIN = []
IOU_SCORE_TRAIN = []
DICE_LOSS_VAL = []
IOU_SCORE_VAL = []


for i in range(0, 50):
    
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
        torch.save(model, 'C:/PyTorch/lujiayi/best_model.pth')
        print('Model saved!')
        
    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')
        
        
        
REPORT = pd.DataFrame({
        'DICE_LOSS_TRAIN':DICE_LOSS_TRAIN,
        'IOU_SCORE_TRAIN':IOU_SCORE_TRAIN,
        'DICE_LOSS_VAL':DICE_LOSS_VAL,
        'IOU_SCORE_VAL':IOU_SCORE_VAL
})
        
REPORT.to_csv("C:/PyTorch/lujiayi/seg_progress_report.csv")
        



        
#        
#        
## load best saved checkpoint
#best_model = torch.load('C:/PyTorch/lujiayi/best_model.pth')
#
## create test dataset
#test_dataset = SegDataset(
#    x_test_dir, 
#    y_test_dir, 
#    augmentation=get_validation_augmentation(), 
#    preprocessing=get_preprocessing(preprocessing_fn)
#)
#
#img ,mask = test_dataset[0]
#
#print(img.shape)
#print(mask.shape)
##
##train_dataset = SegDataset(x_train_dir, y_train_dir, 
##                           augmentation=get_training_augmentation(),
##                           preprocessing=get_preprocessing(preprocessing_fn)
##                           )
##
##img ,mask = train_dataset[0]
##print(img.shape)
##print(mask.shape)
##
##test_dataset = SegDataset(
##    x_valid_dir, 
##    y_valid_dir, 
##    augmentation=get_training_augmentation(), 
##    preprocessing=get_preprocessing(preprocessing_fn),
##)
##test_dataset = SegDataset(
##    x_train_dir, 
##    y_train_dir, 
##    augmentation=get_validation_augmentation(), 
##    preprocessing=get_preprocessing(preprocessing_fn),
##)
#
#
##test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
#test_dataloader = DataLoader(test_dataset)
#
## evaluate model on test set
#test_epoch = smp.utils.train.ValidEpoch(
#    model=best_model,
#    loss=loss,
#    metrics=metrics,
#    device=DEVICE,
#)
#
#logs = test_epoch.run(test_dataloader)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

#test_dataset_vis = SegDataset(
#    x_test_dir, y_test_dir, 
#)
#        
#img, mask = test_dataset_vis[0]  
#        
#img.shape
#mask.shape
#
#n = 1
#
#    image_vis = test_dataset_vis[n][0].astype('uint8')
#    image, gt_mask = test_dataset_vis[n]
#    
#    gt_mask = gt_mask.squeeze()
#    
#    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
#    pr_mask = model.predict(x_tensor)
#    pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        
        
        
        
