import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
from efficientnet_pytorch import EfficientNet
import torchvision
from matplotlib import pyplot as plt
model = smp.Unet('efficientnet-b0', encoder_weights='imagenet', classes = 1)

EfficientNet.get_image_size('efficientnet-b0')


num_epochs = 5
num_classes = 10
batch_size = 3
learning_rate = 0.001

#output = torch.sigmoid(model(images).float())



input_size = 224
tfms = transforms.Compose([
        transforms.Resize(input_size), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #transforms.Grayscale(num_output_channels=1),
        #transforms.Normalize([0.485], [0.229]),
        ])



all_image_path = "C:/LV_CHAO_IMAGE/simulation_data/"
all_data = torchvision.datasets.ImageFolder(root=all_image_path, transform = tfms)

train_set, val_set = torch.utils.data.random_split(all_data, [2000, 1000])
train_set_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True,  num_workers = 0)
val_set_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle = True,  num_workers = 0)


L = list(val_set_loader)

L[0][0].shape

y_hat = model(L[0][0])



plt.imshow(y_hat[0][0].detach().numpy())
