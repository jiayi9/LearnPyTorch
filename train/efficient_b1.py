import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from functions import evaluate_accuracy, train_3c, train_1c
from models import AlexNet,EfficientNet_b4,EfficientNet_b0,EfficientNet_b1
from matplotlib import pyplot as plt
import os

os.chdir('D:/vision/Test_CRN_cls')

##########################################################################################################
####################################           General config            #################################
##########################################################################################################

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

##########################################################################################################
####################################            Hyper parameters         #################################
##########################################################################################################

num_epochs = 200
num_classes = 2
batch_size = 20
learning_rate = 0.001


##########################################################################################################
####################################            transformers             #################################
##########################################################################################################

# This value is for the chosen model
input_size = 224 

##########  for 3-channel models  ##########

tfms = transforms.Compose([
        #transforms.Resize(input_size), 
        transforms.Resize((input_size,input_size)), 

        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #transforms.Grayscale(num_output_channels=1),
        #transforms.Normalize([0.485], [0.229]),
        ])


##########  for single channel models  ##########

#tfms = transforms.Compose([
#        transforms.Resize(input_size), 
#        transforms.Grayscale(num_output_channels=1),
#        # Grayscale has to precede ToTensor
#        transforms.ToTensor(),
#        transforms.Normalize([0.485], [0.229]),
#        ])




##########################################################################################################
####################################             load data               #################################
##########################################################################################################

#########  Separate folder for training and vaildation data  ##########
#
#train_data_path = "CRN_CLASS_OK_FLAKE"
#
#val_data_path = "CRN_CLASS_OK_FLAKE"
#
#train_set = torchvision.datasets.ImageFolder(root=train_data_path, transform = tfms)
#
#train_set_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True,  num_workers = 0)
#
#val_set = torchvision.datasets.ImageFolder(root=val_data_path, transform = tfms)
#
#val_set_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle = True,  num_workers = 0)


##########  One folder for random split  ##########

all_image_path = "CRN"
all_data = torchvision.datasets.ImageFolder(root=all_image_path, transform = tfms)

LEN = len(all_data.imgs)
LENS = [int(round(LEN*0.7)), int(round(LEN*0.3))]

#LENS = [int(round(LEN*0.99)), int(round(LEN*0.01))]


train_set, val_set = torch.utils.data.random_split(all_data, LENS)
train_set_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True,  num_workers = 0)
val_set_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle = True,  num_workers = 0)

#L = list(val_set)
#
#len(L)
#
#len(L[0])
#
#L[0][0].shape



#for i, x in enumerate(val_set_loader):
#    print(x)
#    if i == 1:
#        break

##########################################################################################################
####################################             train                 #################################
##########################################################################################################

#model_E0 = EfficientNet_b0(2)
#model_E1 = EfficientNet_b1(2)
#model_E2 = EfficientNet_b2(2)
#model_E3 = EfficientNet_b3(2)
#model_E4 = EfficientNet_b4(2)
#model_E5 = EfficientNet_b5(2)
#model_E6 = EfficientNet_b6(2)
#model_E7 = EfficientNet_b7(2)

net = EfficientNet_b1(2)
net.fc = nn.Linear(in_features=2048, out_features=2, bias=True)

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

fit = train_1c(net, train_set_loader, val_set_loader, batch_size, optimizer, device, num_epochs)


torch.save(net, 'C:/daten/effb1_on_CRN_2')
plt.plot(fit['EPOCH'], fit['train_loss'])
plt.title("Loss")
plt.show()
plt.plot(fit['EPOCH'], fit['train_accuracy'])
plt.plot(fit['EPOCH'], fit['val_accuracy'])
plt.title("Accuracy")
plt.show()
