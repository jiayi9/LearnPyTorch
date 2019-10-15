import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def load_dataset():
    data_path = 'data/train/'
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        num_workers=0,
        shuffle=True
    )
    return train_loader

# for batch_idx, (data, target) in enumerate(load_dataset()):
    #train network

data_path = "C:\LV_CHAO_IMAGE\simulation_data"
train_dataset = torchvision.datasets.ImageFolder(
    root=data_path,
    transform=torchvision.transforms.ToTensor()
)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,
    num_workers=0,
    shuffle=True
)

for i in train_loader:
    print(i)

train_loader