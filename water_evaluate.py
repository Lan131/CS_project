import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import glob
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from albumentations import HorizontalFlip, VerticalFlip, Rotate
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.image as mpimg


class conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()
    
    def forward(self, images):
        x = self.conv1(images)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x

class encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = conv(in_channels, out_channels)
        self.pool = nn.MaxPool2d((2,2))

    def forward(self, images):
        x = self.conv(images)
        p = self.pool(x)

        return x, p

class decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.conv = conv(out_channels * 2, out_channels)

    def forward(self, images, prev):
        x = self.upconv(images)
        x = torch.cat([x, prev], axis=1)
        x = self.conv(x)

        return x

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.e1 = encoder(3, 64)
        self.e2 = encoder(64, 128)
        self.e3 = encoder(128, 256)
        self.e4 = encoder(256, 512)

        self.b = conv(512, 1024)

        self.d1 = decoder(1024, 512)
        self.d2 = decoder(512, 256)
        self.d3 = decoder(256, 128)
        self.d4 = decoder(128, 64)

        self.output = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, images):
        x1, p1 = self.e1(images)
        x2, p2 = self.e2(p1)
        x3, p3 = self.e3(p2)
        x4, p4 = self.e4(p3)

        b = self.b(p4)
        
        d1 = self.d1(b, x4)
        d2 = self.d2(d1, x3)
        d3 = self.d3(d2, x2)
        d4 = self.d4(d3, x1)

        output_mask = self.output(d4)

        return output_mask 

class LoadData_src(Dataset):
    def __init__(self, images_path):
        super().__init__()

        self.images_path = images_path
        #self.masks_path = masks_path
        self.len = len(images_path)
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1,0.1,0.1),
        ])
        

    def __getitem__(self, idx):
        img = Image.open(self.images_path[idx])
        img = self.transform(img)
        img = np.transpose(img, (2, 0, 1))
        img = img/255.0
        img = torch.tensor(img)

        return img
    
    def __len__(self):
        return self.len

size = (256, 256)
device = 'cuda'
model = UNet()
model = model.to(device)

model.load_state_dict(torch.load('./checkpoint.pth'))

image_list_train =[]
image_list_test = []
mode = 'train'
#image,mask=valid_dataset[20]
with open('src/train/training.csv', 'r') as f:
    lines = f.readlines()
    #import pdb; pdb.set_trace()
    for line in lines:
        image_path = os.path.join('src',mode, line.split(',')[0])
        image_list_train.append(image_path)

mode = 'test'
with open('src/test/testing.csv', 'r') as f:
    lines = f.readlines()
    #import pdb; pdb.set_trace()
    for line in lines:
        image_path = os.path.join(mode, line.split(',')[0])
        image_list_test.append(image_path)

train_list = sorted(glob.glob('src/train/*/*'))
test_list = sorted(glob.glob('src/test/*/*'))
#import pdb; pdb.set_trace()
train_data_src = LoadData_src(image_list_train)
test_data_src = LoadData_src(image_list_test)
# for i in range(len(train_data_src)):
#     #import pdb; pdb.set_trace()
#     image=train_data_src[1]
    
#     logits_mask=model(image.to(device, dtype=torch.float32).unsqueeze(0))
#     pred_mask0=torch.sigmoid(logits_mask)
#     pred_mask=(pred_mask0 > 0.1)*1.0

#     greater_than_05 = pred_mask > 0.9
#     count = greater_than_05.sum().item()
#     print(count)
#     print(pred_mask.sum().item())
#     if count> 39000:
#         with open('train_withwater.txt','a') as f:
#             f.write(train_list[i]+'\n')
#     else:
#         with open('train_withoutwater.txt','a') as f:
#             f.write(train_list[i]+'\n')

for i in range(len(test_data_src)):
    #import pdb; pdb.set_trace()
    image=train_data_src[1]
    
    logits_mask=model(image.to(device, dtype=torch.float32).unsqueeze(0))
    pred_mask0=torch.sigmoid(logits_mask)
    pred_mask=(pred_mask0 > 0.1)*1.0

    greater_than_05 = pred_mask > 0.9
    count = greater_than_05.sum().item()
    print(count)
    print(pred_mask.sum().item())
    if count> 39000:
        with open('test_withwater.txt','a') as f:
            f.write(train_list[i]+'\n')
    else:
        with open('test_withoutwater.txt','a') as f:
            f.write(train_list[i]+'\n')
    
# f, axarr = plt.subplots(1,3) 
# axarr[1].imshow(np.squeeze(mask.numpy()), cmap='gray')
# axarr[0].imshow(np.transpose(image.numpy(), (1,2,0)))
# axarr[2].imshow(np.transpose(pred_mask.detach().cpu().squeeze(0), (1,2,0)))
