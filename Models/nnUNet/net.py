import os 
from PIL import Image, ImageFile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import accuracy_score

ImageFile.LOAD_TRUNCATED_IMAGES = True 
allowed_exts = {'.jpg', '.png', '.nii', ',jpeg', '.tif', '.tiff', '.webp'}

#nnUnet Dynamic addaptation of network
#class Dynamic()

class GBMDataset(Dataset): #/Users/eddiejabrouti/.cache/kagglehub/datasets/awsaf49/brats2020-training-data/versions/3
    def __init__(self, root_dir, transforms = None): 
        self.root_dir = root_dir
        self.transforms = transforms
        self.samples = []
    def process(self, data): 
        print("Data preprocessing")


class Double3DConv(nn.Module): 
    def __init__(self, in_channels, out_ channels): 
        super().__init__()


class Diceloss(nn.Module): 
    def __init__(self, smooth_scale = 1e-6): 
        self.smooth_scale = smooth_scale

    def forwardMLP(self, logits, targets): 
        probs = torch.softmax(logits, dim=1)
        targets = (torch.nn.functional.one_hot(
            targets, num_classes=probs.shape[1]
        ).permute(0,4,1,2,3).float())

        intersection = (probs * targets).sum(dim=(2, 3, 4))
        union = probs.sum(dim=(2, 3, 4)) + targets.sum(dim=(2, 3, 4))

        cross_entropy = nn.CrossEntropyLoss()
        CE_LOSS = cross_entropy(probs, targets)



        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return (1 - dice.mean()) + CE_LOSS
    
class UNet3D(nn.Module):
    def __init__(self, in_channels=1, num_classes = 4, base_filters=32):
        super().__init__()

        #encode - Multiple Convolutional layers to downsize image and extract features
        self.enc1 = DoubleConv3D(in_channels, base_filters)
        self.enc2 = DoubleConv3D(in_channels, base_filters * 2)
        self.enc3 = DoubleConv3D(in_channels * 2, base_filters * 4)
        self.enc4 = DoubleConv3D(in_channels * 4, base_filters * 8)

        self.pool = nn.MaxPool3d(2)

        #bottleneck
        self.bottleneck = DoubleConv3D(base_filters  * 8, base_filters * 16)

        #decoder
        self.up4 = nn.ConvTranspose3d(base_filters * 16, base_filters * 8, 2, stride=2)
        self.dec4 = DoubleConv3D(base_filters * 16, base_filters * 8)

        self.up3 = nn.ConvTranspose3d(base_filters * 8, base_filters * 4, 2, stride=2)
        self.dec3 = DoubleConv3D(base_filters * 8, base_filters * 4)

        self.up2 = nn.ConvTranspose3d(base_filters * 4, base_filters * 2, 2, stride=2)
        self.dec2 = DoubleConv3D(base_filters * 4, base_filters * 2)

        self.up1 = nn.ConvTranspose3d(base_filters * 2, base_filters, 2, stride=2)
        self.dec1 = DoubleConv3D(base_filters *2, base_filters)

        self.out = nn.Conv3d(base_filters, num_classes, kernel_size=1)

    # concatenating feature map from downsampling convolution to upsampled feature map (Maintain classification and segmentation context)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(x)
        e3 = self.enc3(x)
        e4 = self.enc4(x)

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up4(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up4(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up4(b)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)
