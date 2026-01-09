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

patch_size_3d = (128, 128, 128)

#patch_size_3d = (96, 96, 96) - if memory is tighter


class GBMDataset(Dataset): #/Users/eddiejabrouti/.cache/kagglehub/datasets/awsaf49/brats2020-training-data/versions/3
    def __init__(self, root_dir, transforms = None): 
        self.root_dir = root_dir
        self.transforms = transforms
        self.samples = []
    def process(self, data): 
        print("Data preprocessing")


class Double3DConv(nn.Module): 
    def __init__(self, in_channels, out_channels): 
        super().__init__()


class Diceloss(nn.Module): 
    def __init__(self, smooth= 1e-6): 
        super(Diceloss, self).__init__()
        self.smooth = smooth

    

    def forward(self, logits, targets): 
        """
        Args:
            logits: raw network output (B, C, [D,] H, W)
            targets: ground truth labels (B, [D,] H, W) with class indices
        """
    
        probs = torch.softmax(logits, dim=1)
        num_classes = logits.shape[1]
        targets = torch.nn.functional.one_hot(
           targets, num_classes = num_classes
        )

        if targets.ndim == 4: 
            targets.permute(0, 3, 1, 2).float()

        else: 
            targets.permute(0,4,1,2,3).float()

        dims = tuple(range(2, probs.ndim))
        intersection = (probs * targets).sum(dim=dims)
        union = probs.sum(dim=(2, 3, 4)) + targets.sum(dim=(2, 3, 4))

        dice_loss_per_class = (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_loss_per_class.mean() 


        ce_loss = nn.functional.cross_entropy(logits, targets)

        total_loss = dice_loss + ce_loss

        return total_loss
#intermediary loss function for hidden layers, helps with gradient flow.
class DeepSupervisionLoss(nn.module): 
    def __init__(self, num_outputs = 5):
        super().__init__()
        self.weights = [1.0/(2**i) for i in range(num_outputs)]

        total = sum(self.weights)
        self.weights = (w / total for w in self.weights) #normalize

        self.dice_ce = Diceloss() 


    def forward(self, output, targets): 
        """
        Args:
            outputs: list of predictions at different scales
                     [full_res, 1/2_res, 1/4_res, 1/8_res, 1/16_res]
            targets: ground truth at full resolution
        """
        loss = 0
        for i , (output, weight) in enumerate(zip(output, self.weights)): 
            if i == 0: 
                downsampled_target = targets
            else: 
                downsampled_target = F.interpolate(
                    targets.unsqueeze(1).float(), 
                    size = output.shape[2:],
                    mode ='nearest'
                ).squeeze(1).long()

            loss += weight * self.dice_ce(output, downsampled_target)

        return loss


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, num_classes = 4, base_filters=32, deep_supervision=True):
        super().__init__()

        self.deep_supervision= deep_supervision

        #encode - Multiple Convolutional layers to downsize image and extract features
        self.enc1 = DoubleConv3D(in_channels, base_filters)
        self.enc2 = DoubleConv3D(base_filters, base_filters * 2)
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

        if deep_supervision: 
            self.ds_out4 = nn.Conv3d(base_filters * 8, num_classes, kernel_size = 1)
            self.ds_out3 = nn.Conv3d(base_filters * 4, num_classes, kernel_size = 1)
            self.ds_out2 = nn.Conv3d(base_filters * 2, num_classes, kernel_size = 1)
            self.ds_out1 = nn.Conv3d(base_filters, num_classes, kernel_size = 1)
            

    # concatenating feature map from downsampling convolution to upsampled feature map (Maintain classification and segmentation context)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(b)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)
