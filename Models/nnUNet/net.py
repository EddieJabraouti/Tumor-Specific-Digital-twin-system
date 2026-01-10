import os 
from PIL import Image, ImageFile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import accuracy_score

import h5py



ImageFile.LOAD_TRUNCATED_IMAGES = True 
allowed_exts = {'.jpg', '.png', '.nii', ',jpeg', '.tif', '.tiff', '.webp'}

#nnUnet Dynamic addaptation of network
#class Dynamic()

patch_size_3d = (128, 128, 128)
#patch_size_3d = (96, 96, 96) - if memory is tighter
batch_size = 32
learning_rate = 1e-4 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 12
num_epochs = 16


"""
Dataset Dims: 
 Input image [4, 155, 240, 240]  - Channels x Depth x Height x Width
 Masks [155, 240, 240] - Depth x height x Width


"""



class GBMDataset(Dataset): #/Users/eddiejabrouti/.cache/kagglehub/datasets/awsaf49/brats2020-training-data/versions/3
    def __init__(self, root_dir, transform = None, normalize=True, return_3d=False): 
        self.data_dir = Path(root_dir) / "BraTS2020_training_data" / "content" / "data"
        self.transform = transform
        self.normalize = normalize
        self.return_3d = return_3d

        if not self.data_dir.exists(): 
            raise ValueError(f"Data directory not found {self.data_dir}")

        print(f"Looking for data from: {self.data_dir}")

        self.load_metadata()
        self.slice_files = sorted(self.data_dir.glob("volume_*_slice_*.h5"))

        if len(self.slice_files) == 0:
            raise ValueError(f"No .h5 files found in {self.data_dir}")        
        
        print(f"Found {len(self.slice_files)} slice files")

        if self.return_3d: 
            self.volumes = self._group_slices_by_volume()
            print(f"Grouped into {len(self.volumes)} 3D volumes")
        else: 
            print(f"Using 2D slices")

    def load_metadata(self): 
        try: 
            meta_path = self.data_dir / "meta_data.csv"
            name_path = self.data_dir / "name_mapping.csv"
            survival_path = self.data_dir / "survival_info.csv"

            if meta_path.exists(): 
                self.meta_data = pd.read_csv(meta_path)
                print(f"Loaded metadata for {len(self.meta_data)} entries")

            if name_path.exists(): 
                self.name_mapping = pd.read_csv(name_path)

            if survival_path.exists(): 
                self.survival_info = pd.read_csv(survival_path)

        except Exception as e: 
            print(f"Failed to retrieve meta data from {meta_path}: {e}")

    def _group_slices_by_volume(self):
        volumes = {}
        
        for slice_file in self.slice_files:
            parts = slice_file.stem.split('_')
            volume_id = parts[1]
            slice_num = int(parts[3])
            
            if volume_id not in volumes:
                volumes[volume_id] = []
            
            volumes[volume_id].append((slice_num, slice_file))
        
        for volume_id in volumes:
            volumes[volume_id] = sorted(volumes[volume_id], key=lambda x: x[0])

        volumes_list = [(vid, slices) for vid, slices in sorted(volumes.items())]
        return volumes_list
    
    
    def __len__(self):
        if self.return_3d: 
            return len(self.volumes)
        else: 
            return len(self.slice_files)
    
    def __getitem__(self, idx): 

        """
        Returns: 
         Image: Tensor of shape(4, D, H, W) - the 4 modalities stacked
         Seg mask: Tensor of shape (D, H, W) - segmentation labels
        """
        if self.return_3d: 
            return self._get_3d_volume(idx)
        else: 
            return self._get_2d_slice(idx)
        
    def _get_2d_slice(self, idx):
        slice_file = self.slice_files[idx]
    
        try:
            with h5py.File(slice_file, 'r') as f:
                image = f['image'][:]
                mask = f['mask'][:]
            
            image = image.astype(np.float32)
            mask = mask.astype(np.int64)
            mask[mask == 4] = 3
            
            if self.normalize:
                image = self._normalize_2d(image)
            
            image = np.transpose(image, (2,0,1)) #(4, 240, 240)
            mask = torch.from_numpy(mask)
            
            if self.transform:
                image, mask = self.transform(image, mask)
            
            return image, mask
            
        except Exception as e:
            print(f"Error loading {slice_file.name}: {e}")
            return torch.zeros(4, 240, 240), torch.zeros(240, 240, dtype=torch.long)
    
    def _get_3d_volume(self, idx):
        volume_id, slices = self.volumes[idx]
        
        images = []
        masks = []
        
        for slice_num, slice_file in slices:
            try:
                with h5py.File(slice_file, 'r') as f:
                    image = f['image'][:]
                    mask = f['mask'][:]
                    
                    images.append(image)
                    masks.append(mask)
            except Exception as e:
                print(f"Error loading slice {slice_file.name}: {e}")
                continue
        
        if len(images) == 0:
            return torch.zeros(4, 155, 240, 240), torch.zeros(155, 240, 240, dtype=torch.long)
        
        image_3d = np.stack(images, axis=0)
        image_3d = np.transpose(image_3d, (3,0,1,2))

        mask_3d = np.stack(masks, axis=0)
        mask_3d = np.argmax(mask_3d, axis=-1)

        image_3d = image_3d.astype(np.float32)
        mask_3d = mask_3d.astype(np.int64)
        
        if self.normalize:
            image_3d = self._normalize_3d(image_3d)
        
        image_3d = torch.from_numpy(image_3d)
        mask_3d = torch.from_numpy(mask_3d)
        
        if self.transform:
            image_3d, mask_3d = self.transform(image_3d, mask_3d)
        
        return image_3d, mask_3d
    
    def _normalize_2d(self, image):
        for i in range(image.shape[0]):
            modality = image[i]
            brain_mask = modality > 0
            
            if brain_mask.sum() > 0:
                mean = modality[brain_mask].mean()
                std = modality[brain_mask].std()
                
                if std > 0:
                    modality[brain_mask] = (modality[brain_mask] - mean) / std
            
            image[i] = modality
        
        return image
    
    def _normalize_3d(self, image):
        for i in range(image.shape[0]):
            modality = image[i]
            brain_mask = modality > 0
            
            if brain_mask.sum() > 0:
                mean = modality[brain_mask].mean()
                std = modality[brain_mask].std()
                
                if std > 0:
                    modality[brain_mask] = (modality[brain_mask] - mean) / std
            
            image[i] = modality
        
        return image

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

        ce_loss = F.cross_entropy(logits, targets)

        probs = torch.softmax(logits, dim=1)
        num_classes = logits.shape[1]
        targets = F.one_hot(targets, num_classes)
        targets = targets.permute(0,4,1,2,3).float()


        dims = tuple(range(2, probs.ndim))
        intersection = (probs * targets).sum(dim=dims)
        union = probs.sum(dim=(2, 3, 4)) + targets.sum(dim=(2, 3, 4))

        dice_loss_per_class = (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_loss_per_class.mean() 

        total_loss = dice_loss + ce_loss

        return total_loss
#intermediary loss function for hidden layers, helps with gradient flow.
class DeepSupervisionLoss(nn.Module): 
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

class ConvBlock3D(nn.Module): 
    def __init__(self, in_channels, out_channels, norm=True): 
        super().__init__()
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size = 3, padding = 1, bias=False),
            nn.InstanceNorm3d(out_channels) if norm else nn.Identity(),
            nn.LeakyReLu(0.01, inplace=True)
        ]

        self.block = nn.sequntial(*layers)
    def forward(self, x): 
        return self.block(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, num_classes = 4, base_filters=32, deep_supervision=True):
        super().__init__()

        self.deep_supervision= deep_supervision

        #encode - Multiple Convolutional layers to downsize image and extract features
        self.enc1 = ConvBlock3D(in_channels, base_filters)
        self.enc2 = ConvBlock3D(base_filters, base_filters * 2)
        self.enc3 = ConvBlock3D(in_channels * 2, base_filters * 4)
        self.enc4 = ConvBlock3D(in_channels * 4, base_filters * 8)

        self.pool = nn.MaxPool3d(2)

        #bottleneck
        self.bottleneck = ConvBlock3D(base_filters  * 8, base_filters * 16)

        #decoder
        self.up4 = nn.ConvTranspose3d(base_filters * 16, base_filters * 8, 2, stride=2)
        self.dec4 = ConvBlock3D(base_filters * 16, base_filters * 8)

        self.up3 = nn.ConvTranspose3d(base_filters * 8, base_filters * 4, 2, stride=2)
        self.dec3 = ConvBlock3D(base_filters * 8, base_filters * 4)

        self.up2 = nn.ConvTranspose3d(base_filters * 4, base_filters * 2, 2, stride=2)
        self.dec2 = ConvBlock3D(base_filters * 4, base_filters * 2)

        self.up1 = nn.ConvTranspose3d(base_filters * 2, base_filters, 2, stride=2)
        self.dec1 = ConvBlock3D(base_filters *2, base_filters)

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


if __name__ =="__main__": 
    data_path = "/Users/eddiejabrouti/.cache/kagglehub/datasets/awsaf49/brats2020-training-data/versions/3"

    # Debug: Check what's actually in the h5 files
    import h5py
    test_file = Path(data_path) / "BraTS2020_training_data" / "content" / "data" / "volume_100_slice_0.h5"
    
    with h5py.File(test_file, 'r') as f:
        print("Keys in h5 file:", list(f.keys()))
        if 'image' in f:
            print("Image shape in file:", f['image'].shape)
        if 'mask' in f:
            print("Mask shape in file:", f['mask'].shape)
    
    dataset_3d = GBMDataset(root_dir=data_path, return_3d = True)
    print(f"\nDataset size: {len(dataset_3d)} volumes")

    dataset_2d = GBMDataset(root_dir=data_path, return_3d=False)

    image_2d, mask_2d = dataset_2d[0]

    image_3d, mask_3d = dataset_3d[0]
    print(f"Image shape: {image_3d.shape}")
    print(f"Mask shape: {mask_3d.shape}")
    print("We want: Input image [4, 155, 240, 240]  - Channels x Depth x Height x Width \n Masks [155, 240, 240] - Depth x height x Width")
    print(f"mask Unique vals: {torch.unique(mask_3d)}")
    print(f"\n 2d Image shapes {image_2d.shape}")
    print(f"\n 2d mask shapes {mask_2d.shape}")

"""

Whats left to do: 

1.) Split dataset into Train, Eval, Test
2.) Implement Data Loaders
3.) initalize model, criterion, optimizer
4.) Training loop
5.) Eval loop
6.) Train on cloud

"""