import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import os
from PIL import Image
from torch.utils.data import random_split
import torchvision.transforms as transforms

SIZE_W = 256
SIZE_H = 256

# Function to create fan mask
def create_fan_mask(resultWidth=290, resultHeight=200, centerX=290/2, centerY=-50, 
                    minAngle=-np.pi * 35 / 180, maxAngle=np.pi * 35 / 180,
                    minRadius=61, maxRadius=250):
    """
    Creates a binary mask corresponding to the fan-shaped valid region.
    """
    # Create coordinate grid
    x = torch.linspace(0, resultWidth - 1, resultWidth) - centerX
    y = torch.linspace(0, resultHeight - 1, resultHeight) - centerY
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    # Compute angle and radius
    angle = torch.atan2(xx, yy)
    radius = torch.sqrt(xx ** 2 + yy ** 2)

    # Create binary mask where valid pixels are 1
    angle_mask = (angle > minAngle) & (angle < maxAngle)
    radius_mask = (radius > minRadius) & (radius < maxRadius)
    
    fan_mask = (angle_mask & radius_mask).float()  # Convert to float tensor

    return fan_mask.unsqueeze(0).unsqueeze(0)  # Shape [1, 1, H, W]

class RealUSGTDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir_imgs = root_dir + 'imgs/'
        self.root_dir_masks = root_dir + 'masks/'
        self.transform_img = transforms.Compose([
            transforms.Resize([SIZE_W, SIZE_H], transforms.InterpolationMode.BICUBIC),
            #transforms.RandomHorizontalFlip(p=0.0),  # Flip image horizontally
            transforms.Grayscale(1),
            #transforms.ColorJitter(contrast=(1, 2)),  # Apply random contrast adjustment
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        self.transform_mask = transforms.Compose([
            transforms.Resize([SIZE_W, SIZE_H], transforms.InterpolationMode.NEAREST),
            #transforms.RandomHorizontalFlip(p=0.0),  # Flip mask horizontally
            transforms.PILToTensor()  # Keeps original pixel values
        ])

        self.image_files = [f for f in sorted(os.listdir(os.getcwd() + '/' + self.root_dir_imgs)) if f.endswith('.jpg') or f.endswith('.png')]
        self.masks_files = [f for f in sorted(os.listdir(self.root_dir_masks)) if f.endswith('.jpg') or f.endswith('.png')]
        print("len(self.image_files): ", len(self.image_files))

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir_imgs, self.image_files[idx])
        image = Image.open(image_path)
        image = self.transform_img(image)

        mask_path = os.path.join(self.root_dir_masks, self.masks_files[idx])
        mask = Image.open(mask_path)  # Keep it as PIL Image

        mask = self.transform_mask(mask)  # Now, this will work since mask is PIL Image

        # Apply the fan mask
        fan_mask = create_fan_mask()
        fan_mask = torch.nn.functional.interpolate(fan_mask, size=(256, 256), mode='bilinear', align_corners=True)

        # Combine the fan mask with the GT mask (e.g., element-wise multiplication)
        mask = mask * fan_mask.squeeze(0).long()

        print("test")
        
        return image, mask

class RealUSGTDataLoader():
    def __init__(self, param, root_dir, batch_size=1, transform=None, num_workers=1):
        super().__init__()
        self.params = param
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.transform = transform
        self.num_workers = num_workers

    def get_dataloaders(self):
        dataset = RealUSGTDataset(root_dir=self.root_dir, transform=self.transform)
       
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


        return train_loader, val_loader