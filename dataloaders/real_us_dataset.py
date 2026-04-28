from torch.utils.data import Dataset
import torch
import os
from PIL import Image
from torch.utils.data import random_split
from torchvision import transforms  # Import transforms
import numpy as np

SIZE_W = 256
SIZE_H = 256

class RealUSDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg') or f.endswith('.png')]
        print("len(self.image_files): ", len(self.image_files))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(image_path).convert('L')

        if self.transform:
            image = self.transform(image)

        return image


class RealUSDataLoader():
    def __init__(self, param, root_dir, batch_size=1, num_workers=1):
        super().__init__()
        self.params = param
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Add random horizontal flip to transform
        self.transform = transforms.Compose([
            transforms.Resize([SIZE_W, SIZE_H], transforms.InterpolationMode.BICUBIC),
            transforms.ColorJitter(contrast=(0.9, 1.1)),  # Apply random contrast adjustment
            #transforms.RandomHorizontalFlip(p=0.5),  # 50% chance to flip horizontally
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def get_dataloaders(self):
        dataset = RealUSDataset(root_dir=self.root_dir, transform=self.transform)
       
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        return train_loader, val_loader