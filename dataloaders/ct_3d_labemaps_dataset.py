import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch
import os
import nibabel as nib
from torch.utils.data import random_split
import torchvision.transforms as transforms
from monai.transforms import Compose, RandAffine, Rotate90, RandZoom, Resize, RandSpatialCrop, Rotate90, Rand3DElastic
from math import radians as rad
from PIL import Image

SIZE_W = 256
SIZE_H = 256
class CT3DLabelmapDataset(Dataset):
    def __init__(self, params):
        self.params = params
        self.n_classes = params.n_classes
        self.complex_aumgmentation = False
        self.offline_augmented_labelmap = True
        self.human_CT = True

        self.base_folder_data_imgs = params.base_folder_data_path
        self.base_folder_data_masks = params.base_folder_mask_path
        self.labelmap_path = params.labelmap_path

        self.sub_folder_CT = [sub_f for sub_f in sorted(os.listdir(os.getcwd() + '/' + self.base_folder_data_imgs))]
        self.full_labelmap_path_imgs = [self.base_folder_data_imgs + s + self.labelmap_path for s in self.sub_folder_CT]
        self.full_labelmap_path_masks = [self.base_folder_data_masks + s + self.labelmap_path for s in self.sub_folder_CT]

        self.slice_indices, self.volume_indices, self.total_slices, self.volumes = self.read_volumes(self.full_labelmap_path_imgs)
        self.mask_slice_indices, self.mask_volume_indices, self.mask_total_slices, self.mask_volumes = self.read_volumes(self.full_labelmap_path_masks)

        self.transform_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomAffine(
                degrees=(0, 0), 
                translate=(0.1, 0),
                scale=(1, 1.4), 
                fill=1
            ),
            transforms.Resize([380, 380], transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop((SIZE_W)),
        ])

        self.transform_img_complex = Compose([
            RandAffine(
                prob=1,
                rotate_range=(
                    (rad(-45), rad(45)),
                    (rad(-180), rad(180)),
                    (rad(-15), rad(15))), # Random rotation range 30 360 10  (100, 100, 150)
                translate_range=((-70,40), (-10, 10), (-80,120)),  # Random translation range 60 10 150 # -60 at 1.5 (-80,120 liver)
                spatial_size=(SIZE_W, SIZE_H, 1),
                mode='nearest',  # Interpolation mode
            ),
            RandZoom(
                prob=1,
                min_zoom=1,
                max_zoom=1.2,
                mode='nearest',
            ),
        ])

        if self.offline_augmented_labelmap:
            self.transform_img = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomRotation(degrees=(90, 90)),  # Rotate 90 degrees counterclockwise
            ])

    def __len__(self):
        if self.params.debug:
            return self.total_slices  // 20     #reduce dataset size for debugging
        else:
            return self.total_slices 

    def read_volumes(self, full_labelmap_path):
        slice_indices = []
        volume_indices = []
        total_slices = 0
        volumes = []

        for idx, folder in enumerate(full_labelmap_path):
            # Ensure the folder path ends with a separator
            folder_path = os.path.join(folder, "")

            if self.offline_augmented_labelmap:
                # Load augmented PNGs if enabled
                png_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])

                volume_slices = []
                for png_file in png_files:
                    img_path = os.path.join(folder_path, png_file)
                    img = Image.open(img_path).convert('L')
                    img_np = np.array(img, dtype=np.int64)
                    volume_slices.append(img_np)
                
                volume = np.stack(volume_slices, axis=-1)
                volumes.append(volume)

                slice_indices.extend(np.arange(volume.shape[2]))
                volume_indices.extend(np.full(shape=volume.shape[2], fill_value=idx, dtype=np.int32))
                total_slices += volume.shape[2]

            else:
                # Find the .nii.gz file in the subdirectory
                labelmap_files = [lm for lm in sorted(os.listdir(folder_path)) if lm.endswith('.nii.gz') and "_" not in lm]
                if not labelmap_files:
                    raise FileNotFoundError(f"No valid .nii.gz files found in {folder_path}")

                # Load the volume
                labelmap = labelmap_files[0]  # Assume the first valid file is the correct one
                vol_nib = nib.load(os.path.join(folder_path, labelmap))
                vol = vol_nib.get_fdata()

                # Store slice indices and volume information
                slice_indices.extend(np.arange(vol.shape[2]))  # Append slice indices
                volume_indices.extend(np.full(shape=vol.shape[2], fill_value=idx, dtype=np.int32))  # Append the volume index
                total_slices += vol.shape[2]
                volumes.append(vol)

        return slice_indices, volume_indices, total_slices, volumes


    def preprocess(self, img, mask):
        if mask:
            img = np.where(img != self.params.pred_label, 0, 1)
        return img 
    def __getitem__(self, idx):
        if self.complex_aumgmentation:
            # Randomly sample a slice instead of using idx directly
            sampled_idx = np.random.randint(0, self.total_slices)

            vol_nr = self.volume_indices[sampled_idx]

            state = torch.get_rng_state()
            labelmap_volume = self.volumes[vol_nr]

            # Ensure the channel dimension is first, for the entire volume (not just a slice)
            if labelmap_volume.ndim == 3:  # Assuming it's 3D (H, W, D)
                labelmap_volume = np.expand_dims(labelmap_volume, axis=0)  # Add a channel dimension first

            # Apply transformations to the whole volume (EnsureChannelFirst ensures channel is first)
            labelmap_image = self.transform_img_complex(labelmap_volume)

            torch.set_rng_state(state)

            # Convert back to int64 (in case transforms modified it)
            labelmap_image = labelmap_image.to(dtype=torch.int64)

            labelmap_image = labelmap_image.squeeze(0)

            mask_image = labelmap_image

            mask_image = mask_image.permute(2, 0, 1)  # Change from [256, 256, 1] to [1, 256, 256]
            labelmap_image = labelmap_image.permute(2, 0, 1)  # Change from [256, 256, 1] to [1, 256, 256]

            mask_slice = mask_image
            labelmap_slice = labelmap_image
        else:
            vol_nr = self.volume_indices[idx]
            labelmap_slice = self.volumes[vol_nr][:, :, self.slice_indices[idx]].astype('int64')        #labelmap input to the US renderer
            if self.full_labelmap_path_imgs != self.base_folder_data_masks:
                mask_slice = self.mask_volumes[vol_nr][:, :, self.slice_indices[idx]].astype('int64')
            else:
                mask_slice = labelmap_slice.astype('int64')
            
            state = torch.get_rng_state()
            labelmap_slice = self.transform_img(labelmap_slice)
            torch.set_rng_state(state)
            mask_slice = self.transform_img(mask_slice)

        mask_slice_remaped = torch.zeros_like(mask_slice)  # Initialize with zeros

        if self.human_CT:
            mask_slice_remaped[mask_slice == 8] = 1  # MPV
            mask_slice_remaped[(mask_slice >= 12) & (mask_slice <= 16)] = 2  # LPV
            mask_slice_remaped[(mask_slice >= 9) & (mask_slice <= 11)] = 3  # RPV
            mask_slice_remaped[(mask_slice >= 4) & (mask_slice <= 7)] = 4  # HV
        else:
            mask_slice_remaped[mask_slice == 11] = 1  # MPV
            mask_slice_remaped[(mask_slice >= 12) & (mask_slice <= 15)] = 2  # LPV
            mask_slice_remaped[(mask_slice >= 16) & (mask_slice <= 20)] = 3  # RPV
            mask_slice_remaped[mask_slice == 22] = 4  # HV

        mask_slice = mask_slice_remaped

        # for spine we flip the labelmap horizontally
        if self.params.pred_label == 13:
            labelmap_slice = transforms.functional.hflip(labelmap_slice)
            mask_slice = transforms.functional.hflip(mask_slice)

        # for aorta_only
        if self.params.aorta_only:
            labelmap_slice = transforms.functional.vflip(labelmap_slice)
            mask_slice = transforms.functional.vflip(mask_slice)

        return labelmap_slice, mask_slice, str(vol_nr) + '_' + str(self.slice_indices[idx])


class CT3DLabelmapDataLoader():
    def __init__(self, params):
        super().__init__()
        self.params = params

    def get_downsampled_indices(self, dataset, ratio):
        dataset_size = len(dataset)
        subset_size = int(dataset_size * ratio)
        indices = np.random.permutation(dataset_size)[:subset_size]
        return indices

    # Create DataLoader with SubsetRandomSampler
    def get_downsampled_loader(self, dataset, batch_size, num_workers, ratio):
        indices = self.get_downsampled_indices(dataset, ratio)
        sampler = SubsetRandomSampler(indices)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
        return loader

    def train_dataloader(self):
        full_dataset = CT3DLabelmapDataset(self.params)
        
        split_ratio = 1 - 200*2 / len(full_dataset) # 20
        train_size = int(split_ratio * len(full_dataset))
        val_size = len(full_dataset) - train_size

        downsample_ratio = 800*2 / train_size  # 800

        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

        # Create DataLoader for training with downsampling
        train_loader = self.get_downsampled_loader(self.train_dataset, 
                                                   batch_size=self.params.batch_size, 
                                                   num_workers=self.params.num_workers, 
                                                   ratio=downsample_ratio)
        
        return train_loader, self.train_dataset, self.val_dataset 

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.params.batch_size, shuffle=False, num_workers=self.params.num_workers)
