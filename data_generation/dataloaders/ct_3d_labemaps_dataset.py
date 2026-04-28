import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch
import os
import nibabel as nib
from torch.utils.data import random_split
import torchvision.transforms as transforms
from monai.transforms import Compose, RandAffine, Rotate90, RandZoom, Resize, RandSpatialCrop, Rand3DElastic, RandFlip
from math import radians as rad
from PIL import Image

SIZE_W = 256
SIZE_H = 256

import gzip
import shutil

CACHE_DIR = "nii_cache"  # you can move this path to your params if you want

def ensure_cached_nii(nii_gz_path):
    """
    Check if an uncompressed .nii version exists in CACHE_DIR.
    If not, decompress it once and return the path to the .nii file.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    filename = os.path.basename(nii_gz_path).replace('.nii.gz', '.nii')
    cached_path = os.path.join(CACHE_DIR, filename)

    if not os.path.exists(cached_path):
        print(f"Decompressing and caching {nii_gz_path} → {cached_path}")
        with gzip.open(nii_gz_path, 'rb') as f_in:
            with open(cached_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    return cached_path

def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)
class CT3DLabelmapDataset(Dataset):
    def __init__(self, params):
        self.params = params
        self.n_classes = params.n_classes
        self.complex_aumgmentation = True
        self.offline_augmented_labelmap = False
        self.deformed_human = False
        self.basic_aumgmentation = False

        self.base_folder_data_imgs = params.base_folder_data_path
        self.base_folder_data_masks = params.base_folder_mask_path
        self.labelmap_path = params.labelmap_path

        self.sub_folder_CT = [sub_f for sub_f in sorted(os.listdir(os.getcwd() + '/' + self.base_folder_data_imgs))]
        self.full_labelmap_path_imgs = [self.base_folder_data_imgs + s + self.labelmap_path for s in self.sub_folder_CT]
        self.full_labelmap_path_masks = [self.base_folder_data_masks + s + self.labelmap_path for s in self.sub_folder_CT]

        self.slice_indices, self.volume_indices, self.total_slices, self.volumes = self.read_volumes(self.full_labelmap_path_imgs)
        self.mask_slice_indices, self.mask_volume_indices, self.mask_total_slices, self.mask_volumes = self.read_volumes(self.full_labelmap_path_masks)

        # Synthetic dataset length to support generating a fixed number of augmented samples
        # If generate_us_dataset is set with us_dataset_size, use that size; else fall back to total_slices
        self.synthetic_length = getattr(self.params, 'us_dataset_size', None) if getattr(self.params, 'generate_us_dataset', False) else None
        if not self.synthetic_length:
            self.synthetic_length = self.total_slices

        self.transform_img = transforms.Compose([
            transforms.ToTensor(),
            #transforms.RandomAffine(
            #    degrees=(0, 0), 
            #    translate=(0.1, 0),
            #    scale=(1, 1.4), 
            #    fill=1
            #),
            transforms.Resize([380, 380], transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop((SIZE_W)),
        ])

        self.transform_img_complex = Compose([
            Rotate90(
                k=3,
                spatial_axes=(0, 1)  # Rotate in the plane of H and W
            ),

            RandAffine(
                prob=1,
                rotate_range=(
                    rad(20), # 20 #! 45
                    rad(45), # 45 #! 180 
                    rad(5)), # 5 #! 15 # Random rotation range 30 360 10  (100, 100, 150)
                translate_range=((-10, 10), (-10, 80), (-70, 120)), # (-10,10), (-30, 80), (-90, 120)
                #translate_range=((-70,40), (-10, 10), (-80,120)),  # Random translation range 60 10 150 # -60 at 1.5 (-80,120 liver)
                spatial_size=(SIZE_W, SIZE_H, 1),
                mode='nearest',  # Interpolation mode
            ),

            #Rand3DElastic(
            #    prob=1,
            #    sigma_range=(14, 16),
            #    magnitude_range=(1000, 3000),
            #    padding_mode='reflection',
            #    mode="nearest", 
            #),

            RandZoom(
                prob=1,
                min_zoom=1,
                max_zoom=1.1,
                mode='nearest',
            ),

            RandFlip(
                prob=0.5,
                spatial_axis=1,
            )
        ])

        if self.deformed_human:
            self.transform_img_complex = Compose([

                Rotate90(
                    k=1,
                    spatial_axes=(1, 2)  # Rotate in the plane of H and W
                ),

                Rotate90(
                    k=1,
                    spatial_axes=(0, 1)  # Rotate in the plane of H and W
                ),

                RandAffine(
                    prob=1,
                    rotate_range=(
                        rad(20), # 20 #! 45
                        rad(45), # 45 #! 180 
                        rad(5)), # 5 #! 15 # Random rotation range 30 360 10  (100, 100, 150)
                    translate_range=((-30, 30), (-50, 50), (-60, 60)), # (-10,10), (-20, 30), (-50, 60)
                    #translate_range=((-70,40), (-10, 10), (-80,120)),  # Random translation range 60 10 150 # -60 at 1.5 (-80,120 liver)
                    spatial_size=(SIZE_W, SIZE_H, 1),
                    mode='nearest',  # Interpolation mode
                ),

                #Rand3DElastic(
                #    prob=1,
                #    sigma_range=(14, 16),
                #    magnitude_range=(1000, 3000),
                #    padding_mode='reflection',
                #    mode="nearest", 
                #),

                RandZoom(
                    prob=1,
                    min_zoom=1.2,
                    max_zoom=1.6,
                    mode='nearest',
                ),

                RandFlip(
                    prob=0.5,
                    spatial_axis=1,
                )
            ])

        if self.basic_aumgmentation:
            self.transform_img_complex = Compose([

                Rotate90(
                    k=1,
                    spatial_axes=(1, 2)  # Rotate in the plane of H and W
                ),

                Rotate90(
                    k=1,
                    spatial_axes=(0, 1)  # Rotate in the plane of H and W
                ),

                RandAffine(
                    prob=1,
                    rotate_range=(
                        rad(0), # 20 #! 45
                        rad(0), # 45 #! 180 
                        rad(0)), # 5 #! 15 # Random rotation range 30 360 10  (100, 100, 150)
                    translate_range=((0, 0), (0, 0), (-80, 80)), # (-10,10), (-20, 30), (-50, 60)
                    #translate_range=((-70,40), (-10, 10), (-80,120)),  # Random translation range 60 10 150 # -60 at 1.5 (-80,120 liver)
                    spatial_size=(SIZE_W, SIZE_H, 1),
                    mode='nearest',  # Interpolation mode
                ),

                #Rand3DElastic(
                #    prob=1,
                #    sigma_range=(14, 16),
                #    magnitude_range=(1000, 3000),
                #    padding_mode='reflection',
                #    mode="nearest", 
                #),

                RandZoom(
                    prob=1,
                    min_zoom=1.0,
                    max_zoom=1.4,
                    mode='nearest',
                ),

                RandFlip(
                    prob=0.5,
                    spatial_axis=1,
                )
            ])

        if self.offline_augmented_labelmap:
            self.transform_img = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomRotation(degrees=(90, 90)),  # Rotate 90 degrees counterclockwise
            ])

    def __len__(self):
        if self.params.debug:
            return self.synthetic_length // 20     #reduce dataset size for debugging
        # When complex augmentation is enabled and a synthetic size is requested, use it
        if self.complex_aumgmentation and getattr(self.params, 'generate_us_dataset', False):
            return int(self.synthetic_length)
        return self.synthetic_length

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
                labelmap_files = [lm for lm in sorted(os.listdir(folder_path)) if lm.endswith('.nii.gz')]
                if not labelmap_files:
                    raise FileNotFoundError(f"No valid .nii.gz files found in {folder_path}")

                volume_group_paths = []
                for lm in labelmap_files:
                    gz_path = os.path.join(folder_path, lm)
                    #nii_path = ensure_cached_nii(gz_path)
                    volume_group_paths.append(gz_path)

                volumes.append(volume_group_paths)

                # Load one reference volume just to determine the shape
                ref_vol_nib = nib.load(volume_group_paths[0])
                ref_vol = ref_vol_nib.get_fdata()
                slice_indices.extend(np.arange(ref_vol.shape[2]))
                volume_indices.extend(np.full(shape=ref_vol.shape[2], fill_value=idx, dtype=np.int32))
                total_slices += ref_vol.shape[2]


        return slice_indices, volume_indices, total_slices, volumes


    def preprocess(self, img, mask):
        if mask:
            img = np.where(img != self.params.pred_label, 0, 1)
        return img 

    def __getitem__(self, idx):
        if self.complex_aumgmentation:
            # Deterministically map dataset index to a source slice; augmentation randomness provides diversity
            # Randomly select a folder (patient/study)
            vol_nr = np.random.randint(0, len(self.volumes))
            volume_group_paths = self.volumes[vol_nr]

            # Randomly select one .nii.gz file from that folder
            random_volume_path = random.choice(volume_group_paths)

            # Load the chosen volume on the fly
            vol_nib = nib.load(random_volume_path)
            labelmap_volume = np.rint(vol_nib.get_fdata()).astype(np.int64)

            state = torch.get_rng_state()

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
        mask_slice_remaped[mask_slice == 11] = 1  # MPV
        mask_slice_remaped[(mask_slice >= 12) & (mask_slice <= 15)] = 2  # LPV
        mask_slice_remaped[(mask_slice >= 16) & (mask_slice <= 20)] = 3  # RPV
        mask_slice_remaped[mask_slice == 21] = 4  # HV

        mask_slice = mask_slice_remaped

        # for spine we flip the labelmap horizontally
        if self.params.pred_label == 13:
            labelmap_slice = transforms.functional.hflip(labelmap_slice)
            mask_slice = transforms.functional.hflip(mask_slice)

        # for aorta_only
        if self.params.aorta_only:
            labelmap_slice = transforms.functional.vflip(labelmap_slice)
            mask_slice = transforms.functional.vflip(mask_slice)

        return labelmap_slice, mask_slice, str(vol_nr) + '_' + str(self.slice_indices[idx % self.total_slices])


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
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            worker_init_fn=_seed_worker,
            persistent_workers=(num_workers > 0),
            pin_memory=True,
            prefetch_factor=2 if num_workers > 0 else None,
        )
        return loader

    def train_dataloader(self):
        full_dataset = CT3DLabelmapDataset(self.params)
        
        split_ratio = 0
        train_size = int(split_ratio * len(full_dataset))
        val_size = len(full_dataset) - train_size

        downsample_ratio = train_size  

        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

        # Create DataLoader for training with downsampling
        train_loader = self.get_downsampled_loader(self.train_dataset, 
                                                   batch_size=self.params.batch_size, 
                                                   num_workers=self.params.num_workers, 
                                                   ratio=downsample_ratio)
        
        return train_loader, self.train_dataset, self.val_dataset 

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.params.batch_size,
            shuffle=False,
            num_workers=self.params.num_workers,
            worker_init_fn=_seed_worker,
            persistent_workers=(self.params.num_workers > 0),
            pin_memory=True,
            prefetch_factor=2 if self.params.num_workers > 0 else None,
        )
