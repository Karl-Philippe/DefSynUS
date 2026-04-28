import os
import shutil
import re
from sklearn.model_selection import train_test_split

def create_split_folders(base_dir, subfolders):
    for subfolder in subfolders:
        os.makedirs(os.path.join(base_dir, subfolder), exist_ok=True)

def copy_files(img_files, mask_files, dest_img_dir, dest_mask_dir=None):
    for i, img_file in enumerate(img_files):
        shutil.copy(img_file, os.path.join(dest_img_dir, os.path.basename(img_file)))
        if mask_files and dest_mask_dir:
            shutil.copy(mask_files[i], os.path.join(dest_mask_dir, os.path.basename(mask_files[i])))

def extract_id(filename):
    """
    Extract numeric part of filename as integer to allow robust matching
    (ignores leading zeros).
    """
    digits = ''.join(ch for ch in os.path.basename(filename) if ch.isdigit())
    return int(digits) if digits else None

def split_data(img_dir, mask_dir, split_ratio):
    img_files = sorted([
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if f.endswith(('.png', '.jpg', '.jpeg', '.nii.gz'))
    ])
    mask_files = sorted([
        os.path.join(mask_dir, f)
        for f in os.listdir(mask_dir)
        if f.endswith(('.png', '.jpg', '.jpeg', '.nii.gz'))
    ])

    print(f"Total images: {len(img_files)}, Total masks: {len(mask_files)}")

    # Build dictionary: key = numeric ID, value = mask file path
    mask_dict = {extract_id(f): f for f in mask_files}

    # Match masks to images based on numeric ID
    matched_masks = []
    valid_img_files = []
    for img in img_files:
        img_id = extract_id(img)
        mask_file = mask_dict.get(img_id)
        if mask_file:
            matched_masks.append(mask_file)
            valid_img_files.append(img)
        else:
            print(f"Warning: No mask found for {os.path.basename(img)}")

    if not matched_masks:
        print("Error: No matches found between images and masks.")
        return

    total_files = len(valid_img_files)
    total_ratio = sum(split_ratio)
    trainA_size = int((split_ratio[0] / total_ratio) * total_files)
    test_size = int((split_ratio[1] / total_ratio) * total_files)
    validation_size = total_files - trainA_size - test_size

    trainA_split, temp_split, trainA_masks, temp_masks = train_test_split(
        valid_img_files, matched_masks, train_size=trainA_size, random_state=42
    )

    test_split, validation_split, test_masks, validation_masks = train_test_split(
        temp_split, temp_masks,
        test_size=validation_size / (test_size + validation_size),
        random_state=42
    )

    # Create the target directories
    create_split_folders('datasets/human_baseline_trainA_500', ['imgs', 'masks'])
    create_split_folders('datasets/human_baseline_testing_100', ['imgs', 'masks'])
    create_split_folders('datasets/human_baseline_stopp_crit', ['imgs', 'masks'])

    # Copy files
    copy_files(trainA_split, None, 'datasets/human_baseline_trainA_500/imgs')
    copy_files(test_split, test_masks,
               'datasets/human_baseline_testing_100/imgs',
               'datasets/human_baseline_testing_100/masks')
    copy_files(validation_split, validation_masks,
               'datasets/human_baseline_stopp_crit/imgs',
               'datasets/human_baseline_stopp_crit/masks')

    print(f"Data split completed. Train: {len(trainA_split)}, Test: {len(test_split)}, Validation: {len(validation_split)}")


if __name__ == '__main__':
    img_dir = 'datasets/all_data_human_baseline/imgs'
    mask_dir = 'datasets/all_data_human_baseline/masks'
    split_ratio = [500, 100, 10]
    split_data(img_dir, mask_dir, split_ratio)
