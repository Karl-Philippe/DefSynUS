import os
import numpy as np
import cv2
from tqdm import tqdm


def run_mask_generation(dataset="phantom"):
    labels_folder = f"CT_label_maps_{dataset}/labels_warped"
    masks_folder = f"CT_label_maps_{dataset}/masks_warped"
    human_CT = (dataset == "human")

    os.makedirs(masks_folder, exist_ok=True)

    label_stats = {}

    for filename in tqdm(os.listdir(labels_folder)):
        if not filename.endswith(".png"):
            continue
        label_path = os.path.join(labels_folder, filename)
        mask_path = os.path.join(masks_folder, filename)

        label_img = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        if label_img is None:
            print(f"Warning: Could not read {label_path}")
            continue

        unique_labels, counts = np.unique(label_img, return_counts=True)
        for label, count in zip(unique_labels, counts):
            label_stats[label] = label_stats.get(label, 0) + count

        mask = np.zeros_like(label_img, dtype=np.uint8)

        if human_CT:
            mask[label_img == 8] = 1  # MPV
            mask[(label_img >= 12) & (label_img <= 16)] = 2  # LPV
            mask[(label_img >= 9) & (label_img <= 11)] = 3  # RPV
            mask[(label_img >= 4) & (label_img <= 7)] = 4  # HV
        else:
            mask[label_img == 11] = 1  # MPV
            mask[(label_img >= 12) & (label_img <= 15)] = 2  # LPV
            mask[(label_img >= 16) & (label_img <= 20)] = 3  # RPV
            mask[label_img == 22] = 4  # HV

        cv2.imwrite(mask_path, mask)

    print("Label statistics:")
    for label, count in sorted(label_stats.items()):
        print(f"Label {label}: {count} pixels")

    print("Multilabel mask generation complete.")


if __name__ == "__main__":
    run_mask_generation(dataset="phantom")
