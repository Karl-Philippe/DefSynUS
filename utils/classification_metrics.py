# utils/island_metrics.py

import numpy as np
from skimage import measure
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

import torch

def convert_to_one_hot(label_img, num_labels=5):
    """
    Convert a 2D label image to a 3D one-hot encoded tensor.

    Args:
        label_img (torch.Tensor): Input label image with shape [H, W] where pixel values range from 0 to num_labels-1.
        num_labels (int): Number of unique labels.

    Returns:
        torch.Tensor: One-hot encoded tensor with shape [num_labels, H, W].
    """
    # Initialize a tensor of zeros with shape (num_labels, H, W)
    one_hot = torch.zeros((num_labels, label_img.shape[0], label_img.shape[1]), dtype=torch.uint8)

    label_img = torch.tensor(label_img) if not isinstance(label_img, torch.Tensor) else label_img

    # Set the corresponding index for each label to 1
    for label in range(num_labels):
        one_hot[label, :, :] = (label_img == label).long()  # Use .long() to convert to the correct dtype

    return one_hot

def save_masks(gt_mask, pred_mask, label, file_path="masks"):
    """
    Saves the ground truth (GT) mask and predicted mask for a specific batch and label.

    Args:
        gt_mask (np.ndarray): Ground truth mask.
        pred_mask (np.ndarray): Predicted mask.
        label (int): The label for the current class.
        file_path (str): Directory path to save the mask images.

    Returns:
        None
    """
    # Create figure with 2 subplots: one for GT mask and one for Predicted mask
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Plot predicted mask with Viridis colormap
    ax[0].imshow(pred_mask, cmap='viridis', interpolation='none')
    ax[0].set_title(f"Predicted - Label {label}")
    ax[0].axis('off')  # Hide axes

    # Plot ground truth mask with Viridis colormap
    ax[1].imshow(gt_mask.squeeze(), cmap='viridis', interpolation='none')
    ax[1].set_title(f"Ground Truth - Label {label}")
    ax[1].axis('off')  # Hide axes

    # Save the figure
    plt.tight_layout()
    mask_filename = f"{file_path}/label_{label}.png"
    plt.savefig(mask_filename)
    plt.close()

def save_classification_metrics(metrics, file_path="metrics.txt"):
    with open(file_path, "w") as f:
        f.write("Classification Metrics:\n")
        
        # Mean per label
        f.write("\nMean per Label:\n")
        for label, values in metrics['mean_per_label'].items():
            f.write(f"  Label {label}:\n")
            for metric, value in values.items():
                std_value = metrics['std_per_label'][label].get(metric, 0)
                f.write(f"    {metric}: {round(value, 2)} ± {round(std_value, 2)}\n")

        # Overall mean
        f.write("\nOverall Mean:\n")
        for metric, value in metrics['overall_mean'].items():
            std_value = metrics['overall_std'].get(metric, 0)
            f.write(f"  {metric}: {round(value, 2)} ± {round(std_value, 2)}\n")

def calculate_classification_metrics(pred_masks, gt_masks, num_classes=5, size_threshold=50, result_file="results_test/metrics.txt"):
    """
    Calculate TP, FP, FN, Precision, and Recall for each label, then compute mean and std.

    Args:
        pred_masks (np.ndarray): Predicted masks of shape [B, C, H, W].
        gt_masks (np.ndarray): Ground truth masks of shape [B, H, W] or [B, 1, H, W].
        num_classes (int): Number of classes (including background).
        size_threshold (int): Minimum size of islands to be considered.
        result_file (str): Path to the file where results will be saved.

    Returns:
        dict: Mean and std of TP, FP, FN, Precision, and Recall for each label (1 to num_classes-1).
    """
    # Check shape and convert gt_masks to one-hot if necessary

    
    pred_masks = np.squeeze(pred_masks)
    gt_masks = np.squeeze(gt_masks)

    pred_masks = pred_masks.argmax(axis=0)

    pred_masks = convert_to_one_hot(pred_masks, num_labels=5)
    gt_masks = convert_to_one_hot(gt_masks, num_labels=5)

    if gt_masks.ndim == 3:  # Shape [B, H, W]
        gt_masks = np.expand_dims(gt_masks, axis=1)  # Convert to [B, 1, H, W]

    # Store metrics for all labels (1 to num_classes-1)
    all_metrics = {label: {'TP': [], 'FP': [], 'FN': [], 'Precision': [], 'Recall': []} 
                   for label in range(1, num_classes)}
    
    # Open the result file to save the results
    with open(result_file, "w") as f:
        # Iterate over each label (ignoring background)
        for label in range(1, num_classes):
            pred_mask = pred_masks[label]
            gt_mask = gt_masks[label]

            # Apply binary dilation with a 6x6 kernel
            gt_mask = np.squeeze(gt_mask)
            gt_mask = binary_dilation(gt_mask, structure=np.ones((9, 9))).astype(int)

            # Before size filtering: count islands
            pred_labels_before = measure.label(pred_mask.cpu())
            gt_labels_before = measure.label(gt_mask)
            num_pred_islands_before = len(np.unique(pred_labels_before)) - 1  # Exclude background
            num_gt_islands_before = len(np.unique(gt_labels_before)) - 1  # Exclude background

            # Remove small islands by size thresholding
            pred_mask = measure.label(pred_mask.cpu())
            for region in measure.regionprops(pred_mask):
                if region.area < size_threshold:
                    pred_mask[pred_mask == region.label] = 0
            pred_mask = pred_mask > 0

            # After size filtering: count islands
            pred_labels_after = measure.label(pred_mask)
            gt_labels_after = measure.label(gt_mask)
            num_pred_islands_after = len(np.unique(pred_labels_after)) - 1  # Exclude background
            num_gt_islands_after = len(np.unique(gt_labels_after)) - 1  # Exclude background

            # If no predictions and no ground truth, set metrics to NaN
            if np.sum(pred_mask) == 0 and np.sum(gt_mask) == 0:
                f.write(f"Label {label} - TP: Nan, FP: Nan, FN: Nan, Precision: Nan, Recall: Nan\n")
                all_metrics[label]['TP'].append(np.nan)
                all_metrics[label]['FP'].append(np.nan)
                all_metrics[label]['FN'].append(np.nan)
                all_metrics[label]['Precision'].append(np.nan)
                all_metrics[label]['Recall'].append(np.nan)
                continue

            # Write number of islands before filtering
            f.write(f"  Label {label} - Filtering - Pred: {num_pred_islands_before} -> {num_pred_islands_after} , GT: {num_gt_islands_before} -> {num_gt_islands_after}\n")


            # Initialize counts for TP, FP, FN
            TP, FP, FN = 0, 0, 0
            
            # True Positives and False Positives
            for pred_label in range(1, pred_labels_after.max() + 1):
                pred_island = (pred_labels_after == pred_label)
                overlap = pred_island & (gt_labels_after > 0)
                if np.any(overlap):
                    TP += 1
                else:
                    FP += 1
            
            # False Negatives
            for gt_label in range(1, gt_labels_after.max() + 1):
                gt_island = (gt_labels_after == gt_label)
                overlap = gt_island & (pred_labels_after > 0)
                if not np.any(overlap):
                    FN += 1
            
            # Calculate Precision and Recall
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            
            # Write metrics for this batch and label
            f.write(f"Label {label} - TP: {TP}, FP: {FP}, FN: {FN}, Precision: {precision:.2f}, Recall: {recall:.2f}\n")
            
            # Store metrics for this batch and label
            all_metrics[label]['TP'].append(TP)
            all_metrics[label]['FP'].append(FP)
            all_metrics[label]['FN'].append(FN)
            all_metrics[label]['Precision'].append(precision)
            all_metrics[label]['Recall'].append(recall)

            save_masks(gt_mask, pred_mask, label, file_path="results_test/results_masks")

        # Calculate mean and std for each label and overall
        mean_metrics = {}
        std_metrics = {}
        for label in range(1, num_classes):
            mean_metrics[label] = {metric: np.nanmean(all_metrics[label][metric]) 
                                   for metric in ['TP', 'FP', 'FN', 'Precision', 'Recall']}
            std_metrics[label] = {metric: np.nanstd(all_metrics[label][metric]) 
                                  for metric in ['TP', 'FP', 'FN', 'Precision', 'Recall']}
        
        # Calculate overall mean and std across all labels
        overall_mean = {metric: np.nanmean([mean_metrics[label][metric] 
                                         for label in range(1, num_classes)]) 
                        for metric in ['TP', 'FP', 'FN', 'Precision', 'Recall']}
        overall_std = {metric: np.nanstd([mean_metrics[label][metric] 
                                       for label in range(1, num_classes)]) 
                       for metric in ['TP', 'FP', 'FN', 'Precision', 'Recall']}
        
        metrics = {'mean_per_label': mean_metrics, 'std_per_label': std_metrics,
                   'overall_mean': overall_mean, 'overall_std': overall_std}

        # Call the function to save the metrics in the result file
        save_classification_metrics(metrics, result_file)

    return metrics