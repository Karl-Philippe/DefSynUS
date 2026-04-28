import numpy as np
from skimage import measure
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

def convert_to_one_hot(label_img, num_labels=2):
    """
    Convert a 2D label image to a 3D one-hot encoded tensor.
    Args:
        label_img (torch.Tensor): Input label image with shape [H, W].
        num_labels (int): Number of unique labels.
    Returns:
        torch.Tensor: One-hot encoded tensor with shape [num_labels, H, W].
    """
    one_hot = torch.zeros((num_labels, label_img.shape[0], label_img.shape[1]), dtype=torch.uint8)
    for label in range(num_labels):
        one_hot[label, :, :] = (label_img == label).long()
    return one_hot

def remap_labels(label_img):
    """
    Remap the labels to:
    - 1, 2, 3 (MPV, RPV, LPV) -> 1 (PV)
    - 4 (HV) -> 2 (HV)
    """
    remapped = torch.zeros_like(label_img)
    remapped[(label_img == 1) | (label_img == 2) | (label_img == 3)] = 1  # PV
    remapped[label_img == 4] = 2  # HV
    return remapped

def calculate_classification_metrics_PV_HV(pred_masks, gt_masks, num_classes=2, size_threshold=50, result_file="results_test/metrics.txt"):
    """
    Calculate TP, FP, FN, Precision, and Recall for PV vs HV classification.
    Args:
        pred_masks (np.ndarray): Predicted masks of shape [B, C, H, W].
        gt_masks (np.ndarray): Ground truth masks of shape [B, H, W] or [B, 1, H, W].
        num_classes (int): Number of classes (2: PV and HV).
        size_threshold (int): Minimum size of islands to be considered.
        result_file (str): Path to the file where results will be saved.
    Returns:
        dict: Mean and std of TP, FP, FN, Precision, and Recall for each label.
    """
    pred_masks = np.squeeze(pred_masks)
    gt_masks = np.squeeze(gt_masks)
    pred_masks = pred_masks.argmax(axis=0)

    # Remap labels for PV vs HV classification
    pred_masks = remap_labels(torch.tensor(pred_masks).cpu()).numpy()
    gt_masks = remap_labels(torch.tensor(gt_masks).cpu()).numpy()

    pred_masks = convert_to_one_hot(torch.tensor(pred_masks), num_labels=2).numpy()
    gt_masks = convert_to_one_hot(torch.tensor(gt_masks), num_labels=2).numpy()

    all_metrics = {label: {'TP': [], 'FP': [], 'FN': [], 'Precision': [], 'Recall': []} 
                   for label in range(1, num_classes+1)}
    
    with open(result_file, "w") as f:
        for label in range(1, num_classes+1):
            pred_mask = pred_masks[label-1]
            gt_mask = gt_masks[label-1]

            gt_mask = binary_dilation(gt_mask, structure=np.ones((9, 9))).astype(int)

            pred_labels_after = measure.label(pred_mask)
            gt_labels_after = measure.label(gt_mask)

            TP, FP, FN = 0, 0, 0
            
            for pred_label in range(1, pred_labels_after.max() + 1):
                pred_island = (pred_labels_after == pred_label)
                overlap = pred_island & (gt_labels_after > 0)
                if np.any(overlap):
                    TP += 1
                else:
                    FP += 1
            
            for gt_label in range(1, gt_labels_after.max() + 1):
                gt_island = (gt_labels_after == gt_label)
                overlap = gt_island & (pred_labels_after > 0)
                if not np.any(overlap):
                    FN += 1
            
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            
            f.write(f"Label {label} - TP: {TP}, FP: {FP}, FN: {FN}, Precision: {precision:.2f}, Recall: {recall:.2f}\n")
            
            all_metrics[label]['TP'].append(TP)
            all_metrics[label]['FP'].append(FP)
            all_metrics[label]['FN'].append(FN)
            all_metrics[label]['Precision'].append(precision)
            all_metrics[label]['Recall'].append(recall)

        mean_metrics = {}
        std_metrics = {}
        for label in range(1, num_classes+1):
            mean_metrics[label] = {metric: np.nanmean(all_metrics[label][metric]) 
                                   for metric in ['TP', 'FP', 'FN', 'Precision', 'Recall']}
            std_metrics[label] = {metric: np.nanstd(all_metrics[label][metric]) 
                                  for metric in ['TP', 'FP', 'FN', 'Precision', 'Recall']}
        
        overall_mean = {metric: np.nanmean([mean_metrics[label][metric] 
                                         for label in range(1, num_classes+1)]) 
                        for metric in ['TP', 'FP', 'FN', 'Precision', 'Recall']}
        overall_std = {metric: np.nanstd([mean_metrics[label][metric] 
                                       for label in range(1, num_classes+1)]) 
                       for metric in ['TP', 'FP', 'FN', 'Precision', 'Recall']}

        metrics = {
            'mean_per_label': mean_metrics,
            'std_per_label': std_metrics,
            'overall_mean': overall_mean,
            'overall_std': overall_std
        }

    return metrics
