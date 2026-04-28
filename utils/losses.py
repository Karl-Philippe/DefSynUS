""" https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
Common image segmentation losses.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class SoftDiceLoss(nn.Module):

    def forward(self, logits, true, eps=1e-7):
        """
        Computes the Sørensen–Dice loss.
        
        Args:
            logits: a tensor of shape [B, C, H, W]. Corresponds to the raw output or logits of the model.
            true: a tensor of shape [B, 1, H, W] with class indices.
            eps: added to the denominator for numerical stability.
        
        Returns:
            dice_loss: the Sørensen–Dice loss.
        """
        
        num_classes = logits.shape[1]
        true = true.long()
        
        if num_classes == 1:
            # Binary segmentation
            true_1_hot = torch.eye(2, device=true.device)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            # Multi-class segmentation
            true_1_hot = torch.eye(num_classes, device=true.device)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(logits, dim=1)
        
        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = (2. * intersection / (cardinality + eps)).mean()
        
        return (1 - dice_loss), probas

class DiceLoss(nn.Module):

    def forward(self, logits, true, eps=1e-7, threshold=0.5):
        """
        Computes the Sørensen–Dice loss.
        
        Args:
            logits: a tensor of shape [B, C, H, W]. Corresponds to the raw output or logits of the model.
            true: a tensor of shape [B, 1, H, W] with class indices.
            eps: added to the denominator for numerical stability.
            threshold: threshold for binarization in binary segmentation.
        
        Returns:
            dice_loss: the Sørensen–Dice loss.
        """
        
        num_classes = logits.shape[1]
        true = true.long()
        
        if num_classes == 1:
            # Binary segmentation
            true_1_hot = torch.eye(2, device=true.device)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(logits)
            pos_prob.data = torch.ge(pos_prob.data, threshold).float()
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            # Multi-class segmentation
            true_1_hot = torch.eye(num_classes, device=true.device)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(logits, dim=1)
        
        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = (2. * intersection / (cardinality + eps)).mean()
        
        return (1 - dice_loss), probas

