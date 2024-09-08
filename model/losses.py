# --------------------------------------------------------
# GSVA: Generalized Segmentation via Multimodal Large Language Models
# Written by Zhuofan Xia
# --------------------------------------------------------

import torch
import torch.nn.functional as F

def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss

def iou_loss(
    pred_iou: torch.Tensor,
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    num_masks: float
):
    pred_iou = pred_iou.to(torch.float32).sigmoid()
    pred_mask_ = pred_mask.detach().clone()
    target_mask_ = target_mask.detach().clone()
    inter = (pred_mask_ * target_mask_).sum()
    union = pred_mask_.sum() + target_mask_.sum() - inter
    gt_iou = inter / (union + 1e-8)
    
    iou_loss = ((gt_iou - pred_iou) ** 2).sum() / (num_masks + 1e-8)
    return iou_loss

