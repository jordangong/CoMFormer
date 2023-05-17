# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""

import torch
import torch.nn.functional as F
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)
from detectron2.utils.comm import get_world_size
from torch import nn

from .matcher import HungarianMatcher, SoftmaxMatcher
from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list


def focal_loss(inputs, targets, alpha=10, gamma=2, reduction='mean', ignore_index=255):
    ce_loss = F.cross_entropy(inputs, targets, reduction="none", ignore_index=ignore_index)
    pt = torch.exp(-ce_loss)
    f_loss = alpha * (1 - pt) ** gamma * ce_loss
    if reduction == 'mean':
        f_loss = f_loss.mean()
    return f_loss


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum()


def sigmoid_focal_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum()


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum()


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


def setup_mask_criterion(cfg, num_classes):
    # loss weights
    no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT
    weight_dict = {
        "class": cfg.MODEL.MASK_FORMER.CLASS_WEIGHT,
        "mask": cfg.MODEL.MASK_FORMER.MASK_WEIGHT,
        "dice": cfg.MODEL.MASK_FORMER.DICE_WEIGHT
    }
    # building criterion
    if cfg.MODEL.MASK_FORMER.SOFTMASK:
        matcher = SoftmaxMatcher(
            weight_dict=weight_dict,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )
    else:
        matcher = HungarianMatcher(
            weight_dict=weight_dict,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

    criterion = SoftmaxCriterion if cfg.MODEL.MASK_FORMER.SOFTMASK else SetCriterion

    return criterion(
        num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=no_object_weight,
        num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
        importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        focal=cfg.MODEL.MASK_FORMER.FOCAL,
        focal_alpha=cfg.MODEL.MASK_FORMER.FOCAL_ALPHA,
        focal_gamma=cfg.MODEL.MASK_FORMER.FOCAL_GAMMA,
    )


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef,
                 num_points, oversample_ratio, importance_sample_ratio,
                 focal=False, focal_gamma=2, focal_alpha=10):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        class_weight = torch.ones(self.num_classes + 1)
        class_weight[-1] = self.eos_coef
        self.register_buffer("class_weight", class_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.focal = focal
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

    def loss_class(self, logits, targets):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        if self.focal:
            loss = focal_loss(logits.transpose(1, 2), targets,
                              gamma=self.focal_gamma, alpha=self.focal_alpha)
        else:
            loss = F.cross_entropy(logits.transpose(1, 2), targets, self.class_weight)
        return loss

    def rend_point(self, logits, targets):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        logits = logits.unsqueeze(1)
        targets = targets.unsqueeze(1)

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                logits,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_targets = point_sample(
                targets,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            logits,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        return point_logits, point_targets

    def loss_mask(self, logits, targets):
        return sigmoid_ce_loss(logits, targets)

    def loss_dice(self, logits, targets):
        return dice_loss(logits, targets)

    def _get_permutation_idx(self, *indices):
        # permute following indices
        idx = torch.cat([
            torch.stack([torch.full_like(idx[0], fill_value=i), *idx])
            for i, idx in enumerate(zip(*indices))
        ], dim=1)
        return idx

    def match_n_get_losses(self, outputs, targets, num_masks, loss_suffix=""):
        class_logits, mask_logits = outputs["pred_logits"], outputs["pred_masks"]
        class_targets, mask_targets = [], []
        for target in targets:
            class_targets.append(target.gt_classes)
            mask_targets.append(target.gt_masks)

        # Retrieve the matching between the outputs of the last layer and the targets
        pred_indices, target_indices = self.matcher(
            class_logits, class_targets, mask_logits, mask_targets,
        )

        losses = {}
        batch_permu, logits_permu, target_permu = self._get_permutation_idx(
            pred_indices, target_indices,
        )

        class_targets_ = torch.cat([t[i] for t, i in zip(class_targets, target_indices)])
        class_targets = torch.full(
            class_logits.shape[:2], self.num_classes, device=class_logits.device
        )
        class_targets[batch_permu, logits_permu] = class_targets_

        mask_logits = mask_logits[batch_permu, logits_permu]
        mask_targets, valid = nested_tensor_from_tensor_list(mask_targets).decompose()
        mask_targets = mask_targets.to(mask_logits)
        mask_targets = mask_targets[batch_permu, target_permu]
        point_logits, point_targets = self.rend_point(mask_logits, mask_targets)

        for loss_name, weight in self.weight_dict.items():
            loss_fn = getattr(self, f"loss_{loss_name}")
            if loss_name == "class":
                loss = loss_fn(class_logits, class_targets)
            elif loss_name in ["mask", "dice"]:
                loss = loss_fn(point_logits, point_targets) / num_masks
            if loss_suffix != "":
                loss_name += f"_{loss_suffix}"
            losses[f"loss_{loss_name}"] = loss * weight
        return losses

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Compute the average number of target masks across all nodes, for normalization purposes
        num_masks = sum(len(t.gt_masks) for t in targets)
        num_masks = torch.tensor(num_masks, device=outputs["pred_masks"].device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1)

        # Compute all the requested losses
        losses = {}
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if aux_outputs := outputs.pop("aux_outputs", None):
            for i, aux_outputs_i in enumerate(aux_outputs):
                losses |= self.match_n_get_losses(
                    aux_outputs_i, targets, num_masks, loss_suffix=str(i)
                )
        # Main loss
        losses |= self.match_n_get_losses(outputs, targets, num_masks)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


class SoftmaxCriterion(SetCriterion):

    def match_n_get_losses(self, outputs, targets, num_masks, loss_suffix=""):
        class_logits, mask_logits = outputs["pred_logits"], outputs["pred_masks"]
        class_targets, mask_targets = [], []
        for target in targets:
            class_targets.append(target.gt_classes)
            mask_targets.append(target.gt_masks)

        # Retrieve the matching between the outputs of the last layer and the targets
        pred_indices, target_indices = self.matcher(
            class_logits, class_targets, mask_logits, mask_targets,
        )

        losses = {}
        batch_permu, logits_permu, target_permu = self._get_permutation_idx(
            pred_indices, target_indices,
        )

        class_targets_ = torch.cat([t[i] for t, i in zip(class_targets, target_indices)])
        class_targets = torch.full(
            class_logits.shape[:2], self.num_classes, device=class_logits.device
        )
        class_targets[batch_permu, logits_permu] = class_targets_

        mask_preds = mask_logits.softmax(1)
        mask_preds = mask_preds[batch_permu, logits_permu]
        mask_logits = mask_logits.log_softmax(1)
        mask_logits = mask_logits[batch_permu, logits_permu]
        mask_targets, valid = nested_tensor_from_tensor_list(mask_targets).decompose()
        mask_targets = mask_targets.to(mask_logits)
        mask_targets = mask_targets[batch_permu, target_permu]
        point_preds, point_logits, point_targets = self.rend_point(
            mask_preds, mask_logits, mask_targets
        )

        for loss_name, weight in self.weight_dict.items():
            loss_fn = getattr(self, f"loss_{loss_name}")
            if loss_name == "class":
                loss = loss_fn(class_logits, class_targets)
            elif loss_name == "mask":
                loss = loss_fn(point_logits, point_targets) / num_masks
            elif loss_name == "dice":
                loss = loss_fn(point_preds, point_targets) / num_masks
            if loss_suffix != "":
                loss_name += f"_{loss_suffix}"
            losses[f"loss_{loss_name}"] = loss * weight
        return losses

    def rend_point(self, preds, logits, targets):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        preds = preds.unsqueeze(1)
        logits = logits.unsqueeze(1)
        targets = targets.unsqueeze(1)

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                preds,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_targets = point_sample(
                targets,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_preds = point_sample(
            preds,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        point_logits = point_sample(
            logits,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        return point_preds, point_logits, point_targets
