# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from detectron2.projects.point_rend.point_features import point_sample
from scipy.optimize import linear_sum_assignment
from torch import nn


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
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
    numerator = 2 * inputs @ targets.T
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


@torch.cuda.amp.autocast(enabled=False)
def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
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
    num_points = inputs.size(1)

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = pos @ targets.T + neg @ (1 - targets).T

    return loss / num_points


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
            self,
            weight_dict: dict,
            num_points: int = 0
    ):
        """Creates the matcher

        Params:
            class_weight: This is the relative weight of the classification error in the matching cost
            mask_weight: This is the relative weight of the focal loss of the binary mask in the matching cost
            dice_weight: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        assert any(weight_dict.values()), "all costs cannot be 0"
        self.weight_dict = weight_dict
        self.num_points = num_points

    @torch.no_grad()
    def forward(self, class_logits, class_targets, mask_logits, mask_targets):
        """More memory-friendly matching"""
        batch_size, num_queries, _ = class_logits.size()
        class_predicts = class_logits.softmax(-1)  # [batch_size, num_queries, num_classes]
        mask_logits = mask_logits.unsqueeze(2)

        pred_indices, target_indices = [], []
        # Iterate through batch size
        for class_pred, class_target, mask_logit, mask_target in zip(
                class_predicts, class_targets, mask_logits, mask_targets
        ):
            num_targets = len(class_target)
            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -class_pred[:, class_target]

            mask_target = mask_target.unsqueeze(1).float()
            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=mask_logit.device)

            mask_logit = point_sample(
                mask_logit,
                point_coords.expand(num_queries, -1, -1),
                align_corners=False,
            ).squeeze(1)
            mask_target = point_sample(
                mask_target,
                point_coords.expand(num_targets, -1, -1),
                align_corners=False,
            ).squeeze(1)

            cost_mask = batch_sigmoid_ce_loss(mask_logit, mask_target)
            cost_dice = batch_dice_loss(mask_logit, mask_target)

            # Final cost matrix
            cost = self.weight_dict["class"] * cost_class \
                   + self.weight_dict["mask"] * cost_mask \
                   + self.weight_dict["dice"] * cost_dice
            cost = cost.view(num_queries, -1).cpu()
            pred_idx, target_idx = linear_sum_assignment(cost)

            pred_indices.append(torch.from_numpy(pred_idx))
            target_indices.append(torch.from_numpy(target_idx))

        return pred_indices, target_indices

    def __repr__(self, _repr_indent=4):
        return self.__class__.__name__


class SoftmaxMatcher(HungarianMatcher):

    @torch.no_grad()
    def forward(self, class_logits, class_targets, mask_logits, mask_targets):
        """More memory-friendly matching"""
        batch_size, num_queries, _ = class_logits.size()
        class_predicts = class_logits.softmax(-1)  # [batch_size, num_queries, num_classes]
        mask_logits = mask_logits.softmax(1).unsqueeze(2)

        pred_indices, target_indices = [], []
        # Iterate through batch size
        for class_pred, class_target, mask_logit, mask_target in zip(
                class_predicts, class_targets, mask_logits, mask_targets
        ):
            num_targets = len(class_target)
            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -class_pred[:, class_target]

            mask_target = mask_target.unsqueeze(1).float()
            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=mask_logit.device)

            mask_logit = point_sample(
                mask_logit,
                point_coords.expand(num_queries, -1, -1),
                align_corners=False,
            ).squeeze(1)
            mask_target = point_sample(
                mask_target,
                point_coords.expand(num_targets, -1, -1),
                align_corners=False,
            ).squeeze(1)

            cost_dice = batch_dice_loss(mask_logit, mask_target)

            # cost class in [0,1], cost dice in [0,1] -> 0 is worst!
            cost = - 2 * (cost_class * cost_dice) / (cost_class + cost_dice)
            cost = cost.view(num_queries, -1).cpu()
            pred_idx, target_idx = linear_sum_assignment(cost)

            pred_indices.append(torch.from_numpy(pred_idx))
            target_indices.append(torch.from_numpy(target_idx))

        return pred_indices, target_indices
