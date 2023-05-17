# Copyright (c) Facebook, Inc. and its affiliates.

import torch
from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import Instances
from panopticapi.utils import rgb2id
from torch.nn import functional as F
from torch.utils.data import default_collate

from .mask_former_semantic_dataset_mapper import MaskFormerSemanticDatasetMapper

__all__ = ["MaskFormerPanopticDatasetMapper"]



class MaskFormerPanopticDatasetMapper(MaskFormerSemanticDatasetMapper):
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
            self,
            is_train=True,
            *,
            augmentations,
            image_format,
            ignore_label,
            size_divisibility,
            remove_bkg,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
        """
        super().__init__(
            is_train,
            augmentations=augmentations,
            image_format=image_format,
            ignore_label=ignore_label,
            size_divisibility=size_divisibility,
            remove_bkg=remove_bkg
        )

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert self.is_train, "MaskFormerPanopticDatasetMapper should only be used for training!"
        assert "annotations" not in dataset_dict, ValueError(
            "Panoptic segmentation dataset should not have 'annotations'.")

        dataset_dict = dataset_dict.copy()  # it will be modified by code below
        if image_file_name := dataset_dict.pop("file_name", None):
            image = utils.read_image(image_file_name, format=self.img_format)
            utils.check_image_size(dataset_dict, image)

            if sem_seg_file_name := dataset_dict.pop("sem_seg_file_name", None):
                # PyTorch transformation not implemented for uint16, so converting it to double first
                sem_seg_gt = utils.read_image(sem_seg_file_name).astype("double")
            else:
                raise ValueError(
                    f"Cannot find 'sem_seg_file_name' for image {image_file_name}."
                )
            if pan_seg_file_name := dataset_dict.pop("pan_seg_file_name", None):
                pan_seg_gt = utils.read_image(pan_seg_file_name, "RGB")
            else:
                raise ValueError(
                    f"Cannot find 'pan_seg_file_name' for image {image_file_name}."
                )
            if (segments_info := dataset_dict.pop("segments_info", None)) is not None:
                if len(segments_info) != 0:
                    segments_info = default_collate(segments_info)
            else:
                raise ValueError(
                    f"Cannot find 'segments_info' for image {image_file_name}."
                )
        else:
            raise ValueError(
                f"Cannot find image {image_file_name}."
            )

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        image = aug_input.image
        sem_seg_gt = aug_input.sem_seg

        # apply the same transformation to panoptic segmentation
        pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)
        pan_seg_gt = rgb2id(pan_seg_gt)

        # Pad image and segmentation label here!
        image = torch.from_numpy(image.transpose(2, 0, 1).copy())
        sem_seg_gt = torch.from_numpy(sem_seg_gt.astype("int64"))
        pan_seg_gt = torch.from_numpy(pan_seg_gt.astype("int64"))

        if self.size_divisibility > 0:
            _, height, width = image.size()
            padding_size = [
                0,
                self.size_divisibility - width,
                0,
                self.size_divisibility - height,
            ]
            image = F.pad(image, padding_size, value=128).contiguous()
            sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()
            pan_seg_gt = F.pad(pan_seg_gt, padding_size, value=0).contiguous()  # 0 is the VOID panoptic label

        dataset_dict["image"] = image
        dataset_dict["sem_seg"] = sem_seg_gt

        # Prepare per-category binary masks
        _, *image_size = image.size()
        if segments_info:
            not_crowd_mask = ~segments_info["iscrowd"].bool()
            classes = segments_info["category_id"][not_crowd_mask]
            segments_id = segments_info["id"][not_crowd_mask]
            if not self.remove_bkg:
                not_bkg_mask = classes != 0
                classes = classes[not_bkg_mask]
                segments_id = segments_info[not_bkg_mask]
            masks = pan_seg_gt == segments_id[:, None, None]
        else:
            classes = torch.zeros(0, dtype=torch.int64)
            masks = torch.zeros(0, *image_size, dtype=torch.bool)

        instances = Instances(image_size, gt_classes=classes, gt_masks=masks)
        dataset_dict["instances"] = instances

        return dataset_dict
