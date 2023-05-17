# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging

import torch
from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import Instances
from torch.nn import functional as F

__all__ = ["MaskFormerInstanceDatasetMapper"]

from torch.utils.data import default_collate

from mask2former.data.utils import transform_instance_annotations


class MaskFormerInstanceDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for instance segmentation.

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
            size_divisibility,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            size_divisibility: pad image size to be divisible by this value
        """
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.size_divisibility = size_divisibility

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            augs.append(
                T.RandomCrop(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                )
            )
        if cfg.INPUT.COLOR_AUG_SSD:
            augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
        augs.append(T.RandomFlip())

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert self.is_train, "MaskFormerInstanceDatasetMapper should only be used for training!"

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        if image_file_name := dataset_dict.pop("file_name", None):
            image = utils.read_image(image_file_name, format=self.img_format)
            utils.check_image_size(dataset_dict, image)
        else:
            raise ValueError(
                f"Cannot find image {image_file_name}."
            )

        aug_input = T.AugInput(image)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        image = aug_input.image

        # transform instance masks
        annotations_transformed = []
        if annotations := dataset_dict.pop("annotations", None):
            for anno in annotations:
                if anno.pop("iscrowd", 0) == 0:
                    anno.pop("bbox", None)
                    anno.pop("bbox_mode", None)
                    anno.pop("keypoints", None)
                    anno = transform_instance_annotations(anno, transforms, image.shape[:2])
                    annotations_transformed.append(anno)
        else:
            raise ValueError(
                f"Cannot find 'annotations' for image {image_file_name}."
            )
        annotations = default_collate(annotations_transformed)

        # Pad image and segmentation label here!
        image = torch.from_numpy(image.transpose(2, 0, 1).copy())
        classes = annotations.get("category_id", None)
        masks = annotations.get("segmentation", None)

        if self.size_divisibility > 0:
            _, height, width = image.size()
            padding_size = [
                0,
                self.size_divisibility - width,
                0,
                self.size_divisibility - height,
            ]
            # pad image
            image = F.pad(image, padding_size, value=128).contiguous()
            # pad mask
            if masks is not None:
                masks = F.pad(masks, padding_size, value=0).contiguous()

        dataset_dict["image"] = image

        # Prepare per-category binary masks
        _, *image_size = image.size()
        if classes is None:
            # Some image does not have annotation (all ignored)
            classes = torch.tensor(0, dtype=torch.int64)
            masks = torch.zeros(0, *image_size, dtype=torch.bool)

        instances = Instances(image_size, gt_classes=classes, gt_masks=masks)
        dataset_dict["instances"] = instances

        return dataset_dict
