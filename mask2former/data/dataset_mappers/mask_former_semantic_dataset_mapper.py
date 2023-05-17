# Copyright (c) Facebook, Inc. and its affiliates.
import logging

import torch
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import Instances
from torch.nn import functional as F

__all__ = ["MaskFormerSemanticDatasetMapper"]


class MaskFormerSemanticDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for semantic segmentation.

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
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility
        self.remove_bkg = remove_bkg

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
                T.RandomCrop_CategoryAreaConstraint(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                )
            )
        if cfg.INPUT.COLOR_AUG_SSD:
            augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
        augs.append(T.RandomFlip())

        # Assume always applies to the training set.
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label
        remove_bkg = not cfg.MODEL.MASK_FORMER.TEST.MASK_BG

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
            'remove_bkg': remove_bkg
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert self.is_train, "MaskFormerSemanticDatasetMapper should only be used for training!"
        assert "annotations" not in dataset_dict, ValueError(
            "Semantic segmentation dataset should not have 'annotations'.")

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
        else:
            raise ValueError(
                f"Cannot find image {image_file_name}."
            )

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        image = aug_input.image
        sem_seg_gt = aug_input.sem_seg

        # Pad image and segmentation label here!
        image = torch.from_numpy(image.transpose(2, 0, 1).copy())
        sem_seg_gt = torch.from_numpy(sem_seg_gt.astype("int64"))

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

        dataset_dict["image"] = image
        dataset_dict["sem_seg"] = sem_seg_gt

        # Prepare per-category binary masks
        _, *image_size = image.size()
        classes = sem_seg_gt.unique()
        # remove ignored region
        classes = classes[classes != self.ignore_label]
        if self.remove_bkg:
            classes = classes[classes != 0]
        if len(classes) == 0:
            masks = torch.zeros(0, *image_size, dtype=torch.bool)
        else:
            masks = sem_seg_gt == classes[:, None, None]

        instances = Instances(image_size, gt_classes=classes, gt_masks=masks)
        dataset_dict["instances"] = instances

        return dataset_dict
