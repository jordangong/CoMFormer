import os

import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog

CONTEXT_CLASS_NAMES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'bag', 'bed',
    'bench', 'book', 'building', 'cabinet', 'ceiling', 'cloth', 'computer',
    'cup', 'door', 'fence', 'floor', 'flower', 'food', 'grass', 'ground',
    'keyboard', 'light', 'mountain', 'mouse', 'curtain', 'platform', 'sign',
    'plate', 'road', 'rock', 'shelves', 'sidewalk', 'sky', 'snow', 'bedclothes',
    'track', 'tree', 'truck', 'wall', 'water', 'window', 'wood'
]


def load_pascal_context_semantic(dirname: str, split: str):
    """
    Load Pascal VOC segmentation annotations to Detectron2 format.

    Args:
        dirname: Contain "ImageSets", "JPEGImages", and "SegmentationClassContext"
        split (str): one of "train" and "val"
    """
    img_set = os.path.join(dirname, "ImageSets", "SegmentationContext")
    anno_dirname = os.path.join(dirname, "SegmentationClassContext_detectron2")
    with open(os.path.join(dirname, "ImageSets", img_set, split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=str)
        img_files = [os.path.join("JPEGImages", f"{fid}.jpg") for fid in fileids]
        gt_files = [os.path.join(f"{fid}.png") for fid in fileids]

    dataset_dicts = []
    for (img_file, gt_file) in zip(img_files, gt_files):
        dataset_dicts.append({
            "file_name": os.path.join(dirname, img_file),
            "sem_seg_file_name": os.path.join(anno_dirname, gt_file),
        })
    return dataset_dicts


def register_pascal_context_sem_seg(name, dirname, split):
    DatasetCatalog.register(name, lambda: load_pascal_context_semantic(dirname, split))
    MetadataCatalog.get(name).set(dirname=dirname, split=split)


def register_all_pascal_context_sem_seg(root):
    for split in ["train", "val"]:
        name = f"pascal_context_sem_seg_{split}"
        register_pascal_context_sem_seg(name, os.path.join(root, "VOC2010"), split)
        MetadataCatalog.get(name).set(
            stuff_classes=CONTEXT_CLASS_NAMES[:],
            evaluator_type="sem_seg",
            ignore_label=255,
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_pascal_context_sem_seg(_root)
