import os

import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.pascal_voc import CLASS_NAMES


def load_voc_semantic(dirname: str, split: str, aug: bool):
    """
    Load Pascal VOC segmentation annotations to Detectron2 format.

    Args:
        dirname: Contain "ImageSets", "JPEGImages",
                and "SegmentationClass" or "SegmentationClassAug"
        split (str): one of "train", "val", "trainval", and "test"
        aug (bool): use augmented dataset or not
    """
    img_set = "SegmentationAug" if aug else "Segmentation"
    split = f"{split}_aug" if aug else split
    anno_basedirname = "SegmentationClassAug_detectron2" if aug else "SegmentationClass_detectron2"
    anno_dirname = os.path.join(dirname, anno_basedirname)
    with open(os.path.join(dirname, "ImageSets", img_set, split + ".txt")) as f:
        if aug:
            file_list = np.loadtxt(f, dtype=str)
            img_files = [f[1:] for f in file_list[:, 0]]
            gt_files = [os.path.basename(f) for f in file_list[:, 1]]
        else:
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


def register_pascal_voc_sem_seg(name, dirname, split, year, aug):
    DatasetCatalog.register(name, lambda: load_voc_semantic(dirname, split, aug))
    MetadataCatalog.get(name).set(dirname=dirname, year=year, split=split, aug=aug)


def register_all_pascal_voc_sem_seg(root):
    SPLITS = [
        ("voc_2007_sem_seg_trainval", "VOC2007", "trainval"),
        ("voc_2007_sem_seg_train", "VOC2007", "train"),
        ("voc_2007_sem_seg_val", "VOC2007", "val"),
        ("voc_2007_sem_seg_test", "VOC2007", "test"),
        ("voc_2012_sem_seg_trainval", "VOC2012", "trainval"),
        ("voc_2012_sem_seg_train", "VOC2012", "train"),
        ("voc_2012_sem_seg_val", "VOC2012", "val"),
        ("voc_2012_sem_seg_trainval_aug", "VOC2012", "trainval"),
        ("voc_2012_sem_seg_train_aug", "VOC2012", "train"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        aug = True if "aug" in name else False
        register_pascal_voc_sem_seg(name, os.path.join(root, dirname), split, year, aug)
        MetadataCatalog.get(name).set(
            stuff_classes=CLASS_NAMES[:],
            evaluator_type="sem_seg",
            ignore_label=255,
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_pascal_voc_sem_seg(_root)
