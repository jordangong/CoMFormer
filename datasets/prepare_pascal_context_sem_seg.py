"""
Modified from mmsegmentation
link: https://github.com/open-mmlab/mmsegmentation/blob/e64548fda0221ad708f5da29dc907e51a644c345/tools/dataset_converters/pascal_context.py
"""
import os
from pathlib import Path

import numpy as np
import tqdm
from PIL import Image
from detail import Detail

_mapping = [
    0, 2, 9, 18, 19, 22, 23, 25, 31, 33, 34, 44, 45, 46, 59, 65, 68, 72, 80,
    85, 98, 104, 105, 113, 115, 144, 158, 159, 162, 187, 189, 207, 220, 232,
    258, 259, 260, 284, 295, 296, 308, 324, 326, 347, 349, 354, 355, 360, 366,
    368, 397, 415, 416, 420, 424, 427, 440, 445, 454, 458,
]
ID_MAP = dict(zip(_mapping, range(len(_mapping))))
idmap_fn = np.vectorize(lambda v: ID_MAP[v])

if __name__ == "__main__":
    root_dir = Path(os.getenv("DETECTRON2_DATASETS", "datasets"))
    voc_dir = root_dir / "VOC2010"
    img_dir = voc_dir / "JPEGImages"
    img_set_dir = voc_dir / "ImageSets" / "SegmentationContext"
    anno_dir = voc_dir / "SegmentationClassContext"
    anno_d2_dir = voc_dir / "SegmentationClassContext_detectron2"
    json_path = voc_dir / "trainval_merged.json"
    anno_dir.mkdir(parents=True, exist_ok=True)
    anno_d2_dir.mkdir(parents=True, exist_ok=True)
    img_set_dir.mkdir(parents=True, exist_ok=True)

    train_detail = Detail(json_path, img_dir, "train")
    val_detail = Detail(json_path, img_dir, "val")
    train_imgs = train_detail.getImgs()
    val_imgs = val_detail.getImgs()
    train_filenames = sorted([Path(img["file_name"]).stem for img in train_imgs])
    val_filenames = sorted([Path(img["file_name"]).stem for img in val_imgs])

    np.savetxt(img_set_dir / "train.txt", np.asarray(train_filenames), fmt="%s")
    np.savetxt(img_set_dir / "val.txt", np.asarray(val_filenames), fmt="%s")


    def convert(mask, filename):
        mask = idmap_fn(mask).astype("uint8")
        mask_d2 = mask - 1  # 0 (background) becomes 255. others are shifted by 1
        Image.fromarray(mask).save(anno_dir / f"{filename}.png")
        Image.fromarray(mask_d2).save(anno_d2_dir / f"{filename}.png")


    for filename in tqdm.tqdm(train_filenames):
        convert(train_detail.getMask(filename), filename)

    for filename in tqdm.tqdm(val_filenames):
        convert(val_detail.getMask(filename), filename)
