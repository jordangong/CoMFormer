#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pathlib import Path

import numpy as np
import tqdm
from PIL import Image

ID_MAP = {0: 255, 255: 255} | dict(zip(range(1, 21), range(20)))
idmap_fn = np.vectorize(lambda v: ID_MAP[v])


def convert(input, output):
    img = np.asarray(Image.open(input))
    assert img.dtype == np.uint8
    # 0 (background) becomes 255
    img = idmap_fn(img).astype("uint8")
    Image.fromarray(img).save(output)


if __name__ == "__main__":
    root_dir = Path(os.getenv("DETECTRON2_DATASETS", "datasets"))
    for year in ["2007", "2012"]:
        dataset_dir = root_dir / f"VOC{year}"
        for annotation_base_dir in ["SegmentationClass", "SegmentationClassAug"]:
            annotation_dir = dataset_dir / annotation_base_dir
            if Path.exists(annotation_dir):
                print(f"Processing VOC{year} {annotation_base_dir}...")
                output_dir = dataset_dir / f"{annotation_base_dir}_detectron2"
                output_dir.mkdir(parents=True, exist_ok=True)
                for file in tqdm.tqdm(list(annotation_dir.iterdir())):
                    output_file = output_dir / file.name
                    convert(file, output_file)
