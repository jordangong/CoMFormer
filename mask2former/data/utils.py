import numpy as np
import pycocotools.mask as mask_util
from detectron2.data import transforms as T
from detectron2.data.detection_utils import transform_keypoint_annotations
from detectron2.structures import BoxMode


def transform_instance_annotations(
        annotation, transforms, image_size, *, keypoint_hflip_indices=None
):
    """
    Apply transforms to box, segmentation and keypoints annotations of a single instance.

    It will use `transforms.apply_box` for the box, and
    `transforms.apply_coords` for segmentation polygons & keypoints.
    If you need anything more specially designed for each data structure,
    you'll need to implement your own version of this function or the transforms.

    Args:
        annotation (dict): dict of instance annotations for a single instance.
            It will be modified in-place.
        transforms (TransformList or list[Transform]):
        image_size (tuple): the height, width of the transformed image
        keypoint_hflip_indices (ndarray[int]): see `create_keypoint_hflip_indices`.

    Returns:
        dict:
            the same input dict with fields "bbox", "segmentation", "keypoints"
            transformed according to `transforms`.
            The "bbox_mode" field will be set to XYXY_ABS.
    """
    if isinstance(transforms, (tuple, list)):
        transforms = T.TransformList(transforms)

    if "bbox" in annotation and "bbox_mode" in annotation:
        bbox, bbox_mode = transform_bbox_annotations(
            annotation["bbox"], annotation["bbox_mode"], transforms, image_size
        )
        annotation["bbox"], annotation["bbox_mode"] = bbox, bbox_mode

    if "segmentation" in annotation:
        segm = transform_segmentation_annotations(
            annotation["segmentation"], transforms, image_size
        )
        annotation["segmentation"] = segm

    if "keypoints" in annotation:
        keypoints = transform_keypoint_annotations(
            annotation["keypoints"], transforms, image_size, keypoint_hflip_indices
        )
        annotation["keypoints"] = keypoints

    return annotation


def transform_bbox_annotations(bbox, mode, transforms, image_size):
    # bbox is 1d (per-instance bounding box)
    bbox = BoxMode.convert(bbox, mode, BoxMode.XYXY_ABS)
    # clip transformed bbox to image size
    bbox = transforms.apply_box(np.array([bbox]))[0].clip(min=0)
    bbox = np.minimum(bbox, list(image_size + image_size)[::-1])
    mode = BoxMode.XYXY_ABS
    return bbox, mode


def transform_segmentation_annotations(segm, transforms, image_size):
    # each instance contains 1 or more polygons
    if isinstance(segm, list):
        # polygons
        polygons = [np.asarray(p).reshape(-1, 2) for p in segm]
        segm = [
            p.reshape(-1) for p in transforms.apply_polygons(polygons)
        ]
    elif isinstance(segm, dict):
        # RLE
        mask = mask_util.decode(segm)
        mask = transforms.apply_segmentation(mask)
        assert tuple(mask.shape[:2]) == image_size
        segm = mask.copy()
    else:
        raise ValueError(
            "Cannot transform segmentation of type '{}'!"
            "Supported types are: polygons as list[list[float] or ndarray],"
            " COCO-style RLE as a dict.".format(type(segm))
        )
    return segm
