import random
import cv2
import albumentations as A
from typing import List, Tuple, Dict, Any


def convert_and_normalize_bbox(
        bbox: Tuple[int, int, int, int], 
        image_size: Tuple[int, int], 
    ): 
    """Converts bounding box format and normalizes coordinates.

    Args:
        bbox: A list of bounding box coordinates in the format [xmin, ymin, xmax, ymax].
        image_size: A tuple representing the image size in the format (width, height).

    Returns:
        A list of normalized bounding box coordinates in the format [x_center, y_center, width, height].
    """
    xmin, ymin, xmax, ymax = bbox
    im_height, im_width = image_size

    x_center = ((xmax - xmin)/2 + xmin)/im_width 
    y_center = ((ymax - ymin)/2 + ymin)/im_height
    width = (xmax - xmin)/im_width
    height = (ymax - ymin)/im_height
    if x_center > 1 or y_center > 1 or width > 1 or height > 1:
        print("w h:", image_size)
        print("WARNING: Value(s) is more than 1!")

    return x_center, y_center, width, height

def augment_a_file(
        filepath_pair: Tuple[str, str],
        category_id_to_name: Dict[int, str] = {0: 'table', 1: 'ref', 2: 'object'},
        convert_to_yolo: bool = True,
        n_augments: int = 3,
    ):
    """
    Augment a file of ap pair of image and annotation files.
    Args:
        filepath_pair: A  tuple of two strings (file.jpg, file.txt)
        category_id_to_name: A dictionary mapping category ids to names.
        annotation_format: The format of the annotation file. (Default: 'yolo')
        n_augments: The number of augmentations to perform.

    Returns:
        A list of augmented images and bounding boxes.
    """
    # Read image file
    image = cv2.imread(filepath_pair[0])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Read annotation file
    with open(filepath_pair[1], "r") as f:
        lines = f.readlines()

    # Get the annotation information
    bboxes = [[float(x) for x in line.split(",")[:4]] for line in lines]
    categories = [line.strip().split(",")[-1] for line in lines]
    category_ids = [list(category_id_to_name.keys())[list(category_id_to_name.values()).index(name)] for name in categories]

    # Build the augmentation object
    transform = A.Compose([
        A.RandomBrightnessContrast(p=0.7),
        A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.7),
        A.GaussNoise(p=0.7),
        A.GaussianBlur(p=0.7),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc", # only processing for xmin, ymin, xmax, ymax format for convinience
            label_fields=['category_ids']
        ),
    )

    # Augment the image
    transformed = []
    for _ in range(n_augments):
        if convert_to_yolo:
            tfd = transform(image=image, bboxes=bboxes, category_ids=category_ids)
            im_height, im_width, _ = tfd["image"].shape
            norm_bboxes = [convert_and_normalize_bbox(bbox, (im_height, im_width)) for bbox in tfd["bboxes"]]
            transformed.append({
                "image": tfd["image"],
                "bboxes": norm_bboxes,
                "category_ids": tfd["category_ids"]
            })
        else:
            transformed.append(transform(image=image, bboxes=bboxes, category_ids=category_ids))

    return transformed
