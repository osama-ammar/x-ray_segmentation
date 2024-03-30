import json
from typing import AnyStr, Literal, Tuple
import numpy.typing as npt
import numpy as np
import cv2
import os
from typing import List, Optional, Sequence, Tuple, Union, AnyStr

Bbox = Tuple[int, int, int, int]


def get_roi_bbox_from_json(file_name: AnyStr, projection: Literal["seg", "cor"]) -> Bbox:
    with open(file_name, 'r') as file:
        data = json.load(file)

    coordinates = data['all-roi']
    # getting coordinates ---> json["all_roi"]: [[xmin, ymin, zmin], [xmax, ymax, zmax]]
    if projection == "cor":
        x_min, z_min, x_max, z_max = coordinates[0][0], coordinates[0][2], coordinates[1][0], coordinates[1][2]
        bbox = (x_min, z_min, x_max, z_max)
    else:
        y_min, z_min, y_max, z_max = coordinates[0][1], coordinates[0][2], coordinates[1][1], coordinates[1][2]
        bbox = (y_min, z_min, y_max, z_max)

    return bbox


def create_mask_from_bbox(image: npt.NDArray, bbox: Bbox, mode: Literal["bbox", "threshold"] = "bbox",
                          threshold: float = 0.5) -> npt.NDArray:
    # Create a binary mask initialized with zeros
    mask = np.zeros_like(image[:, :])
    # Extract bounding box coordinates
    x_min, y_min, x_max, y_max = bbox

    if mode == "bbox":
        # getting bbox mask
        mask[y_min:y_max, x_min:x_max] = 1
    else:
        # getting threshold mask region
        mask[y_min:y_max, x_min:x_max] = np.where(
            image[y_min:y_max, x_min:x_max] > threshold, 1, 0)

    return mask


def normalize(image: npt.NDArray) -> npt.NDArray:
    # the input image is normalize to range [0, 255] by the function `_enamel_normalize_image_` from the dataset generation script
    # the function does two steps: 1) image_window_map_ [-1000, 3096] -> [0, 255]. 2) gamma correction with gamma = 2.
    # the output is an image in the range [0, 1]
    image[image < 0] = 0
    image = image / 255
    return image


def resize(image: npt.NDArray, input_size: int, interpolation=cv2.INTER_NEAREST_EXACT) -> npt.NDArray:
    # Apply resize transformation
    resized_image = cv2.resize(image, dsize=(input_size, input_size), interpolation=interpolation)
    # print(resized_image.shape)
    return resized_image


def list_files(directory: AnyStr) -> List[AnyStr]:
    images_path = os.path.join(directory,"images")
    files_list = os.listdir(images_path)
    pathes_list = [os.path.join(images_path,file_name) for file_name in files_list]
    
    return pathes_list
