from typing import AnyStr, Optional, List, Tuple, Callable, Literal
import torch
from torch.utils.data import Dataset, DataLoader
from .training_utils import apply_augmentations
import os
import json
import cv2
import numpy as np
import numpy.typing as npt
import imageio
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


def create_mask_from_bbox(image: npt.NDArray, bbox: Bbox, mode: Literal["bbox", "threshold"] = "bbox", threshold: float = 0.5) -> npt.NDArray:
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


def resize(image: npt.NDArray, input_size: int, interpolation= cv2.INTER_NEAREST_EXACT) -> npt.NDArray:
    # Apply resize transformation
    resized_image = cv2.resize(image, dsize=(input_size, input_size), interpolation=interpolation)
    # print(resized_image.shape)
    return resized_image


class MIPDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input_paths: List[AnyStr],
        input_size: int,
        augment=True
    ):
        super().__init__()
        self.input_paths = input_paths
        self.img_files = []
        self.label_files = []
        self.cases_ids = []
        self.input_size = input_size,
        self.transform = augment
        self.read_data()

    def __len__(self):
        return len(self.img_files)

    def read_data(self):
        # print(self.input_paths)
        cases_ids = [path.split(os.path.sep)[-1][0:-4] for path in self.input_paths]

        for path, case_id in zip(self.input_paths, cases_ids):
            try:
                # print( os.path.join(path, f'{case_id}_coronal_mip.npy'))
                image_path = path
                mask_path = os.path.join(path[0:-24],'masks', f'{case_id}_mask.png')
                
                # load numpy
                image = normalize(imageio.imread(image_path))
                image = resize(image, self.input_size[0], cv2.INTER_LINEAR)
                
                mask = normalize(imageio.imread(mask_path))
                mask = resize(mask, self.input_size[0])
                
            except FileNotFoundError:
                continue

            if np.all(mask_path == 0) or np.all(mask_path == 0):
                # Mask is empty, skip processing
                print("Mask is empty, skipping processing")
                continue

            self.cases_ids.extend([case_id])
            self.img_files.extend([image])
            self.label_files.extend([mask])

    def __getitem__(self, idx):

        image = self.img_files[idx]
        mask = self.label_files[idx]
        case_name = self.cases_ids[idx]
        # print(img, label_mask)
        if self.transform:
           image, mask= apply_augmentations(images=[image, mask])
        
        return {
            "img": image,
            "mask": mask,
            "case_name": case_name
        }


# testing dataset elements before training
if __name__ == "__main__":
    # to run this file import like below ...if you run main.py ------> from .training_utils import apply_augmentations
    from training_utils import apply_augmentations

    data_folder_path = "C:\\Atomica_CV_source_code\\cycle30\\cvml\\roi-segmentation\\dataset\\data\\original_data"

    dataset = MIPDataset([data_folder_path], 400)

    # Create an iterator for the dataset
    iterator = iter(dataset)

    # # Get the next element in the dataset
    element = next(iterator)

    print(element["img"].shape, element["mask"].shape)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(element["img"])
    plt.axis('off')  # Turn off axis

    axes[1].imshow(element["mask"])
    plt.axis('off')  # Turn off axis

    plt.show()
