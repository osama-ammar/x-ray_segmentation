from typing import Optional
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from .data_operations import normalize, get_roi_bbox_from_json, resize, create_mask_from_bbox ,list_files
import imageio



class MIPDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, input_size: int, batch_size: int = 32, num_workers: int = 4,
                 train_validate_ratio: float = 0.8, test_validate_ratio: float = 0.5,mode='train'):
        super().__init__()
        self.data_dir = data_dir
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = train_validate_ratio
        self.mode= mode

        self.data = []  # List to store prepared data

    def prepare_data(self):
        # Prepare your data here and load it into self.data
        paths = list_files(self.data_dir)  # Get list of file paths in data directory
        cases_ids = [path.split(os.path.sep)[-1][0:-4] for path in paths]  # Extract case IDs from file paths

        if self.mode=='overfit' or self.mode=='test_onnx':
                print("overfitting modeeeeeeeeeeeeeeeeeeeeee")
                paths = paths[0:8]
                cases_ids = cases_ids[0:8]

        # Iterate over image and mask paths
        for image_path, case_id in zip(paths, cases_ids):
            try:
                # Load image and mask
                mask_path = os.path.join(image_path[0:-24],'masks', f'{case_id}_mask.png')
                image = normalize(imageio.imread(image_path))
                image = resize(image, self.input_size, cv2.INTER_LINEAR)
    
                mask = normalize(imageio.imread(mask_path))
                mask = resize(mask, self.input_size)
                
            except FileNotFoundError:
                continue  # Skip processing if file not found

            # Skip processing if mask is empty
            if np.all(mask_path == 0) or np.all(mask_path == 0):
                print("Mask is empty, skipping processing")
                continue

            # Store image, mask, and case ID
            self.data.append({"img": image, "mask": mask, "case_id": case_id})

    def setup(self, stage: Optional[str] = None):
        # Splitting the dataset into train and validation sets
        dataset_size = len(self.data)
        val_size = int(self.val_split * dataset_size)
        train_size = dataset_size - val_size

        # Randomly split data into train and validation sets
        train_data, val_data = random_split(self.data, [train_size, val_size])

        # Create datasets for train and validation
        self.train_dataset = SubsetMIPDataset(train_data)
        self.val_dataset = SubsetMIPDataset(val_data)

    def train_dataloader(self):
        # DataLoader for training set
        return DataLoader(self.data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,persistent_workers=True)

    # def test_dataloader(self):
    #     # DataLoader for test set
    #     return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


class SubsetMIPDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]









# testing dataset elements before training
if __name__ == "__main__":
    # to run this file import like below ...if you run main.py ------> from .training_utils import apply_augmentations
    from src.training_utils import apply_augmentations

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
