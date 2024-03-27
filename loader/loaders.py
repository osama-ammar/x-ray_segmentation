import os
from typing import AnyStr, Union, Tuple, List
from .dataset import MIPDataset
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torch.utils.data.distributed import DistributedSampler
import random


def list_files(directory: AnyStr) -> List[AnyStr]:
    images_path = os.path.join(directory,"images")
    files_list = os.listdir(images_path)
    pathes_list = [os.path.join(images_path,file_name) for file_name in files_list]
    
    return pathes_list


def get_data_loaders(dataset_path: AnyStr,
                     batch_size, train_ratio: float = 0.8,
                     test_validate_ratio: float = 0.5,
                     mode: AnyStr = "train",
                     augment: bool = False,
                     input_size: int = 400,
                     random_state: int = 42,
                     distributed_lr: bool = False) -> Tuple[DataLoader, DataLoader]:
    assert mode in ["export", "train", "overfit", "test"]

    # get paths of dataset files
    paths = list_files(dataset_path)

    # Split the data into training and remaining data (validation + test)
    train_paths, val_test_paths = train_test_split(
        paths, train_size=train_ratio, random_state=random_state, shuffle=False)

    # Split the remaining data into validation and test sets
    val_paths, test_paths = train_test_split(
        val_test_paths, train_size=test_validate_ratio, random_state=random_state)


    if mode == "test":
        test_dataset = MIPDataset(
            input_paths=paths,
            input_size=input_size,
            augment=augment,
        )

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=test_sampler ==
                                 None, pin_memory=False, num_workers=0, sampler=validate_sampler)
        print('number of test  instances: ', len(test_dataset))

        return test_loader

    # run training on a sample of data to confirm it's working well before actual training
    # if the model overfits on a small sample then it can generalize on all data and vice verse.
    if mode == "overfit":
        train_dataset = MIPDataset(
            input_paths=paths[0:5],
            input_size=input_size,
            augment=augment,
        )        

        validate_dataset = MIPDataset(
            input_paths=paths[5:8],
            input_size=input_size,
            augment=augment,
        )           


    else:
        # Data Loading
        # Create the training dataset
        train_dataset = MIPDataset(
            input_paths=train_paths,
            input_size=input_size,
            augment=augment,
        )

        # for this dataset we need to suffle validation data once before training , and not during training because every case has ~50 images in seq

        validate_dataset = MIPDataset(
            input_paths=val_test_paths,
            input_size=input_size,
            augment=False,
        )    
        

    train_sampler = None
    validate_sampler = None
    test_sampler = None
    if distributed_lr:
        '''
        this solves a problem when 2 gpus are not working exactly in parallel , due to different number of images ex( gpu1: 900 images , gpu2: 901 image)
        as this cause a problem during training ,  where one gpu finishes and goes outside train loop and the 2nd gpu then finished and waiting the other to synchronize their losses
        '''
        train_sampler = DistributedSampler(train_dataset)
        validate_sampler = DistributedSampler(validate_dataset)
        test_sampler = DistributedSampler(test_dataset)

    print('number of train instances: ', len(train_dataset))
    print('number of validate  instances: ', len(validate_dataset))

    # Create data loaders for training and testing sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_sampler ==
                              None, pin_memory=False, num_workers=1, sampler=train_sampler)
    valid_loader = DataLoader(validate_dataset, batch_size=batch_size,
                              shuffle=False, pin_memory=False, num_workers=0, sampler=validate_sampler)


    return train_loader, valid_loader


# for testing data loaders .... useful for debugging every part of the experiment
if __name__ == "__main__":
    from dataset import MIPDataset

    data_folder_path = "D:\\cases_dicom\\chest_x-ray\\dataset"
    get_data_loaders(data_folder_path,
                     batch_size=4,
                     train_ratio=0.8,
                     test_validate_ratio=0.5,
                     mode="overfit",
                     augment=False,
                     random_state=42,
                     distributed_lr=False)
    
