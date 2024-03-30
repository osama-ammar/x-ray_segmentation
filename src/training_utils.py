import matplotlib.patches as patches
import numpy as np
import numpy.typing as npt
import cv2
import torch
import csv
from matplotlib import pyplot as plt
import seaborn as sn
from typing import Optional, Sequence, Tuple, Union, AnyStr, Dict, List
import os
import datetime
import torch.nn.functional as F
import os
import albumentations as A
import random

#######################
# Augmentations
#######################


def apply_augmentations(images: Sequence[npt.NDArray], probability: float = 0.5) -> Sequence[npt.NDArray]:
    """defining some augmentations using Albumentations library"""
    img, mask = images
    transform = A.Compose([
        # A.HorizontalFlip(p=probability),
        # A.RandomBrightnessContrast(p=probability),
        A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT, p=probability),
        # A.GaussianBlur(p=probability),
        # A.RandomResizedCrop(256, 256, scale=(0.5, 1.0),ratio=(0.75, 1.333), p=probability),
    ])

    # Apply the augmentations to each image in the list
    transformed_images = transform(image=img, mask=mask)
    img = transformed_images["image"]
    label_mask = transformed_images["mask"]

    return img, label_mask


def bbox_from_mask(mask_batch: torch.Tensor, logits: bool = False) -> List[Dict]:
    """getting bbox indices from a mask"""
    # (b,c,h,w)--->(b,h,w)
    mask_batch = mask_batch.cpu()

    if logits:
        mask_batch = F.softmax(mask_batch, dim=1)
        # Getting the argmax (B,2,400,400)-->(B,400,400)
        mask_batch = torch.argmax(mask_batch, dim=1)

    batch_boxes = []
    temp_labels = [0] * len(mask_batch)
    # print(mask_batch)
    for mask in mask_batch:
        # Find non-zero indices in the mask
        indices = np.argwhere(mask == 1)
        # checking if indices is not empty
        if indices.numel() > 0:
            # Extract x and y coordinates
            x_coords, y_coords = indices[0], indices[1]
            # Calculate bounding box coordinates
            x_min, x_max = torch.min(x_coords), torch.max(x_coords)
            y_min, y_max = torch.min(y_coords), torch.max(y_coords)
            batch_boxes.append([x_min.item(), y_min.item(),
                                x_max.item(), y_max.item()])
        else:
            batch_boxes.append([0, 0, 0, 0])
    # I made this shape so that it can be easily used with torch metrics IOU calculations
    output_dict = {"boxes": torch.tensor(
        batch_boxes), "labels": torch.tensor(temp_labels), "scores": torch.tensor([0.0]*len(mask_batch))}
    # print(output_dict)
    return [output_dict]


################################################################
# experiment reporting (maybe deleted ...we are now using mlflow
################################################################
def make_log_folder(log_folder_path: AnyStr, trial_name: AnyStr) -> AnyStr:
    """making a logs folder for each trial labeled by time (if not exist)"""

    if not os.path.exists(os.path.join(log_folder_path, "logs")):
        os.mkdir(os.path.join(log_folder_path, "logs"))

    # making a folder for a trial
    log_folder_path = os.path.join(
        log_folder_path, "logs", f"{datetime.datetime.now().strftime('%Y-%m-%d___%H-%M')}-{trial_name}")
    if not os.path.exists(log_folder_path):
        os.mkdir(log_folder_path)

    train_imgs_folder_path = os.path.join(log_folder_path, "training_imgs")
    if not os.path.exists(train_imgs_folder_path):
        os.mkdir(train_imgs_folder_path)

    val_imgs_folder_path = os.path.join(log_folder_path, "validation_imgs")
    if not os.path.exists(val_imgs_folder_path):
        os.mkdir(val_imgs_folder_path)

    return log_folder_path


def save_hyperparams(log_folder_path: AnyStr,
                     epochs: int,
                     batch_size: int,
                     optimizer: torch.optim.Optimizer, criterion: torch.nn.Module,
                     lr_scheduler: torch.optim.lr_scheduler._LRScheduler):
    """saving hyperparameters in "hyperparameters.txt """

    with open(os.path.join(log_folder_path, "hyperparameters.txt"), "w+") as f:
        f.write(f"epochs: {epochs}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"Optimizer: {optimizer.__repr__()}\n")
        f.write(f"criterion: {criterion.__repr__()}\n")
        if lr_scheduler:
            f.write(f"LR Scheduler: {lr_scheduler.__repr__()}\n")


def save_online_logs(log_folder_path, header, log_line):
    """used to save logs (hyper parameters and losses ) of a training trial"""

    # if csv file doesn't exist , make it and add header to it
    csv_path = os.path.join(log_folder_path, "testing_metrics.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, mode='a', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(header)

    # if csv file  exist , append lines to it
    with open(csv_path, mode='a', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(log_line)

    # returned to be used  later if needed ...to be updated during training
    return csv_path, header


def get_mask_outlines(mask: npt.NDArray) -> npt.NDArray:
    """given a mask we will get boundary points of this mask  """
    # Convert mask to binary image (0 and 255 values)
    mask_binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    # Apply morphological gradient which is the difference between dilation and erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    mask_binary= cv2.morphologyEx(mask_binary, cv2.MORPH_GRADIENT, kernel)
    coordinates = np.where(mask_binary == 255)
    x_coordinates , y_coordinates = coordinates
    points=[]
    for _ in range(30):
        # Generate a random index within the range of the coordinates
        random_index = random.randint(0, len(x_coordinates) - 1)
        
        # Retrieve the x and y coordinates at the random index
        x = x_coordinates[random_index]
        y = y_coordinates[random_index]
        
        # Append the random point to the list
        points.append((x, y))
    return points

def visualize_model_output(input_images: torch.Tensor,
                           targets: torch.Tensor,
                           logits: torch.Tensor,
                           result_path: AnyStr,
                           epoch: int,
                           training: bool,
                           case_name: AnyStr,
):
    """used to visualize model (input , output , ground truth while training or validation...)"""
    outputs = logits
    if len(logits.shape) == 4:
        # Apply softmax and get argmax
        outputs = F.softmax(logits, dim=1)
        # Getting the argmax (B,2,400,400)-->(B,400,400)
        outputs = torch.argmax(outputs, dim=1).to(torch.float32)
    targets = targets.to(torch.float32)
    random_item = random.randint(0, input_images.shape[0]-1)
    # Add the rectangle patch to the axis
    for index, (img, output, target) in enumerate(zip(input_images[random_item].unsqueeze(0), outputs[random_item].unsqueeze(0), targets[random_item].unsqueeze(0))):
        output_filename = os.path.join(
            result_path,
            "training_imgs" if training else "validation_imgs",
            f"train_{epoch}.png" if training else f"val_{epoch}.png",
        )



        fig, axs = plt.subplots(1, 4, figsize=(12, 4))
        edge_points = get_mask_outlines(target.cpu().numpy())
        #print(edge_points)
        axs[0].imshow(img.squeeze().cpu().numpy(), cmap="gray")
        axs[0].set_title(f"case {case_name[random_item]}")
        axs[0].axis("off")

        axs[1].imshow(output.cpu().numpy(), cmap="gray")

        axs[1].set_title(f"Output Mask  ")
        axs[1].axis("off")

        axs[2].imshow(target.cpu().numpy(), cmap="gray")
        axs[2].set_title(f"Ground Truth ")
        axs[2].axis("off")

        axs[3].imshow(img.squeeze().cpu().numpy(), cmap="gray")
        for point in edge_points:
            axs[3].scatter(point[1], point[0], color='red')  # Scatter plot each point on the image

        #axs[3].contour(edge_points, colors='red', levels=[0.1])
        
        axs[3].set_title(f"mask points  ")
        axs[3].axis("off")

        plt.savefig(output_filename)
        # plt.show()
        plt.close()


def save_plots(train_loss: List, validation_loss: List, conf_matrix: npt.NDArray, log_folder_path: AnyStr) -> Tuple[AnyStr, AnyStr]:
    """Save losses and metrics plots to disk"""

    # plot and save the train and validation loss
    plt.figure()
    plt.plot(train_loss, label="Training Loss")
    plt.plot(validation_loss, label="Validation Loss")
    plt.legend()
    loss_fig_path = os.path.join(log_folder_path, "loss.png")
    plt.savefig(loss_fig_path)
    plt.show()

    # plot and save the conf matrix
    plt.figure()
    ax = sn.heatmap(conf_matrix, annot=True, cmap="Blues",
                    linewidth=.5, fmt=".2f", linecolor="black")
    ax.xaxis.tick_top()
    conf_mat_path = os.path.join(log_folder_path, "conf_matrix.png")
    plt.savefig(conf_mat_path)
    return loss_fig_path, conf_mat_path
