# here will be the main training functionality
import evaluate
from loader import visualize_model_output, bbox_from_mask
import torch.cuda.amp as amp
import numpy as np
import numpy.typing as npt

import torch
from torch.utils.data import DataLoader
from torch.nn import Module 
import  torch.nn as nn 
from torch import Tensor, device, no_grad, set_grad_enabled
import torch
from torch.types import _size as torch_size
from torch.optim import Optimizer
import torch.nn.functional as F
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec

from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List, Optional, Sequence, Tuple, Union, AnyStr

from networks import save_ckp


def loss_fn_test(model: Module, input_size: torch_size, label_size: torch_size, loss_fn: Module, device: AnyStr = "cuda"):
    " used to test loss function dynamics in UNET (input , output , errors) before training "
    test_tensor = torch.rand(input_size)
    test_tensor = test_tensor.to(device, dtype=torch.float32)

    label_tensor = torch.randint(0, 4, label_size)
    label_tensor = label_tensor.to(device, dtype=torch.uint8)

    output = model(test_tensor)
    print("sizes output")
    print(output.size(), input_size, label_size)
    loss = loss_fn(output, label_tensor.float())

    print("loss function testing .....")
    print("loss , its size :", loss, loss.size())

######################################################################3
T = 3
alpha = 0.7

def loss_fn_kd(teacher_output , student_output ,criterion, data_item, label ):
    '''student_output = student_output.round() 
    student_output[student_output<0] = 0
    label = torch.clamp(label, min = 0, max = 1)
    teacher_output = torch.clamp(teacher_output, min = 0, max = 1)'''

    # student_loss= criterion(student_output, label)
    # teacher_loss= criterion(teacher_output, label)
    # print(teacher_loss)
    student_loss = criterion(student_output, label)
    
    pred_T = torch.softmax(teacher_output/T,dim=1)
    pred_S = torch.log_softmax(student_output/T,dim=1)
    

    # Calculate the knowledge distillation loss
    kd_loss = F.kl_div(pred_S, pred_T, reduction='batchmean') * (T ** 2)

    # Combine the student loss and the knowledge distillation loss
    loss = student_loss * (1 - alpha) + kd_loss * alpha
    
    print(f"kd_loss :{kd_loss} , student_loss:{student_loss} , loss :{loss}")
    
    return loss,student_output,student_loss



######################################################################3

#train_knowledge_distillation`` with a temperature of 2. Arbitrarily set the weights to 0.75 for CE and 0.25 for distillation loss.
"""
    A higher temperature makes the probability distribution softer, spreading the probabilities more evenly across classes, 
    which can provide more nuanced information to the student model.
    A lower temperature makes the distribution sharper, resembling a hard 
    classification with more pronounced peaks.

    Usage: Temperature is applied to the logits (the raw, unnormalized outputs of the model) before the softmax function.
    Both the teacher and student models' logits are divided by the temperature value.
"""
def get_distillation_loss(teacher_outputs , student_outputs ,criterion, data_item ,label, soft_target_loss_weight ,ce_loss_weight,temperature):
    #print(item["img"].unsqueeze(1).to(used_device))
    # Forward pass.

    ######################################################
    #Soften the student logits by applying softmax first and log() second
    soft_targets = nn.functional.softmax(teacher_outputs / temperature, dim=1)
    soft_prob = nn.functional.log_softmax(student_outputs / temperature, dim=1)
    # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
    #soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (temperature**2)
    
    # Calculate the true label loss
    student_loss= criterion(student_outputs, label)
    kd_loss=criterion(soft_targets, soft_prob)
    # Weighted sum of the two losses
    loss = soft_target_loss_weight * kd_loss + ce_loss_weight * student_loss  
    print(f"kd_loss :{kd_loss} , student_loss:{student_loss} ")
      
    
    return loss,student_outputs,student_loss


def train_epoch_distillation(
    teacher_model: Module,
    student_model: Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    criterion: Module,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    auto_cast: bool = False,
    clip_grad: bool = False,
    epoch: int = 0,
    metrics: evaluate.UnetMetrics = None,
    visualization_path: AnyStr = None,
    used_device: torch.device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'),
    distributed_lr: bool = False,
    temperature=4, soft_target_loss_weight=0.25, ce_loss_weight=0.75,
) -> float:
    student_model.train()
    student_model.zero_grad(set_to_none=True)
    
    teacher_model.eval()

    mp_rank = 0
    if distributed_lr:
        import torch.distributed as dist
        mp_rank = dist.get_rank()

    tqdm_train_loader = tqdm(train_loader, disable=(mp_rank != 0))

    # Create a GradScaler object. to enable mixed precision ... that autocast data (ex- float32 -> float16) to save cuda memory during training
    scaler = None
    if auto_cast:
        scaler = amp.GradScaler()

    train_running_loss = 0.0
    with set_grad_enabled(True):
        for item in tqdm_train_loader:
            # Wrap the forward and backward passes in autocast
            # with amp.autocast():
            data = item["img"].float().unsqueeze(1).to('cuda')
            mask = item["mask"].to('cuda').long()
            case_name = item["case_name"]

            student_outputs: Tensor = student_model(data)

            with torch.no_grad():
                teacher_outputs: Tensor = teacher_model(data)
                
            # getting distillation loss
            loss ,student_outputs,student_loss= get_distillation_loss(teacher_outputs , student_outputs ,criterion, data ,mask, soft_target_loss_weight ,ce_loss_weight,temperature)
            #loss ,student_outputs,student_loss= loss_fn_kd(teacher_outputs , student_outputs  ,criterion, data, mask )

            ##################################################3
            
            
            # update metrics.
            metrics(student_outputs, mask)

            train_running_loss += loss.item()

            # compute gradients
            loss.backward()


            # clips the gradients in-place by scaling down the gradients if their norm exceeds the specified max_norm.
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)

            # Update the parameters.(use this two ines instead of (optimizer.step() ) to apply mixed precesion float32<--->float16)
            if auto_cast:
                # Scale the gradients.
                # .backward(retain_graph=True) #scales the loss value by the appropriate factor to avoid gradient underflow or overflow
                scaler.scale(student_loss)
                # performs the optimizer step with the scaled gradients.
                scaler.step(optimizer)
                scaler.update()  # updates scaler scale factor for next iteration
            else:
                # un comment this and comment the above 2 lines to make mixed precision off
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if lr_scheduler is not None:
                # step the scheduler to update the learning rate during training
                lr_scheduler.step(epoch)

            over_all_loss = train_running_loss / len(train_loader)

    if distributed_lr:
        import torch.distributed as dist
        """
        dist.all_reduce() : used to synchronize the gradients of a distributed model across all processes.
        -This is necessary to ensure that all processes are using the same gradients to update their model weights.
        -in-place operation (input tensor will be modified after the call)
        """
        dist.all_reduce(train_running_loss)

    visualize_model_output(data, mask, student_outputs, visualization_path,
                           epoch, True, case_name)
    return over_all_loss


@no_grad()
def validate_epoch(
    model: Module,
    validate_loader: DataLoader,
    criterion: Module,
    metrics: evaluate.UnetMetrics = None,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    epoch: int = 0,
    visualization_path: AnyStr = None,
    used_device: torch.device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'),
    distributed_lr: bool = False
) -> float:
    mp_rank = 0
    if distributed_lr:
        import torch.distributed as dist
        mp_rank = dist.get_rank()

    # setting model to evaluation mode
    model.eval()
    tqdm_validate_loader = tqdm(validate_loader, disable=(mp_rank != 0))
    validate_loss = 0.0
    with set_grad_enabled(False):
        for item in tqdm_validate_loader:
            data = item["img"].float().to('cuda').unsqueeze(1)
            mask = item["mask"].to('cuda').long()
            case_name = item["case_name"]

            # Forward pass.
            outputs: Tensor = model(data)

            # Calculate the loss.
            loss: Tensor = criterion(outputs, mask)

            # update metrics.
            metrics(outputs, mask)

            validate_loss += loss.item()



    validation_loss = validate_loss / len(validate_loader)
    if distributed_lr:
        import torch.distributed as dist
        """
        dist.all_reduce() : used to synchronize the gradients of a distributed model across all processes.
        -This is necessary to ensure that all processes are using the same gradients to update their model weights.
        -in-place operation (input tensor will be modified after the call)
        """
        dist.all_reduce(validate_loss)
    visualize_model_output(data, mask, outputs, visualization_path,
                           epoch, False, case_name)
    return validation_loss




def train_distillation(
    teacher_model: Module,
    student_model: Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: Optimizer,
    criterion: Module,
    out_channels: int,
    save_weight_path: AnyStr,
    log_folder_path: AnyStr,
    trial_name: AnyStr,
    used_device: Optional[device] = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'),
    distributed_lr: bool = False,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    epochs: int = 100,
    auto_cast: bool = False,
    clip_grad: bool = False,
    log_period: int = 4,
    mlflow=None,
    temperature=2, soft_target_loss_weight=0.25, ce_loss_weight=0.75,
    
) -> Tuple[List[float], List[float]]:
    assert log_period >= 1

    mp_rank = 0
    if distributed_lr:
        import torch.distributed as dist
        mp_rank = dist.get_rank()

    # initialize loss accumulation lists and log writers
    train_losses, validate_losses = None, None
    best_val_loss = float('inf')
    if mp_rank == 0:
        train_losses: List[float] = []
        validate_losses: List[float] = []

    # defining our metrics ...to be updated during training and finally computed
    training_metrics = evaluate.UnetMetrics(num_classes=out_channels)
    validation_metrics = evaluate.UnetMetrics(num_classes=out_channels)
    save_weight_path=save_weight_path.replace(".pth","_distilled.pth")
    
    for epoch in range(1, epochs + 1):

        train_loss = train_epoch_distillation(teacher_model,student_model, train_loader, optimizer,
                                 criterion, lr_scheduler, auto_cast, clip_grad, epoch, training_metrics,
                                 log_folder_path, used_device, distributed_lr,temperature, soft_target_loss_weight, ce_loss_weight)
        train_losses.append(train_loss)

        if epoch % log_period == 0 and mp_rank == 0:
            # compute validation loss
            validate_loss = validate_epoch(student_model, test_loader, criterion, validation_metrics, lr_scheduler, epoch,
                                           log_folder_path, used_device, distributed_lr)

            # save model if it's better than the best model so far
            if validate_loss < best_val_loss:
                best_val_loss = validate_loss
                save_ckp(save_weight_path, student_model, optimizer, epoch)
                # Save weights returns the model to the CPU
                # thus we must get it back into the utilized device.
                student_model.to(used_device)

            validate_losses.append(validate_loss)

            # getting eval metrics
            training_accuracy, training_precision, training_recall, training_iou, training_dice = training_metrics.compute()
            validation_accuracy, validation_precision, validation_recall, validation_iou, validation_dice = validation_metrics.compute()


            print([f'epoch:{epoch:04d}', f'train_loss:{train_loss:.5f}', f'validate_loss:{validate_loss:.5f}', f'train accuracy:{training_accuracy:.5f}',
                   f'training_iou:{training_iou:.5f}', f'validation_accuracy:{validation_accuracy:.5f}', f'validation_iou:{validation_iou:.5f}'])

            # MlFlow stuff
            mlflow.log_artifacts(log_folder_path + "\\training_imgs")
            mlflow.log_artifacts(log_folder_path + "\\validation_imgs")
            metrics_dict = {
                'train_loss': train_loss,
                'validation_loss': validate_loss,
                'training_accuracy': training_accuracy,
                'validation_accuracy': validation_accuracy,
                'training_iou': training_iou,
                'validation_iou': validation_iou,
            }

            mlflow.log_metrics(metrics_dict, synchronous=False,step=epoch)

            # Define input and output signatures for model
            input_schema = Schema(
                [TensorSpec(name="input", shape=[None, 1, 400, 400], type=np.dtype(np.float32))])
            output_schema = Schema([TensorSpec(name="output", shape=[
                                   None, 2, 400, 400], type=np.dtype(np.float32))])
            model_signature = ModelSignature(
                inputs=input_schema, outputs=output_schema)
            pip_requirements = [f'torch=={str(torch.__version__)}',
                                'mlflow']  # to supress  mlflow warning
            mlflow.pytorch.log_model(student_model, artifact_path='model', pip_requirements=pip_requirements,
                                     registered_model_name=trial_name, signature=model_signature)

    # getting conf matrix and plot it
    #training_metrics.plot_conf_matrix()

    return train_losses, validate_losses
