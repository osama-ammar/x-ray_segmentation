from networks import U_Net, SMP, onnx_export, load_ckp ,U_Net_distilled
from loader import get_data_loaders, make_log_folder
from test import test
import subprocess
import mlflow
import torch.nn as nn
import torch.optim as optim
import yaml
from train import *
from train_distillation import *

import shutil
import os
import warnings
from typing import Dict

# Ignore the user warning about the missing audio backend
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated.*")

path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "model_config.yaml")
with open(path, "r") as f:
    config: Dict = yaml.load(f, Loader=yaml.FullLoader)




def main():
    ###################
    # Hyper-parameters
    ###################
    mode = config.get("mode")
    epochs = config.get("epochs")
    log_period = config.get("log_period")
    batch_size = config.get("batch_size")
    learning_rate = config.get("learning_rate")
    use_lr_scheduler = config.get("use_lr_scheduler")
    lr_scheduler_gamma = config.get("lr_scheduler_gamma")
    random_seed = config.get("random_seed")
    distributed_lr = config.get("distributed_lr")
    trial_name = config.get("trial_name")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    auto_cast = config.get("auto_cast")
    clip_grad = config.get("clip_grad")
    mlflow_trial_name = config.get("mlflow_trial_name")

    # Data stuff
    dataset_version = config.get("dataset_version")
    dataset_path = config.get("dataset_path")
    weights_path = config.get("weights_path")
    results_path = config.get("results_path")
    test_dataset_path = config.get("test_dataset_path")
    train_validate_ratio = config.get("train_validate_ratio")
    test_validate_ratio = config.get("test_validate_ratio")
    augment = config.get("augment")
    
    # Model stuff
    input_size = config.get("input_size")
    in_channels = config.get("in_channels")
    out_channels = config.get("out_channels")
    fmaps = config.get("fmaps")
    dropout = config.get("dropout")
    

    ####################################
    # moving to selected dataset version
    ####################################
    if dataset_version:
        git_checkout_command = ["git", "checkout", dataset_version]
        dvc_pull_command = ["dvc", "pull"]
        git_show = ["git", "describe"]
        try:
            subprocess.run(git_checkout_command, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True, check=True, cwd=dataset_path)
            subprocess.run(dvc_pull_command, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True, check=True, cwd=dataset_path)
            cmd_output = subprocess.run(git_show, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True, check=True, cwd=dataset_path)
            # To capture the command's output, you can access result.stdout
            print("Git & DVC Outputs:")
            print("working on dataset version :", cmd_output.stdout)
        except subprocess.CalledProcessError as e:
            # If the Git command returns an error, you can capture the error message
            print("Error running Git & DVC commands:")
            print(e.stderr)

    # Set the experiment for MLflow
    mlflow.set_experiment(mlflow_trial_name)
    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
    # mlflow.doctor()
    with mlflow.start_run():
        ###########################
        # Distributed learning
        ##########################

        models = {
            "U_Net": U_Net(in_channels, out_channels, fmaps, dropout),
            "U_Net_distilled": U_Net_distilled(in_channels, out_channels, 10, dropout),
            
            "SMP": SMP(in_channels, out_channels)
        }
        # model initializing
        # rank of the running process in multiple-processing setting or 0 for single-processing
        model = models[config.get("model")].to(device)
        light_model =  models["U_Net_distilled"].to(device)

        if distributed_lr:
            """
            - World size : is the total number of processes participating in the distributed training process.
                This can be the number of GPUs on a single machine, or the number of machines in a cluster.
            - Rank :is a unique identifier assigned to each process in the distributed training process. 
                The rank of a process ranges from 0 to world size - 1
                Processes in a distributed training process can communicate with each other using their ranks

            """
            batch_size = batch_size // int(os.environ["WORLD_SIZE"])
            import torch.distributed as dist
            from torch.nn.parallel import DistributedDataParallel as DDP

            # Initialize the distributed environment
            dist.init_process_group("gloo", rank=int(os.environ["LOCAL_RANK"]),
                                    world_size=int(os.environ['WORLD_SIZE']))

            mp_rank = dist.get_rank()
            print("rank: ", mp_rank, "world size:", os.environ["WORLD_SIZE"])
            device = torch.device('cuda:' + str(mp_rank))
            torch.cuda.set_device(device)

            # model initializing
            model = model.to(device)
            model = DDP(model, device_ids=[mp_rank])

        ################################################
        # training options ( criteria ,optimizers.....)
        ################################################

        criteria = {
            "CrossEntropyLoss": nn.CrossEntropyLoss()
        }

        optimizers = {
            "Adam": optim.Adam(model.parameters(), lr=learning_rate)
        }
        optimizer = optimizers[config.get("optimizer")]
        criterion = criteria[config.get("criterion")]

        #########
        # MlFlow
        #########

        # MlFlow params
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("input_size", input_size)
        mlflow.log_param("clip_grad", clip_grad)
        mlflow.log_param("dataset version", dataset_version)
        mlflow.log_params({"torch_version": "1.12.1+cu116"})

        # MlFlow tags
        mlflow.set_tag('augment', augment)
        mlflow.set_tag('clip_grad', clip_grad)
        mlflow.set_tag('auto_cast', auto_cast)

        lr_scheduler = None
        if use_lr_scheduler:
            # adjust the learning rate during training to improve performance..convergence
            # StepLR:reduces the learning rate by a factor of gamma every step_size epochs.
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer, step_size=epochs // 4, gamma=lr_scheduler_gamma)

        """
        - to parallelize operations on the CPU
        - If you are using a distributed training framework, such as PyTorch DistributedDataParallel, 
            then you should set the number of threads to be equal to the number of workers in the distributed training job
        """
        if distributed_lr:
            torch.set_num_threads(2)
        mp_rank = 0

        # making a log folder labeled with time of trial
        log_folder_path = results_path
        if distributed_lr is False or mp_rank == 0:
            log_folder_path = make_log_folder(results_path, trial_name)

        ###########
        # Export
        ###########
        if mode == 'export' and mp_rank == 0:
            dummy_input = torch.rand(
                (batch_size, in_channels, input_size, input_size))
            checkpoint = torch.load(weights_path,map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            model.to("cpu")
            onnx_export(model, dummy_input, results_path)

        if mp_rank == 0:
            # saving hyperparameters file so that it can be used later to reproduce the trial
            shutil.copyfile(path, os.path.join(
                log_folder_path, "model_config.yaml"))

        ###########
        # Train
        ###########
        if mode == "train" or mode == "overfit":
            train_loader, valid_loader = get_data_loaders(
                dataset_path, batch_size, train_ratio=train_validate_ratio, test_validate_ratio=test_validate_ratio, mode=mode, augment=augment, input_size=input_size,
                random_state=random_seed, distributed_lr=distributed_lr)

            train_loss, validation_loss = train(model=model, train_loader=train_loader, test_loader=valid_loader,
                                                optimizer=optimizer, criterion=criterion, out_channels=out_channels, save_weight_path=weights_path,
                                                log_folder_path=log_folder_path, trial_name=trial_name, used_device=device, distributed_lr=distributed_lr,
                                                lr_scheduler=lr_scheduler,
                                                epochs=epochs, auto_cast=auto_cast, clip_grad=clip_grad, log_period=log_period, mlflow=mlflow)

        if distributed_lr:
            import torch.distributed as dist
            # destroying and de-initializing distributed package
            dist.destroy_process_group()

        ###########
        # Test
        ###########
        if mode == 'test':
            checkpoint = torch.load(weights_path,map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            model.to("cpu")
            test_loader = get_data_loaders(dataset_path, batch_size=1, train_ratio=train_validate_ratio, test_validate_ratio=test_validate_ratio,
                                           mode=mode, input_size=input_size, random_state=random_seed, distributed_lr=distributed_lr)
            test(model=model, test_loader=test_loader,
                 criterion=criterion, visualization_path=log_folder_path, mlflow=mlflow)

        ###########
        # Distillation
        ###########
        if mode == 'distil':
            checkpoint = torch.load(weights_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            train_loader, valid_loader = get_data_loaders(
                dataset_path, batch_size, train_ratio=train_validate_ratio, test_validate_ratio=test_validate_ratio, mode="overfit", augment=augment, input_size=input_size,
                random_state=random_seed, distributed_lr=distributed_lr)

            train_loss, validation_loss = train_distillation(teacher_model=model,student_model=light_model, train_loader=train_loader, test_loader=valid_loader,
                                                optimizer=optimizer, criterion=criterion, out_channels=out_channels, save_weight_path=weights_path,
                                                log_folder_path=log_folder_path, trial_name=trial_name, used_device=device, distributed_lr=distributed_lr,
                                                lr_scheduler=lr_scheduler,
                                                epochs=epochs, auto_cast=auto_cast, clip_grad=clip_grad, log_period=log_period, mlflow=mlflow,temperature=2, soft_target_loss_weight=0.25, ce_loss_weight=0.75)            
            

if __name__ == "__main__":
    main()
    #del os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"]
