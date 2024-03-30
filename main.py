from src.runner import Segmentation
from src.dataset import MIPDataModule
from src.model_operations import onnx_export
from src.training_utils import make_log_folder
import subprocess
import mlflow
import pytorch_lightning as pl
import yaml
import shutil
import os
import warnings
from typing import Dict
from pytorch_lightning.loggers import MLFlowLogger
import torch
# Ignore the user warning about the missing audio backend
warnings.filterwarnings("ignore", category=UserWarning,
                        message="No audio backend is available.", module="torchaudio")
warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated", category=UserWarning)
mlflow.enable_system_metrics_logging()

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
    
    log_folder_path = make_log_folder(results_path, trial_name)
    mp_rank = 0
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

    #########
    # MlFlow
    #########
    #mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.start_run()
    mlflow_logger = MLFlowLogger(
        experiment_name=trial_name,
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
        run_id=mlflow.active_run().info.run_id,
        artifact_location=log_folder_path
    )

    # MlFlow params
    mlflow_logger.log_hyperparams({
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "input_size": input_size,
        "clip_grad": clip_grad,
        "dataset version": dataset_version,
        "torch_version": f"torch=={str(torch.__version__)}"
    })   
    
    ###########
    # Model
    ###########
    models = {
        "U_Net": Segmentation(in_channels, out_channels,log_folder_path,mlflow_logger)
    }
    # model initializing
    # rank of the running process in multiple-processing setting or 0 for single-processing
    model = models[config.get("model")].to(device)

    ###########
    # Export
    ###########
    if mode == 'export' and mp_rank == 0:
        dummy_input = torch.rand(
            (batch_size, in_channels, input_size, input_size))
        checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to("cpu")
        onnx_export(model, dummy_input, log_folder_path)

    if mp_rank == 0:
        # saving hyperparameters file so that it can be used later to reproduce the trial
        shutil.copyfile(path, os.path.join(
            log_folder_path, "model_config.yaml"))

    ###########
    # Train
    ###########
    data = MIPDataModule(dataset_path, batch_size=batch_size,
                            input_size=input_size, train_validate_ratio=train_validate_ratio,test_validate_ratio=test_validate_ratio,num_workers=1)
    trainer = pl.Trainer(max_epochs=epochs,
                        enable_progress_bar=True,
                        devices=1,
                        accelerator="gpu",
                        limit_train_batches = 6 if mode=='overfit' else None,
                        limit_val_batches = 3 if mode=='overfit' else None,
                        logger=mlflow_logger,
                        log_every_n_steps=log_period)  # precision=16)
    
    trainer.fit(model, data)

    ###########
    # Test
    ###########
    if mode == 'test':
        checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.test(model, data)


if __name__ == "__main__":
    main()
