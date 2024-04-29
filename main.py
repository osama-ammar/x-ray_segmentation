from src.runner import Segmentation
from src.dataset import MIPDataModule
from src.model_operations import onnx_export , use_onnx , post_quantize#,onnx_to_quantized
from src.training_utils import make_log_folder ,visualize_model_output ,visualize_onnx_output
from src.metrics import detailed_testing
import subprocess
import mlflow
import pytorch_lightning as pl
import yaml
import shutil
import os
import warnings
from typing import Dict
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.strategies import DDPStrategy
import torch
from pytorch_lightning.callbacks import ModelPruning



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
    early_stopping= config.get("early_stopping")
    pruning=config.get("pruning")

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
        "U_Net": Segmentation(in_channels, out_channels,log_folder_path,weights_path,mlflow_logger)
    }
    # model initializing
    # rank of the running process in multiple-processing setting or 0 for single-processing
    model = models[config.get("model")].to(device)

    ###########
    # Export
    ###########
    if mode == 'export' :
        dummy_input = torch.rand(
            (batch_size, in_channels, input_size, input_size))
        checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
        model=model.model
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to("cpu")
        onnx_path=os.path.join(results_path + "\\model.onnx")
        onnx_export(model, dummy_input, onnx_path)
        
        #quantizing onnx model USING onnxruntime.quantization
        # 1st run[python -m onnxruntime.quantization.preprocess --input results/model.onnx --output results/model_preprocessed.onnx   --skip_symbolic_shape SKIP_SYMBOLIC_SHAPE] to preprocess
        preprocessed_onnx_path=os.path.join(results_path + "\\model_preprocessed.onnx")
        quantized_onnx_path=os.path.join(results_path + "\\model_quantized.onnx")
        
        from onnxruntime.quantization import quantize_dynamic
        from onnxruntime.quantization import QuantType
        quantized_model = quantize_dynamic(
            preprocessed_onnx_path,
            quantized_onnx_path,
            #optimize_model=True,
            per_channel=False,
            reduce_range=False,
            weight_type=QuantType.QUInt8,)



    ###########
    # Train
    ###########
    callbacks_n=[]
    if pruning:
        print("include pruning")
        pruning = ModelPruning("l1_unstructured", amount=0.5)
        callbacks_n.append(pruning)
    if early_stopping:
        early_stopping = EarlyStopping(monitor="", mode="min", min_delta=0.001, patience=5)
        callbacks_n.append(early_stopping)
        
        
    if mode == "train" or mode =="overfit":
        
        
        ddp = DDPStrategy(process_group_backend="gloo")

        data = MIPDataModule(dataset_path, batch_size=batch_size,
                            input_size=input_size, train_validate_ratio=train_validate_ratio,
                            test_validate_ratio=test_validate_ratio,mode=mode)

        trainer = pl.Trainer(max_epochs=epochs,
                            enable_progress_bar=True,
                            accelerator="gpu",
                            logger=mlflow_logger,
                            log_every_n_steps=log_period,
                            limit_train_batches=6 if mode == 'overfit' else None,
                            limit_val_batches=6 if mode == 'overfit' else None,
                            callbacks=callbacks_n if len(callbacks_n)>0 else None,
                            devices=[0, 1] if distributed_lr else "auto",
                            strategy=ddp if distributed_lr else "auto",
                            default_root_dir=log_folder_path)

        print("preparing fitting")
        trainer.fit(model, data)

    ###########
    # Test
    ###########

    if mode == 'test':
        checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.test(model, data)

    if mode=="test_onnx":
        import torch.nn as nn
        sub_dataset = MIPDataModule(dataset_path, batch_size=1,
        input_size=input_size, train_validate_ratio=train_validate_ratio,
        test_validate_ratio=test_validate_ratio,mode="overfit")
        
        sub_dataset.prepare_data()
        sub_dataset.setup("validate")
        validate_dataloader=sub_dataset.val_dataloader()

        model=model.model
        
        checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        #trainer.test(model, data)
        criterion = nn.CrossEntropyLoss()
        detailed_testing(model,validate_dataloader,criterion,log_folder_path,onnx_compare=True,onnx_path="results/model_quantized.onnx",used_device="cpu")

    if mode=="post_quantize":
        sub_dataset = MIPDataModule(dataset_path, batch_size=1,
        input_size=input_size, train_validate_ratio=train_validate_ratio,
        test_validate_ratio=test_validate_ratio,mode="overfit")
        
        sub_dataset.prepare_data()
        sub_dataset.setup("validate")
        validate_dataloader=sub_dataset.val_dataloader()
        post_quantize(model,validate_dataloader)
        
    # saving hyperparameters file so that it can be used later to reproduce the trial
    shutil.copyfile(path, os.path.join(
        log_folder_path, "model_config.yaml"))
    
if __name__ == "__main__":
    main()
