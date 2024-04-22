import os
import onnxruntime as onnxrt
from onnx import load as load_onnx
from onnx.checker import check_model
from typing import  AnyStr
import torch
import numpy.typing as npt
from torch.nn import Module
import torch
from torch.types import _size as torch_size
from neural_compressor.quantization import fit as fit
from neural_compressor.config import PostTrainingQuantConfig


def save_ckp(checkpoint_path: AnyStr, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int) -> None:
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoc': epoch,
    }
    torch.save(state, checkpoint_path)

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


def load_ckp(checkpoint_path: AnyStr, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    """to load model and optimizer last states to resume training if needed"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']


def onnx_export(model: torch.nn.Module, dummy_input: torch.Tensor, save_path: AnyStr):
    """ export model to onnx format"""
    input_names = ["input"]
    output_names = ["output"]
    # dynamic_axes to make the onnx model accept different input batch sizes
    dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    onnx_path = os.path.join(save_path + "\\model.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        export_params=True,
        keep_initializers_as_inputs=False,
        do_constant_folding=True,
        opset_version=16,
    )
    # Checks
    model_onnx = load_onnx(onnx_path)  # load onnx model
    check_model(model_onnx)  # check onnx model
    print("Model exported to ONNX format.")


def use_onnx(onnx_model_path: AnyStr, input_image: npt.NDArray) -> npt.NDArray:
    """ using onnx in inference mode"""
    # Run the ONNX model with the dummy input tensor
    session = onnxrt.InferenceSession(onnx_model_path)
    input_names = ["input"]
    output_names = ["output"]
    return session.run(output_names, {input_names: input_image})

# post quaization using Intel neural compressor
from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion, AccuracyCriterion

accuracy_criterion = AccuracyCriterion(tolerable_loss=0.01)
tuning_criterion = TuningCriterion(max_trials=600)
conf = PostTrainingQuantConfig(approach="static", backend="default", tuning_criterion=tuning_criterion, accuracy_criterion=accuracy_criterion)

def eval_func_for_post_quant(model,model_n, trainer_n,data_module):
    setattr(model, "model", model_n)
    result = trainer_n.validate(model=model, dataloaders=data_module.val_dataloader())
    return result[0]["accuracy"]


def post_quantize(model,model_n,trainer,data_module):
    conf = PostTrainingQuantConfig()
    q_model = fit(model=model.model, conf=conf, calib_dataloader=data_module.val_dataloader(), eval_func=eval_func_for_post_quant(model,model_n,trainer,data_module))
    q_model.save("./results/")
    
    
    
    