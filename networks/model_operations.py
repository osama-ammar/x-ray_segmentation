from typing import AnyStr
import os

import onnxruntime as onnxrt
from onnx import load as load_onnx
from onnx.checker import check_model

import torch
import numpy.typing as npt


def save_ckp(checkpoint_path: AnyStr, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int) -> None:
    state = {
        'model_state_dict': model.to('cpu').state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoc': epoch,
    }
    torch.save(state, checkpoint_path)


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
