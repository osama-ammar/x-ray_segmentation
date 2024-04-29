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
import matplotlib.pyplot as plt
from numpy.testing import assert_allclose as numpy_assert_allclose
from torch.testing import assert_close as torch_assert_close

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


def onnx_export(model: torch.nn.Module, dummy_input: torch.Tensor, onnx_path: AnyStr):
    """ export model to onnx format"""
    input_names = ["input"]
    output_names = ["output"]
    # dynamic_axes to make the onnx model accept different input batch sizes
    dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
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
        opset_version=17,
        verbose=False,    
        
    )
    # Checks
    model_onnx = load_onnx(onnx_path)  # load onnx model
    check_model(model_onnx)  # check onnx model
    print("Model exported to ONNX format.")

# This function will allow us to use the same PyTorch DataLoader with ONNX
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def compare_pt_onnx_outputs(pt_model,pt_model_path, onnx_model_path, input_data):

    with torch.no_grad():
        
        # load pt model
        checkpoint = torch.load(pt_model_path, map_location='cpu')
        pt_model.load_state_dict(checkpoint['state_dict'])
        # Map models to CPU
        pt_model.eval()
        pt_model.to('cpu')
        # delete checkpoint
        del checkpoint
        # Forward pass with PyTorch model
        pt_output = pt_model(input_data)

        # Forward pass with ONNX model
        import onnxruntime
        ort_session = onnxruntime.InferenceSession(
            onnx_model_path, providers=['CPUExecutionProvider'])
        ort_inputs = {
            ort_session.get_inputs(
            )[0].name: input_data.cpu().numpy()
        }
        ort_output = ort_session.run(None, ort_inputs)[0]

        ###############################################
        # check that torch and onnx results are equal #
        ###############################################
        torch_assert_close(
            pt_output,
            torch.from_numpy(ort_output),
            rtol=1e-03,
            atol=1e-05
        )
        numpy_assert_allclose(
            pt_output.cpu().numpy(),
            ort_output,
            rtol=1e-03,
            atol=1e-05
        )

        print(
            "The outputs of PyTorch and ONNX models are equal. Congratulations along way to go!")

        




def load_model_weights(model,model_path):

    # load pretrained weights to it
    weights_path = model_path
    checkpoint = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])

    # set the model to evaluation mode
    model.train(False)
    model.eval()
    return model


def model_inferences(input):
    # load model architecture and its weights
    model = load_model_weights()

    # prepare input (2,400,400)-->(1,2,400,400) and get the output
    input = input.unsqueeze(dim=0)
    logits = model(input)  # -->(B,17,400,400)
    output = F.softmax(logits, dim=1)  # -->(B,17,400,400)
    # print (torch.unique(output))
    output = torch.argmax(output, dim=1).squeeze().to(torch.float32)  # -->(B,400,400) pixels values(1-->17)
    # print (torch.unique(output))

    # show the output
    _, axs = plt.subplots(1, 1, figsize=(10, 10))
    axs.imshow(output.cpu().numpy(), cmap="gray")
    plt.tight_layout()
    plt.show()
    
    
def use_onnx(onnx_model_path: AnyStr, input_data: npt.NDArray) -> npt.NDArray:
    """ using onnx in inference mode"""
    # Run the ONNX model with the dummy input tensor
    session = onnxrt.InferenceSession(onnx_model_path)
    # input_names = ["input"]
    # output_names = ["output"]
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_data=to_numpy(input_data)
    print(input_data.shape)
    input_data=[input_data]
    output = session.run(None, {input_name: input_data[0]})
    #output=session.run(output_names, {input_names: input_image})
    return output



# post quaization using Intel neural compressor
from neural_compressor.quantization import fit as fit
from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion, AccuracyCriterion

def eval_func(model,dataloader,metric):
    for input, label in dataloader:
        output = model(input)
        metric.update(output, label)
    accuracy = metric.result()
    return accuracy

def post_quantize(model,val_dataloader):
    accuracy_criterion = AccuracyCriterion(tolerable_loss=0.1)
    tuning_criterion = TuningCriterion(max_trials=1200)
    conf = PostTrainingQuantConfig(
        approach="static", backend="default", tuning_criterion=tuning_criterion, accuracy_criterion=accuracy_criterion
    )
    q_model = fit(model=model.model, conf=conf, calib_dataloader=val_dataloader)
    print(q_model)
    q_model.save("./saved_model/")



