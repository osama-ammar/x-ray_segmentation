from onnxruntime.quantization.calibrate import CalibrationDataReader
import numpy as np
from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion, AccuracyCriterion
import os
import onnxruntime as onnxrt
from onnx import load as load_onnx
from onnx.checker import check_model
from typing import AnyStr
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
from tqdm import tqdm
import onnx
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



def load_model_weights(model, model_path):

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
    output = torch.argmax(output, dim=1).squeeze().to(
        torch.float32)  # -->(B,400,400) pixels values(1-->17)
    # print (torch.unique(output))

    # show the output
    _, axs = plt.subplots(1, 1, figsize=(10, 10))
    axs.imshow(output.cpu().numpy(), cmap="gray")
    plt.tight_layout()
    plt.show()

####################
#      ONNX        #
####################

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def use_onnx(onnx_model_path: AnyStr, input_data: npt.NDArray) -> npt.NDArray:
    """ using onnx in inference mode"""
    # Run the ONNX model with the dummy input tensor
    session = onnxrt.InferenceSession(onnx_model_path)
    # input_names = ["input"]
    # output_names = ["output"]
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_data = to_numpy(input_data)
    input_data = [input_data]
    output = session.run(None, {input_name: input_data[0]})
    #output=session.run(output_names, {input_names: input_image})
    return output



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
def compare_pt_onnx_outputs(pt_model, pt_model_path, onnx_model_path, input_data):

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
            "The outputs of PyTorch and ONNX models are equal.")
        
    

####################################
#          Quantization            #
####################################

# post quaization using Intel neural compressor
def eval_func(model, dataloader, metric):
    for input, label in dataloader:
        output = model(input)
        metric.update(output, label)
    accuracy = metric.result()
    return accuracy


# used for data calibration -required for static quantization-
class DataReader(CalibrationDataReader):
    def __init__(self, input_name, imgs):
        self.input_name = input_name
        self.data = imgs
        self.pos = -1

    def get_next(self):
        if self.pos >= len(self.data) - 1:
            return None
        self.pos += 1
        return {self.input_name: self.data[self.pos]}

    def rewind(self):
        self.pos = -1

# required for static quantization
def get_calibration_data_reader(input_name, input_shape):
    maxN = 50
    imgs = [np.random.rand(*input_shape).astype(np.float32)
            for i in range(maxN)]
    calibration_data_reader = DataReader(input_name, imgs)
    return calibration_data_reader


# for quantizing onnx model. input_shape can be ignored in case of dynamic quantization
def post_quantize(onnx_path, quantization_mode='static', input_shape=None):
    from onnxruntime.quantization import quantize_dynamic, quantize_static
    from onnxruntime.quantization import QuantType
    from onnxruntime.quantization.quant_utils import QuantFormat
    import subprocess

    preprocessed_onnx_path=os.path.join(os.path.dirname(onnx_path) , "model_preprocessed.onnx")
    quantized_onnx_path=os.path.join(os.path.dirname(onnx_path) , "model_quantized.onnx")
    
    #this step is important before quantization to optimize the model ex(merging some components in the model)
    try:
        pre_process_command = f"python -m onnxruntime.quantization.preprocess --input {onnx_path} --output {preprocessed_onnx_path}  --skip_symbolic_shape SKIP_SYMBOLIC_SHAPE"
        subprocess.run(pre_process_command, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, text=True, check=True, cwd=os.path.dirname(onnx_path) )

    except subprocess.CalledProcessError as e:
        print(e.stderr)
            
    if quantization_mode == "dynamic":
        quantized_model = quantize_dynamic(
            preprocessed_onnx_path,
            quantized_onnx_path,
            per_channel=False,
            reduce_range=False,
            weight_type=QuantType.QUInt8,)
    else:
        calibration_data_reader = get_calibration_data_reader(
            input_name="input", input_shape=input_shape)
        quantized_model = quantize_static(
            preprocessed_onnx_path,
            quantized_onnx_path,
            calibration_data_reader=calibration_data_reader,
            quant_format=QuantFormat.QDQ)
        
    
from .training_utils import save_online_logs
# compare onnx model with its quantized version
def compare_onnx_quantized(onnx_path,quantized_onnx_path,test_loader,torch_test_metrics,quantized_torch_test_metrics,report_path,device='cpu'):
    # torch_test_metrics : a redefined torch metrics object
    onnx_header=["case id","onnx_accuracy","onnx_iou segmentation","quantized_onnx_accuracy","quantized_onnx_iou segmentation"]
    tqdm_test_loader = tqdm(test_loader)
  
    for item in tqdm(tqdm_test_loader):

        data = item["img"].to(device).unsqueeze(1).float()
        mask = item["mask"].to(device).long()
        case_name = item["case_id"] 

        #getting onnx output
        onnx_output=use_onnx(onnx_path, data)
        #returning to torch tensors to run eval metrics
        onnx_output=torch.from_numpy(onnx_output[0])
        #calculate metrics
        torch_test_metrics(onnx_output, mask)

        #getting quantized onnx output
        quantized_onnx_output=use_onnx(quantized_onnx_path, data)
        #returning to torch tensors to run eval metrics
        quantized_onnx_output=torch.from_numpy(quantized_onnx_output[0])
        #calculate metrics
        quantized_torch_test_metrics(quantized_onnx_output, mask)
        
        onnx_log_line = [f"{case_name}",f"{torch_test_metrics.compute()[0]}",f"{torch_test_metrics.compute()[3]}",f"{quantized_torch_test_metrics.compute()[0]}",f"{quantized_torch_test_metrics.compute()[3]}"]
        save_online_logs(report_path, onnx_header, onnx_log_line)


from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Callable

def compare_onnx_quantized(onnx_path: str, 
                           quantized_onnx_path: str, 
                           test_loader: DataLoader, 
                           torch_test_metrics: Callable, 
                           quantized_torch_test_metrics: Callable, 
                           report_path: str, 
                           device: str = 'cpu') -> None:
    """
    Compare an ONNX model with its quantized version using evaluation metrics.

    Parameters:
        onnx_path (str): Path to the original ONNX model file.
        quantized_onnx_path (str): Path to the quantized ONNX model file.
        test_loader (DataLoader): DataLoader for test dataset.
        torch_test_metrics (Callable): Torch metrics object for original model evaluation.
        quantized_torch_test_metrics (Callable): Torch metrics object for quantized model evaluation.
        report_path (str): Path to save the comparison report.
        device (str, optional): Device to run inference on. Default is 'cpu'.
    """
    onnx_header = ["case id", "onnx_accuracy", "onnx_iou_segmentation", "quantized_onnx_accuracy", "quantized_onnx_iou_segmentation"]
    tqdm_test_loader = tqdm(test_loader)
  
    for item in tqdm_test_loader:

        data = item["img"].to(device).unsqueeze(1).float()
        mask = item["mask"].to(device).long()
        case_name = item["case_id"] 

        # Getting ONNX output
        onnx_output = use_onnx(onnx_path, data)
        # Returning to torch tensors to run eval metrics
        onnx_output = torch.from_numpy(onnx_output[0])
        # Calculate metrics
        torch_test_metrics(onnx_output, mask)

        # Getting quantized ONNX output
        quantized_onnx_output = use_onnx(quantized_onnx_path, data)
        # Returning to torch tensors to run eval metrics
        quantized_onnx_output = torch.from_numpy(quantized_onnx_output[0])
        # Calculate metrics
        quantized_torch_test_metrics(quantized_onnx_output, mask)
        
        onnx_log_line = [f"{case_name}", f"{torch_test_metrics.compute()[0]}", f"{torch_test_metrics.compute()[3]}", f"{quantized_torch_test_metrics.compute()[0]}", f"{quantized_torch_test_metrics.compute()[3]}"]
        save_online_logs(report_path, onnx_header, onnx_log_line)