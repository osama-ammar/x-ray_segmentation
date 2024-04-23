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
    )
    # Checks
    model_onnx = load_onnx(onnx_path)  # load onnx model
    check_model(model_onnx)  # check onnx model
    print("Model exported to ONNX format.")


def use_onnx(onnx_model_path: AnyStr, input_data: npt.NDArray) -> npt.NDArray:
    """ using onnx in inference mode"""
    # Run the ONNX model with the dummy input tensor
    session = onnxrt.InferenceSession(onnx_model_path)
    input_names = ["input"]
    output_names = ["output"]
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output = session.run([output_name], {input_name: input_data})
    #output=session.run(output_names, {input_names: input_image})
    return output

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
    
    
# This function will allow us to use the same PyTorch DataLoader with ONNX
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


import numpy as np 
from tqdm import tqdm
import onnxruntime as ort
# from onnxruntime import quantization

# def onnx_test_cpu_gpu(model_pt,onnx_path,val_ds):
#     ort_provider = ['CPUExecutionProvider']
#     if torch.cuda.is_available():
#         model_pt.to('cuda')
#         ort_provider = ['CUDAExecutionProvider']

#     ort_sess = ort.InferenceSession(onnx_path, providers=ort_provider)
#     correct_pt = 0
#     correct_onnx = 0
#     tot_abs_error = 0

#     for img_batch, label_batch in tqdm(dl, ascii=True, unit="batches"):

#         ort_inputs = {ort_sess.get_inputs()[0].name: to_numpy(img_batch)}
#         ort_outs = ort_sess.run(None, ort_inputs)[0]

#         ort_preds = np.argmax(ort_outs, axis=1)
#         correct_onnx += np.sum(np.equal(ort_preds, to_numpy(label_batch)))

#         if torch.cuda.is_available():
#             img_batch = img_batch.to('cuda')
#             label_batch = label_batch.to('cuda')

#         with torch.no_grad():
#             pt_outs = model_pt(img_batch)

#         pt_preds = torch.argmax(pt_outs, dim=1)
#         correct_pt += torch.sum(pt_preds == label_batch)

#         tot_abs_error += np.sum(np.abs(to_numpy(pt_outs) - ort_outs))

#     print("\n")

#     print(f"pt top-1 acc = {100.0 * correct_pt/len(val_ds)} with {correct_pt} correct samples")
#     print(f"onnx top-1 acc = {100.0 * correct_onnx/len(val_ds)} with {correct_onnx} correct samples")

#     mae = tot_abs_error/(1000*len(val_ds))
#     print(f"mean abs error = {mae} with total abs error {tot_abs_error}")
    
    


# """
# Since our model is mainly a CNN, we should perform static quantization. But, it requires a dataset to calibrate the quantized model parameters.
# """
# class QuntizationDataReader(quantization.CalibrationDataReader):
#     def __init__(self, torch_ds, batch_size, input_name):

#         self.torch_dl = torch.utils.data.DataLoader(torch_ds, batch_size=batch_size, shuffle=False)

#         self.input_name = input_name
#         self.datasize = len(self.torch_dl)

#         self.enum_data = iter(self.torch_dl)

#     def to_numpy(self, pt_tensor):
#         return pt_tensor.detach().cpu().numpy() if pt_tensor.requires_grad else pt_tensor.cpu().numpy()

#     def get_next(self):
#         batch = next(self.enum_data, None)
#         if batch is not None:
#             return {self.input_name: self.to_numpy(batch[0])}
#         else:
#             return None

#     def rewind(self):
#         self.enum_data = iter(self.torch_dl)

# def onnx_to_quantized( onnx_path, quantized_onnx_output_path,calib_ds):
#     quantization.shape_inference.quant_pre_process(onnx_path, quantized_onnx_output_path, skip_symbolic_shape=False)
#     ort_provider = ['CPUExecutionProvider'] if torch.cuda.is_available() else ['CUDAExecutionProvider']
#     ort_sess = ort.InferenceSession(onnx_path, providers=ort_provider)
#     qdr = QuntizationDataReader(calib_ds, batch_size=4, input_name=ort_sess.get_inputs()[0].name)

#     q_static_opts = {"ActivationSymmetric":False,"WeightSymmetric":True}
#     if torch.cuda.is_available():
#         q_static_opts = {"ActivationSymmetric":True,
#                         "WeightSymmetric":True}

#     model_int8_path = 'resnet18_int8.onnx'
#     quantized_model = quantization.quantize_static(model_input=quantized_onnx_output_path,
#                                                 model_output=model_int8_path,
#                                                 calibration_data_reader=qdr,
#                                                 extra_options=q_static_opts)
    
    
    