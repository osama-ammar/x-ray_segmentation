from typing import Literal, Sequence
from torch import Tensor, device
from torchmetrics import Metric
from .training_utils import bbox_from_mask,save_online_logs,visualize_model_output
import torch
from torch import Tensor, device, no_grad, set_grad_enabled
from torch.nn import Module
from torch.utils.data import DataLoader
import torch
import torchmetrics
from torchmetrics.detection import IntersectionOverUnion
from typing import Optional, Sequence, Tuple, Union, AnyStr, Dict, List ,Callable
from tqdm import tqdm
from src.model_operations import onnx_export , use_onnx #,onnx_to_quantized


class UnetMetrics:
    """here we define metrics instances, and later we will use them to compute metrics,
    the returned objects used during training like [ object.update(outputs,truth) ]
    then after training we [ object.compute()] to get the final results

    """

    def __init__(self, num_classes: int,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.device: device = device
        self.task: Literal["multiclass", "binary"] = "multiclass" if num_classes > 1 else "binary"
        # The default value is "macro", which calculates the unweighted average.
        # The macro average is often used when the classes are imbalanced, because it does not give more weight to the majority class.
        # The micro average is often used when the classes are balanced.
        self.average: Literal["macro", "micro"] = "macro"
        self.out_channels = num_classes

        # metrics (defining metrics instances)
        # note :  accuracy & precision  can be misleading when dealing with imbalanced datasets or outlier classes
        self.confusion_matrix: Metric = torchmetrics.ConfusionMatrix(task=self.task, num_classes=self.out_channels, normalize='all').to(self.device)
        self.accuracy: Metric = torchmetrics.Accuracy(task=self.task, num_classes=self.out_channels).to(self.device)
        self.jaccard_similarity: Metric = torchmetrics.JaccardIndex(task=self.task, num_classes=self.out_channels).to(self.device)  # jaccard_similarity= IOU
        self.dice: Metric = torchmetrics.Dice(num_classes=self.out_channels).to(self.device)
        self.precision: Metric = torchmetrics.Precision(task=self.task, num_classes=self.out_channels, average=self.average).to(self.device)
        self.recall: Metric = torchmetrics.Recall(task=self.task, num_classes=self.out_channels, average=self.average).to(self.device)

    def __call__(self, outputs: Tensor, truth: Tensor) -> None:
        # updating metrics
        self.confusion_matrix.update(outputs, truth)
        self.accuracy.update(outputs, truth)
        self.precision.update(outputs, truth)
        self.recall.update(outputs, truth)
        self.jaccard_similarity.update(outputs, truth)
        self.dice.update(outputs, truth)

    def compute(self) -> Sequence[float]:
        # compute metrics
        accuracy = self.accuracy.compute().item()
        precision = self.precision.compute().item()
        recall = self.recall.compute().item()
        jaccard_similarity = self.jaccard_similarity.compute().item()
        dice = self.dice.compute().item()

        return accuracy, precision, recall, jaccard_similarity, dice

    def get_conf_matrix(self) -> Tensor:
        return self.confusion_matrix.compute()

    def plot_conf_matrix(self):
        self.confusion_matrix.plot()


class DetectionMetrics:
    def __init__(self) -> None:
        self.detection_iou = IntersectionOverUnion()

    def __call__(self, outputs, truth):
        # updating metrics
        self.detection_iou.update(outputs, truth)

    def compute(self):
        # compute metrics
        return self.detection_iou.compute()["iou"].item()


# run eval metrics on pytorch model or pytorch vs onnx model
def detailed_evaluation(
    model: Module,
    test_loader: DataLoader,
    criterion: Callable,
    visualization_path=None,
    mlflow=None,
    onnx_compare=False,
    onnx_path=None,
    used_device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
) -> float:

    mp_rank = 0
    
    test_metrics = UnetMetrics(num_classes=2,device=used_device)
    detection_metrics = DetectionMetrics()

    onnx_test_metrics = UnetMetrics(num_classes=2,device=used_device)
    onnx_detection_metrics = DetectionMetrics()
        
    # setting model to evaluation mode
    model.train(False)
    model.eval()
    model.to(used_device)
    iter = 0
    tqdm_test_loader = tqdm(test_loader, disable=(mp_rank != 0))
    test_loss = 0.0
    with set_grad_enabled(False):
        for item in tqdm(tqdm_test_loader):

            data = item["img"].to(used_device).unsqueeze(1).float()
            mask = item["mask"].to(used_device).long()
            case_name = item["case_id"]

            # Forward pass.
            output: Tensor = model(data)

            # getting bboxes from masks
            mask_bboxes = bbox_from_mask(mask, logits=False)
            output_bboxes = bbox_from_mask(output, logits="logits")

            # Calculate the loss.
            loss: Tensor = criterion(output, mask)
            test_loss += loss.item()
            
            # getting pytorch metrics
            test_metrics(output, mask)
            detection_metrics(output_bboxes, mask_bboxes)

            header = ["case", "accuracy","iou segmentation", "iou detection"]
            log_line = [f"{case_name}", f"{test_metrics.compute()[0]}",f"{test_metrics.compute()[3]}", f"{detection_metrics.compute()}"]
            
            if onnx_compare:
                # getting onnx metrics
                onnx_output=use_onnx(onnx_path, data)
                onnx_output=torch.from_numpy(onnx_output[0])
                onnx_output_bboxes = bbox_from_mask(onnx_output.to("cpu"), logits="logits")
                onnx_test_metrics(onnx_output, mask)
                onnx_detection_metrics(onnx_output_bboxes, mask_bboxes)
                onnx_header=["onnx_accuracy","onnx_iou segmentation", "onnx_iou detection"]
                onnx_log_line = [f"{onnx_test_metrics.compute()[0]}",f"{onnx_test_metrics.compute()[3]}", f"{onnx_detection_metrics.compute()}"]
                header.extend(onnx_header)
                log_line.extend(onnx_log_line)
    
            save_online_logs(visualization_path, header, log_line)
            
            # mlflow stuff
            if mlflow:
                mlflow.log_artifacts(visualization_path+"\\training_imgs")
                mlflow.log_artifacts(visualization_path+"\\validation_imgs")
                metrics_dict = {
                    'test_accuracy': test_metrics.compute()[0],
                    'test_precision': test_metrics.compute()[1],
                    'test_iou_segmentation': test_metrics.compute()[3],
                    'test_iou_box': detection_metrics.compute(),
                }
                mlflow.log_metrics(metrics_dict, synchronous=False)

            visualize_model_output(data, mask, onnx_output.to("cpu"), visualization_path, epoch=iter, training=False,
                                   case_name=case_name)
            iter += 1
            
    test_loss = test_loss/len(test_loader)
    test_accuracy, test_precision, test_recall, test_iou, test_dice = test_metrics.compute()
    detection_metrics = detection_metrics.compute()

    print(f"test_loss:{test_loss}, test_accuracy:{test_accuracy}, test_precision:{test_precision}, test_recall:{test_recall}, test_iou:{test_iou}, test_dice:{test_dice} ,detection_metrics:{detection_metrics}")
    return test_loss, test_accuracy, test_precision, test_recall, test_iou, test_dice, detection_metrics

