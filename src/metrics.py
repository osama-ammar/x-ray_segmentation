from typing import Literal, Sequence
from torch import Tensor, device
from torchmetrics import Metric

import torch
import torchmetrics
from torchmetrics.detection import IntersectionOverUnion


class UnetMetrics:
    """here we define metrics instances, and later we will use them to compute metrics,
    the returned objects used during training like [ object.update(outputs,truth) ]
    then after training we [ object.compute()] to get the final results

    """

    def __init__(self, num_classes: int):
        self.device: device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
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
