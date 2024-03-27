import evaluate
from loader import visualize_model_output, bbox_from_mask, save_online_logs
import torch
from torch import Tensor, device, no_grad, set_grad_enabled
from torch.nn import Module
from torch.utils.data import DataLoader
from typing import List, Optional, Sequence, Tuple, Union, Callable
from tqdm import tqdm


def test(
    model: Module,
    test_loader: DataLoader,
    criterion: Callable,
    visualization_path=None,
    mlflow=None,
    used_device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
) -> float:

    mp_rank = 0
    test_metrics = evaluate.UnetMetrics(num_classes=2)

    detection_metrics = evaluate.DetectionMetrics()
    # setting model to evaluation mode
    model.train(False)
    model.eval()
    iter = 0
    tqdm_test_loader = tqdm(test_loader, disable=(mp_rank != 0))
    test_loss = 0.0
    with set_grad_enabled(False):
        for item in tqdm(tqdm_test_loader):

            data = item["img"].to(used_device).unsqueeze(1)
            mask = item["mask"].to(used_device).long()
            case_name = item["case_name"]

            # Forward pass.
            outputs: Tensor = model(data)

            # getting bboxes from masks
            mask_bboxes = bbox_from_mask(mask.to("cpu"), logits=False)
            output_bboxes = bbox_from_mask(outputs.to("cpu"), logits="logits")

            # Calculate the loss.
            loss: Tensor = criterion(outputs, mask)
            test_loss += loss.item()

            # update metrics(segmentation & bboxes)
            test_metrics(outputs, mask)
            detection_metrics(output_bboxes, mask_bboxes)

            header = ["case", "accuracy", "precision",
                      "iou segmentation", "iou detection"]
            log_line = [f"{case_name}", f"{test_metrics.compute()[0]}", f"{test_metrics.compute()[1]}",
                        f"{test_metrics.compute()[3]}", f"{detection_metrics.compute()}"]
            save_online_logs(visualization_path, header, log_line)

            # mlflow stuff
            mlflow.log_artifacts(visualization_path+"\\training_imgs")
            mlflow.log_artifacts(visualization_path+"\\validation_imgs")
            metrics_dict = {
                'test_accuracy': test_metrics.compute()[0],
                'test_precision': test_metrics.compute()[1],
                'test_iou_segmentation': test_metrics.compute()[3],
                'test_iou_box': detection_metrics.compute(),
            }
            mlflow.log_metrics(metrics_dict, synchronous=False)

            visualize_model_output(data, mask, outputs, visualization_path, epoch=iter, training=False,
                                   case_name=case_name, gt_rect=mask_bboxes, pred_rect=output_bboxes)
            iter += 1
            
    test_loss = test_loss/len(test_loader)
    test_accuracy, test_precision, test_recall, test_iou, test_dice = test_metrics.compute()
    detection_metrics = detection_metrics.compute()

    print(f"test_loss:{test_loss}, test_accuracy:{test_accuracy}, test_precision:{test_precision}, test_recall:{test_recall}, test_iou:{test_iou}, test_dice:{test_dice} ,detection_metrics:{detection_metrics}")
    return test_loss, test_accuracy, test_precision, test_recall, test_iou, test_dice, detection_metrics

