import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from .training_utils import bbox_from_mask, visualize_model_output, save_online_logs
from .metrics import UnetMetrics, DetectionMetrics
from .networks import SMP

class Segmentation(pl.LightningModule):
    #defining model and some requirements
    def __init__(self, in_channels, out_channels, log_path,mlflow_logger,use_lr_scheduler=False):
        super(Segmentation, self).__init__()
        self.model = SMP(in_channels, out_channels)
        self.criterion = nn.CrossEntropyLoss()
        self.unet_metrices = UnetMetrics(out_channels)
        self.detection_metrics = DetectionMetrics()
        self.training_loss = 0.0
        self.validation_loss = 0.0
        self.test_loss = 0.0
        self.log_path = log_path
        self.mlflow_logger = mlflow_logger
        self.use_lr_scheduler = use_lr_scheduler

    #model forward
    def forward(self, x):
        return self.model(x)


    #Defines a common step for processing a batch of data during training, validation, or testing
    def common_step(self, batch, batch_index):
        data = batch["img"].unsqueeze(1).float()
        if len(data.shape) ==5:
            data = torch.matmul(data[..., :3], torch.tensor([0.2989, 0.5870, 0.1140], device=data.device))
            print(data.shape)
            
            print("modified------------------")
        mask = batch["mask"].long()
        case_id=batch["case_id"]
        print(data.shape)
        print(mask.shape)
        print(case_id,'\n',"==========")
        
        

        # Forward pass.
        outputs = self.forward(data)

        # Remove the unnecessary singleton dimension if present
        outputs = outputs.squeeze(1)
        # Calculate the loss
        loss = self.criterion(outputs, mask)

        gt_bboxes = bbox_from_mask(mask)
        pred_bboxes = bbox_from_mask(outputs, True)
        self.unet_metrices(outputs, mask)
        accuracy, precision, recall, jaccard_similarity, dice = self.unet_metrices.compute()
        self.detection_metrics(pred_bboxes, gt_bboxes)
        detection_iou = self.detection_metrics.compute()
        
        self.mlflow_logger.experiment.log_artifact(self.mlflow_logger.run_id, self.log_path)
        
        return {
            "loss": loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "jaccard_similarity": jaccard_similarity,
            "dice": dice,
            "iou": detection_iou,
            "outputs": outputs
        }
        
    def training_step(self, batch, batch_idx):
        result = self.common_step(batch, batch_idx)
        self.training_loss += result["loss"].item()
        self.logger.log_metrics({
            "loss": result["loss"].item(),
            "accuracy": result["accuracy"],
            "precision": result["precision"],
            "recall": result["recall"],
            "jaccard_similarity": result["jaccard_similarity"],
            "dice": result["dice"],
            "iou": result["iou"]
        }, step=batch_idx)
        visualize_model_output(batch["img"],
                               batch["mask"],
                               result["outputs"],
                               self.log_path,
                               batch_idx,
                               True,
                               batch["case_id"][0])
        return result

    def validation_step(self, batch, batch_idx):
        result = self.common_step(batch, batch_idx)
        self.validation_loss += result["loss"].item()
        self.logger.log_metrics({
            "loss": result["loss"].item(),
            "accuracy": result["accuracy"],
            "precision": result["precision"],
            "recall": result["recall"],
            "jaccard_similarity": result["jaccard_similarity"],
            "dice": result["dice"],
            "iou": result["iou"]
        }, step=batch_idx)
        visualize_model_output(batch["img"], batch["mask"],
                               result["outputs"],self.log_path,
                               batch_idx, False, batch["case_id"])
        return result

    def test_step(self, batch, batch_idx):
        result = self.common_step(batch, batch_idx)
        self.test_loss += result["loss"].item()
        self.logger.log_metrics({
            "loss": result["loss"].item(),
            "accuracy": result["accuracy"],
            "precision": result["precision"],
            "recall": result["recall"],
            "jaccard_similarity": result["jaccard_similarity"],
            "dice": result["dice"],
            "iou": result["iou"]
        }, step=batch_idx)
        print(f"saving dir {self.logger.save_dir}")
        save_online_logs(self.logger.save_dir, ["case", "accuracy", "precision",
                                                "iou segmentation", "iou detection"],
                         [f"{batch['case_id']}", f"{result['accuracy']}", f"{result['precision']}",
                          f"{result['jaccard_similarity']}", f"{result['iou']}"])
        
        return result


    #Configures the optimizer(s) and optionally the learning rate scheduler for training.
    def configure_optimizers(self):
        if self.use_lr_scheduler:
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

if __name__ == "__main__":
    import segmentation_models_pytorch as smp

    smp_model = smp.Unet(
        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_name="mobilenet_v2",
        # use `imagenet` pre-trained weights for encoder initialization
        encoder_weights="imagenet",
        # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        in_channels=1,
        # model output channels (number of classes in your dataset)
        classes=2,
    )
    my_model = UNet(1, 2)

    dummy_input = torch.rand(4, 1, 512, 512)
    output = my_model(dummy_input)

    print(output.shape, torch.unique(output))
