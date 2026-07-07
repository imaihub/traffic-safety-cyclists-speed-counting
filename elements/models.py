import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import DetrForObjectDetection

class Detr(pl.LightningModule):
    """
    A PyTorch Lightning module for fine-tuning a DETR object detection model.
    It encapsulates the model, forward pass, training, validation, and optimizer setup.
    """
    def __init__(self, lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4):
        """
        Initializes the Detr Lightning module.

        Args:
            lr (float): Learning rate for the main model parameters.
            lr_backbone (float): Learning rate for the backbone parameters.
            weight_decay (float): Weight decay for the optimizer.
        """
        super().__init__()
        # Save hyperparameters for easy access and logging
        self.save_hyperparameters()

        # Load a pre-trained DETR model from Hugging Face Transformers.
        # num_labels is set to 2 (1 for 'cyclist', 1 for 'no_object').
        # id2label and label2id maps are crucial for correct interpretation of model outputs.
        # ignore_mismatched_sizes=True allows loading a pre-trained model
        # with a different head (e.g., originally trained on COCO with many classes).
        self.model = DetrForObjectDetection.from_pretrained(
            pretrained_model_name_or_path="facebook/detr-resnet-50",
            num_labels=1,
            id2label={0: "cyclist"},
            label2id={"cyclist": 0},
            ignore_mismatched_sizes=True
        )

    def forward(self, pixel_values, pixel_mask):
        """
        Performs a forward pass through the DETR model.

        Args:
            pixel_values (torch.Tensor): Batched image pixel values.
            pixel_mask (torch.Tensor): Batched pixel masks (indicating padding).

        Returns:
            transformers.modeling_outputs.DetrForObjectDetectionOutput: Model outputs.
        """
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def common_step(self, batch, batch_idx):
        """
        A common step function for both training and validation.
        Computes loss and loss dictionary.

        Args:
            batch (dict): A dictionary containing 'pixel_values', 'pixel_mask', and 'labels'.
            batch_idx (int): Index of the current batch.

        Returns:
            tuple: (loss, loss_dict) if batch is not empty, otherwise (None, None).
        """
        # Skip empty batches to prevent errors.
        if batch["pixel_values"].shape[0] == 0:
            return None, None

        # Move labels to the appropriate device (CPU/GPU)
        labels = [{k: v.to(self.device) for k, v in t.items()}
                 for t in batch["labels"]]

        # Pass inputs and labels to the model to compute losses.
        outputs = self.model(
            pixel_values=batch["pixel_values"],
            pixel_mask=batch["pixel_mask"],
            labels=labels
        )
        return outputs.loss, outputs.loss_dict

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step.

        Args:
            batch (dict): A dictionary containing training data.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: The computed training loss.
        """
        loss, loss_dict = self.common_step(batch, batch_idx)
        if loss is None:
            return None

        # Log training losses for monitoring. `prog_bar=True` shows them in the progress bar.
        # `sync_dist=True` is important for distributed training.
        self.log_dict(
            {
                "train_loss": loss,
                "train_loss_ce": loss_dict["loss_ce"],
                "train_loss_bbox": loss_dict["loss_bbox"],
                "train_loss_giou": loss_dict["loss_giou"]
            },
            prog_bar=True,
            sync_dist=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Performs a single validation step.

        Args:
            batch (dict): A dictionary containing validation data.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: The computed validation loss.
        """
        loss, loss_dict = self.common_step(batch, batch_idx)
        if loss is None:
            return None

        # Log validation losses.
        self.log_dict(
            {
                "val_loss": loss,
                "val_loss_ce": loss_dict["loss_ce"],
                "val_loss_bbox": loss_dict["loss_bbox"],
                "val_loss_giou": loss_dict["loss_giou"]
            },
            sync_dist=True
        )
        return loss

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.
        Uses AdamW with different learning rates for the backbone and other parameters.
        Includes a ReduceLROnPlateau scheduler.

        Returns:
            tuple: A list containing the optimizer and a dictionary for the scheduler.
        """
        # Define parameter groups for different learning rates.
        # Backbone parameters typically use a smaller learning rate.
        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters()
                          if "backbone" not in n and p.requires_grad],
                "lr": self.hparams.lr,
            },
            {
                "params": [p for n, p in self.named_parameters()
                          if "backbone" in n and p.requires_grad],
                "lr": self.hparams.lr_backbone,
            },
        ]

        optimizer = torch.optim.AdamW(param_dicts, weight_decay=self.hparams.weight_decay)

        # Configure learning rate scheduler to reduce LR when validation loss plateaus.
        scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.1,
                patience=3
            ),
            "monitor": "val_loss", # Metric to monitor for LR reduction
        }

        return [optimizer], [scheduler]

    # Placeholder for dataloaders. In a real scenario, these would be passed or initialized.
    # For this refactoring, we assume they are globally available or handled externally.
    # def train_dataloader(self):
    #     return TRAIN_DATALOADER

    # def val_dataloader(self):
    #     return VAL_DATALOADER

