import argparse
import os
from typing import Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torchvision  # For CocoDetection base class
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from transformers import DetrImageProcessor

# Import your custom Detr model from elements.models
# Ensure the Detr class in elements/models.py matches the definition below.
from elements.models import Detr


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(
            self,
            image_directory_path: str,
            annotation_file_path: str,
            image_processor,
    ):
        super().__init__(image_directory_path, annotation_file_path)
        self.image_processor = image_processor
        # Pre-compute which indices have annotations
        self.valid_indices = [
            idx for idx in range(len(self.ids))
            if self._has_annotations(idx)
        ]

        # Create a mapping from base filename to image ID
        self.filename_to_id = {}
        for img_id, img_info in self.coco.imgs.items():
            base_filename = img_info['file_name']
            self.filename_to_id[base_filename] = img_id

    def _has_annotations(self, idx):
        """Check if image has any valid annotations"""
        _, annotations = super().__getitem__(idx)
        return bool(self._clean_annotations(annotations))

    def __len__(self):
        return len(self.valid_indices)

    def _fix_image_path(self, image_path):
        """Fix the image path if CVAT's export has broken paths."""
        # Replace only the initial segment (train, val, test) to avoid double replacements
        if image_path.startswith('/train/train/'):
            image_path = image_path.replace('/train/train/', '/train/')
        elif image_path.startswith('/val/val/'):
            image_path = image_path.replace('/val/val/', '/val/')
        elif image_path.startswith('/test/test/'):
            image_path = image_path.replace('/test/test/', '/test/')

        return image_path

    def _clean_annotations(self, annotations):
        cleaned = []
        for ann in annotations:
            # Skip if no bbox or category_id
            if 'bbox' not in ann or 'category_id' not in ann:
                continue

            # Convert bbox to [xmin, ymin, width, height] if needed
            bbox = ann['bbox']
            if len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0:
                cleaned.append({
                    'id': ann.get('id', 0),
                    'image_id': ann.get('image_id', 0),
                    'category_id': ann['category_id'],
                    'bbox': bbox,
                    'area': ann.get('area', bbox[2] * bbox[3]),
                    'iscrowd': ann.get('iscrowd', 0)
                })
        return cleaned

    def show_sample(self, idx):
        sample = self[idx]
        if sample is None:
            print("No valid annotations in this sample")
            return

        img = sample[0].permute(1, 2, 0).numpy()
        plt.imshow(img)
        plt.title(f"{len(sample[1]['class_labels'])} annotations")
        plt.show()

    def __getitem__(self, idx):
        # Map to original dataset index
        original_idx = self.valid_indices[idx]
        try:
            # Get image and annotations using original index
            image, annotations = super().__getitem__(original_idx)
            cleaned_annotations = self._clean_annotations(annotations)

            # This should never be None since we pre-filtered
            if not cleaned_annotations:
                raise ValueError("Unexpected empty annotations")

            image_id = self.ids[original_idx]
            coco_annotations = {
                "image_id": image_id,
                "annotations": cleaned_annotations
            }

            encoding = self.image_processor(
                images=image,
                annotations=coco_annotations,
                return_tensors="pt"
            )

            return encoding["pixel_values"].squeeze(), encoding["labels"][0]

        except Exception as e:
            print(f"Error processing image {original_idx}: {str(e)}")
            return None


def collate_fn(batch):
    """
    Collate function for DETR training.

    Each dataset item returns:
        pixel_values, labels

    This function pads the images and creates a pixel_mask.
    """
    batch = [item for item in batch if item is not None]

    if not batch:
        return {
            "pixel_values": torch.empty((0, 3, 800, 800)),
            "pixel_mask": torch.empty((0, 800, 800), dtype=torch.bool),
            "labels": [],
        }

    pixel_values = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    max_h = max(p.shape[1] for p in pixel_values)
    max_w = max(p.shape[2] for p in pixel_values)

    padded_pixel_values = []
    padded_pixel_mask = []

    for pv in pixel_values:
        h, w = pv.shape[1], pv.shape[2]
        pad_h = max_h - h
        pad_w = max_w - w

        padded_pv = torch.nn.functional.pad(
            pv,
            (0, pad_w, 0, pad_h),
            value=0,
        )

        # DETR pixel_mask usually uses 1 for real pixels and 0 for padding.
        pixel_mask = torch.ones((h, w), dtype=torch.bool)
        padded_pm = torch.nn.functional.pad(
            pixel_mask,
            (0, pad_w, 0, pad_h),
            value=0,
        )

        padded_pixel_values.append(padded_pv)
        padded_pixel_mask.append(padded_pm)

    return {
        "pixel_values": torch.stack(padded_pixel_values),
        "pixel_mask": torch.stack(padded_pixel_mask),
        "labels": labels,
    }

# The Detr LightningModule is assumed to be defined in elements/models.py
# If not, you need to ensure it's either in this file or correctly imported.
# For consistency with previous interactions, it's expected to be in elements/models.py.

def main():
    parser = argparse.ArgumentParser(description='Train DETR model for object detection')

    # Required arguments
    parser.add_argument('--train_dir', required=True,
                        help='Base directory for training images (e.g., /path/to/dataset)')
    parser.add_argument('--val_dir', required=True, help='Base directory for validation images')
    parser.add_argument('--test_dir', required=True, help='Base directory for test images')
    parser.add_argument('--train_annotations', required=True, help='Path to training annotations JSON')
    parser.add_argument('--val_annotations', required=True, help='Path to validation annotations JSON')
    parser.add_argument('--test_annotations', required=True, help='Path to testing annotations JSON')

    # Optional arguments with defaults
    parser.add_argument('--model_name', default='facebook/detr-resnet-50',
                        help='Pre-trained model name from Hugging Face or path to checkpoint')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--log_dir', default='./logs',
                        help='Directory to save TensorBoard logs and model checkpoints')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Training batch size')
    parser.add_argument('--val_batch_size', type=int, default=4,
                        help='Validation batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--lr_backbone', type=float, default=1e-5,
                        help='Backbone learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training (e.g., "cuda:0", "cpu")')
    parser.add_argument('--num_workers', type=int, default=os.cpu_count() // 2,  # Sensible default
                        help='Number of worker processes for data loading')
    # parser.add_argument('--num_workers', type=int, default=0,  # Sensible default
    #                     help='Number of worker processes for data loading')
    parser.add_argument('--precision', default="32-true",
                        help='Precision for training (e.g., "16-mixed", "32-true")')
    parser.add_argument('--gradient_clip_val', type=float, default=0.1,
                        help='Gradient clipping value')
    parser.add_argument('--accumulate_grad_batches', type=int, default=8,
                        help='Number of batches to accumulate gradients over')
    parser.add_argument('--save_top_k', type=int, default=1,
                        help='Save top k models based on validation loss')
    parser.add_argument('--monitor_metric', default="val_loss",
                        help='Metric to monitor for model checkpointing and LR scheduling')

    args = parser.parse_args()

    # Initialize processor
    image_processor = DetrImageProcessor.from_pretrained(args.model_name)

    # Create datasets
    print("Initializing training dataset...")
    # Pass only the *base* directory for images. CocoDetection will combine this with file_name from annotations.
    train_dataset = CocoDetection(
        image_directory_path=args.train_dir,
        annotation_file_path=args.train_annotations,
        image_processor=image_processor
    )
    print("Initializing validation dataset...")
    val_dataset = CocoDetection(
        image_directory_path=args.val_dir,
        annotation_file_path=args.val_annotations,
        image_processor=image_processor
    )
    print("Initializing test dataset (not used in training loop, but useful for evaluation after training)...")
    test_dataset = CocoDetection(
        image_directory_path=args.test_dir,
        annotation_file_path=args.test_annotations,
        image_processor=image_processor
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        dataset=train_dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        collate_fn=collate_fn,
        batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    test_dataloader = DataLoader(  # For post-training evaluation
        dataset=test_dataset,
        collate_fn=collate_fn,
        batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        pin_memory=False
    )

    # Print dataset and dataloader information
    print("\n--- Dataset and Dataloader Information ---")
    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")
    print(f"Test examples: {len(test_dataset)}")
    print(f"Train batches per epoch: {len(train_dataloader)}")
    print(f"Validation batches per epoch: {len(val_dataloader)}")
    print(f"Test batches per epoch: {len(test_dataloader)}\n")

    # Initialize model
    print(f"Initializing DETR LightningModule with model: {args.model_name}")
    model = Detr(lr=args.lr, lr_backbone=args.lr_backbone, weight_decay=args.weight_decay)

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.log_dir, "detr_experiment", "checkpoints"),
        filename="{epoch}-{val_loss:.2f}",
        save_top_k=args.save_top_k,
        monitor=args.monitor_metric,
        mode="min",
        verbose=True
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Setup logger and trainer
    logger = TensorBoardLogger(args.log_dir, name="detr_experiment")

    print("\nSetting up PyTorch Lightning Trainer...")
    trainer = pl.Trainer(
        devices=1,  # Use a single device as specified by the user's initial code
        accelerator=args.device.split(':')[0],  # "cuda" or "cpu"
        max_epochs=args.epochs,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=5,  # Log metrics every 5 steps
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        precision=args.precision,
        # detect_anomaly=True # Uncomment for debugging numerical instabilities
    )
    print("Trainer setup complete.")

    # Start training
    print("\nStarting training...")
    trainer.fit(model, train_dataloader, val_dataloader)
    print("Training complete!")

    # Optional: Test the model after training
    if len(test_dataloader) > 0 and input("Run evaluation on test set after training? (y/n): ").lower() == 'y':
        print("\nRunning evaluation on test set...")
        trainer.test(model, test_dataloader)
        print("Test evaluation complete.")


if __name__ == '__main__':
    main()
