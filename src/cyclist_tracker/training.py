"""Fine-tuning DETR on annotated cyclist frames (PyTorch Lightning).

All boxes in the COCO files are treated as one class, "cyclist". The best
checkpoint (lowest validation loss) is written to
``<output_dir>/checkpoints/`` and TensorBoard logs to ``<output_dir>/logs/``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import lightning.pytorch as pl
import torch
from PIL import Image
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from .dataset import find_annotation_file, resolve_image
from .detection import BASE_MODEL

log = logging.getLogger(__name__)


class Detr(pl.LightningModule):
    """DETR with a single "cyclist" class, trained with AdamW and a plateau scheduler."""

    def __init__(self, lr: float = 1e-4, lr_backbone: float = 1e-5, weight_decay: float = 1e-4, num_labels: int = 1):
        super().__init__()
        self.save_hyperparameters()
        from transformers import DetrForObjectDetection

        self.model = DetrForObjectDetection.from_pretrained(
            BASE_MODEL,
            num_labels=num_labels,
            id2label={0: "cyclist"},
            label2id={"cyclist": 0},
            ignore_mismatched_sizes=True,
        )
        self.model.train()  # from_pretrained() returns eval mode, which would disable dropout while training

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def _step(self, batch, prefix: str):
        if batch["pixel_values"].shape[0] == 0:
            return None
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]
        out = self.model(pixel_values=batch["pixel_values"], pixel_mask=batch["pixel_mask"], labels=labels)
        logs = {f"{prefix}_loss": out.loss} | {f"{prefix}_{k}": v for k, v in out.loss_dict.items()
                                                if k in ("loss_ce", "loss_bbox", "loss_giou")}
        self.log_dict(logs, prog_bar=prefix == "train", batch_size=len(labels))
        return out.loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        groups = [
            {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad],
             "lr": self.hparams.lr},
            {"params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
             "lr": self.hparams.lr_backbone},
        ]
        optimizer = torch.optim.AdamW(groups, weight_decay=self.hparams.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}


class CocoCyclists(Dataset):
    """Images with at least one valid box from a COCO file; every box becomes class 0."""

    def __init__(self, root: str | Path, split: str, annotation_file: str | Path, processor):
        self.root, self.split, self.processor = Path(root), split, processor
        data = json.loads(Path(annotation_file).read_text())
        boxes: dict[int, list[dict]] = {}
        for ann in data.get("annotations", []):
            bbox = ann.get("bbox", [])
            if len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0:
                boxes.setdefault(ann["image_id"], []).append(
                    {"bbox": bbox, "category_id": 0, "area": bbox[2] * bbox[3], "iscrowd": 0})
        self.items = [(img["id"], img["file_name"], boxes[img["id"]])
                      for img in data.get("images", []) if img["id"] in boxes]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx):
        image_id, file_name, anns = self.items[idx]
        image = Image.open(resolve_image(self.root, self.split, file_name)).convert("RGB")
        encoding = self.processor(images=image, annotations={"image_id": image_id, "annotations": anns},
                                  return_tensors="pt")
        return encoding["pixel_values"].squeeze(0), encoding["labels"][0]


def collate_fn(batch):
    """Pad images to the largest size in the batch and build the matching pixel mask."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return {"pixel_values": torch.empty(0, 3, 800, 800), "pixel_mask": torch.empty(0, 800, 800, dtype=torch.long),
                "labels": []}
    max_h = max(pv.shape[1] for pv, _ in batch)
    max_w = max(pv.shape[2] for pv, _ in batch)
    values, masks = [], []
    for pv, _ in batch:
        h, w = pv.shape[1:]
        values.append(torch.nn.functional.pad(pv, (0, max_w - w, 0, max_h - h)))
        mask = torch.zeros(max_h, max_w, dtype=torch.long)
        mask[:h, :w] = 1
        masks.append(mask)
    return {"pixel_values": torch.stack(values), "pixel_mask": torch.stack(masks), "labels": [lb for _, lb in batch]}


@dataclass
class TrainConfig:
    dataset_dir: str
    output_dir: str = "runs/detr"
    epochs: int = 30
    batch_size: int = 2
    val_batch_size: int = 4
    lr: float = 1e-4
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-4
    accumulate_grad_batches: int = 8
    gradient_clip_val: float = 0.1
    precision: str = "32-true"
    num_workers: int = 2
    use_gpu: bool = True
    limit_batches: float | int = 1.0  # for quick trial runs


class _Reporter(pl.Callback):
    def __init__(self, progress: Callable[[int, int, dict], None] | None, should_stop: Callable[[], bool] | None):
        self.progress, self.should_stop = progress, should_stop

    def on_train_batch_end(self, trainer, *args):
        if self.should_stop and self.should_stop():
            trainer.should_stop = True

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        metrics = {k: float(v) for k, v in trainer.callback_metrics.items() if k in ("train_loss", "val_loss")}
        log.info("Epoch %d/%d: %s", trainer.current_epoch + 1, trainer.max_epochs,
                 ", ".join(f"{k} {v:.4f}" for k, v in metrics.items()))
        if self.progress:
            self.progress(trainer.current_epoch + 1, trainer.max_epochs, metrics)


def train_detector(config: TrainConfig, progress: Callable[[int, int, dict], None] | None = None,
                   should_stop: Callable[[], bool] | None = None) -> Path | None:
    """Fine-tune DETR; returns the path of the best checkpoint (``None`` if nothing was saved)."""
    from transformers import DetrImageProcessor

    root = Path(config.dataset_dir)
    files = {split: find_annotation_file(root, split) for split in ("train", "val")}
    missing = [s for s, f in files.items() if f is None]
    if missing:
        raise FileNotFoundError(f"No annotation file for: {', '.join(missing)} (expected annotations/<split>.json)")
    processor = DetrImageProcessor.from_pretrained(BASE_MODEL)
    train_set = CocoCyclists(root, "train", files["train"], processor)
    val_set = CocoCyclists(root, "val", files["val"], processor)
    if not len(train_set) or not len(val_set):
        raise ValueError("The training and validation sets both need images with at least one box.")
    log.info("Training on %d images, validating on %d images", len(train_set), len(val_set))

    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_config.json").write_text(json.dumps(asdict(config), indent=2))
    loaders = dict(collate_fn=collate_fn, num_workers=config.num_workers,
                   persistent_workers=config.num_workers > 0)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, drop_last=len(train_set) >
                              config.batch_size, **loaders)
    val_loader = DataLoader(val_set, batch_size=config.val_batch_size, **loaders)

    model = Detr(lr=config.lr, lr_backbone=config.lr_backbone, weight_decay=config.weight_decay)
    checkpoint = pl.callbacks.ModelCheckpoint(dirpath=out / "checkpoints", filename="detr-{epoch:02d}-{val_loss:.3f}",
                                              monitor="val_loss", mode="min", save_top_k=1)
    gpu = config.use_gpu and torch.cuda.is_available()
    trainer = pl.Trainer(
        accelerator="gpu" if gpu else "cpu",
        devices=1,
        max_epochs=config.epochs,
        precision=config.precision,
        gradient_clip_val=config.gradient_clip_val,
        accumulate_grad_batches=config.accumulate_grad_batches,
        logger=pl.loggers.TensorBoardLogger(out, name="logs"),
        callbacks=[checkpoint, pl.callbacks.LearningRateMonitor("epoch"), _Reporter(progress, should_stop)],
        log_every_n_steps=5,
        limit_train_batches=config.limit_batches,
        limit_val_batches=config.limit_batches,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
    )
    trainer.fit(model, train_loader, val_loader)
    best = checkpoint.best_model_path
    log.info("Best checkpoint: %s", best or "(none)")
    return Path(best) if best else None
