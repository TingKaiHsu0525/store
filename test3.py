# train_lightning.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedShuffleSplit
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks import EarlyStopping

from loader import WaferMapDataset
from models.swin_transformer_v2 import SwinTransformerV2


# --------------------------
# Utils: build Swin V2 model
# --------------------------
def build_swin_v2(
    img_size=384,
    patch_size=4,
    in_chans=3,
    num_classes=7,
    embed_dim=128,
    depths=[2, 2, 18, 2],
    num_heads=[4, 8, 16, 32],
    window_size=24,
    ape=False,
    drop_path_rate=0.2,
    patch_norm=True,
    pretrained_ckpt="swinv2_base_patch4_window12to24_192to384_22kto1k_ft.pth",
):
    """
    Build SwinTransformerV2 backbone, load pretrained weights, and replace the head for fine-tuning.
    """
    # 1) Create model (set num_classes to a temporary value; we'll replace head later)
    model = SwinTransformerV2(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_classes=1000,  # will be replaced
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        ape=ape,
        drop_path_rate=drop_path_rate,
        patch_norm=patch_norm,
        pretrained=False,
    )

    # 2) Load pretrained state dict
    if pretrained_ckpt is not None and os.path.isfile(pretrained_ckpt):
        checkpoint = torch.load(pretrained_ckpt, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)
        # Adjust first conv when in_chans != 3
        if in_chans != 3 and "patch_embed.proj.weight" in state_dict:
            # old shape: [embed_dim, 3, k, k] -> new: [embed_dim, in_chans, k, k]
            old_w = state_dict["patch_embed.proj.weight"]
            if old_w.shape[1] != in_chans:
                new_w = old_w.mean(dim=1, keepdim=True).repeat(1, in_chans, 1, 1)
                state_dict["patch_embed.proj.weight"] = new_w
        # Load with strict=False to tolerate minor key mismatches (e.g., head)
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"[Info] Loaded pretrained weights with message: {msg}")
    else:
        print("[Warn] Pretrained checkpoint not found or not provided; training from scratch.")

    # 3) Replace classifier head for current task
    # if hasattr(model, "head") and isinstance(model.head, nn.Linear):
    #     in_features = model.head.in_features
    #     model.head = nn.Linear(in_features, num_classes)
    # else:
    #     # Fallback: if implementation uses a different head name
    #     raise AttributeError("Model does not expose a linear 'head' attribute.")

    old_head = model.head
    model.head = nn.Sequential(
        old_head,
        nn.Linear(old_head.out_features, num_classes)
    )

    return model


# --------------------------
# Lightning Module
# --------------------------
class LitWaferClassifier(pl.LightningModule):
    """
    LightningModule that wraps the Swin V2 classifier for wafer map classification.
    """

    def __init__(self, model: nn.Module, lr=1e-4, weight_decay=1e-4, t_max=10, freeze_backbone=True):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.t_max = t_max
        self.criterion = nn.CrossEntropyLoss()

        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if not name.startswith("head"):
                    param.requires_grad = False

        # Save hyperparameters for checkpointing/logging
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x):
        # Forward pass
        return self.model(x)

    def _step(self, batch, stage: str):
        # Generic step to compute loss/acc and log
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        # Log metrics
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        # Training step
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        # Validation step
        self._step(batch, "val")

    def configure_optimizers(self):
        # Configure AdamW + CosineAnnealingLR
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.t_max)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss",
            },
        }
    def on_train_epoch_end(self):
        # Write epoch-indexed scalars so TensorBoard x-axis is epoch
        logger = self.logger
        if isinstance(logger, TensorBoardLogger):
            m = self.trainer.callback_metrics  # aggregated epoch metrics
            if "train_loss" in m:
                logger.experiment.add_scalar("loss_epoch/train", m["train_loss"].item(), self.current_epoch)
            if "train_acc" in m:
                logger.experiment.add_scalar("acc_epoch/train",  m["train_acc"].item(),  self.current_epoch)

    def on_validation_epoch_end(self):
        logger = self.logger
        if isinstance(logger, TensorBoardLogger):
            m = self.trainer.callback_metrics
            if "val_loss" in m:
                logger.experiment.add_scalar("loss_epoch/val", m["val_loss"].item(), self.current_epoch)
            if "val_acc" in m:
                logger.experiment.add_scalar("acc_epoch/val",  m["val_acc"].item(),  self.current_epoch)



# --------------------------
# Lightning DataModule
# --------------------------
class WaferDataModule(pl.LightningDataModule):
    """
    DataModule to handle train/val datasets and loaders for WaferMapDataset.
    """

    def __init__(
        self,
        train_root: str,
        val_root: str = None,
        in_chans: int = 3,
        img_size: int = 384,
        batch_size: int = 32,
        num_workers: int = 4,
        mean=None,
        std=None,
        val_split: float = 0.2,   # <<< add: fraction for validation
        seed: int = 42,           # <<< add: reproducibility
    ):
        super().__init__()
        self.train_root = train_root
        self.val_root = val_root
        self.in_chans = in_chans
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.seed = seed
        
        if mean is None or std is None:
            mean = [0.485] * in_chans
            std  = [0.229] * in_chans

        # Basic transforms; extend/augment as needed
        self.train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        self.val_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        self.train_dataset = None
        self.val_dataset = None
        self.num_classes = None

    def _extract_labels(self, dataset):
        """Return list/array of labels for stratification."""
        # Prefer built-in attributes if available
        if hasattr(dataset, "targets"):
            return np.array(dataset.targets)
        if hasattr(dataset, "labels"):
            return np.array(dataset.labels)

        # Fallback: iterate once (may be slower; done only at setup)
        labels = []
        for i in range(len(dataset)):
            # Only label is needed; image read may happen depending on dataset implementation
            _, y = dataset[i]
            labels.append(int(y))
        return np.array(labels)

    def setup(self, stage=None):
        # Case A: user provides a separate validation folder
        if self.val_root is not None and os.path.isdir(self.val_root):
            self.train_dataset = WaferMapDataset(self.train_root, transform=self.train_transform, in_chans=self.in_chans)
            self.val_dataset   = WaferMapDataset(self.val_root,   transform=self.val_transform,   in_chans=self.in_chans)
        else:
            # Case B: split from train_root using stratified split
            # Build two full datasets with different transforms
            full_train = WaferMapDataset(self.train_root, transform=self.train_transform, in_chans=self.in_chans)
            full_val   = WaferMapDataset(self.train_root, transform=self.val_transform,   in_chans=self.in_chans)

            # Extract labels for stratification
            y = self._extract_labels(full_train)
            n = len(y)
            assert 0 < self.val_split < 1, "val_split must be in (0, 1)"
            test_size = self.val_split

            # Stratified split
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=self.seed)
            train_idx, val_idx = next(sss.split(np.arange(n), y))

            # Create subsets that share indices but keep distinct transforms
            self.train_dataset = Subset(full_train, train_idx)
            self.val_dataset   = Subset(full_val,   val_idx)

        # Expose num_classes
        # If dataset is Subset, access the underlying dataset's mapping
        base_ds = self.train_dataset.dataset if isinstance(self.train_dataset, Subset) else self.train_dataset
        self.num_classes = len(base_ds.class_to_idx)
    
    def train_dataloader(self):
        # Return the training DataLoader
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,          # shuffle for training
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        # Always return a valid DataLoader if we created a split
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

# --------------------------
# Main: wiring everything
# --------------------------
def main():
    # ---- Configs ----
    in_chans = 3           # set to 6 if you have 6-channel input
    img_size = 384
    num_classes = 7
    batch_size = 8
    num_workers = 4
    lr = 3e-5
    weight_decay = 0.05
    t_max = 10
    max_epochs = 50
    ckpt_path = "swinv2_base_patch4_window12to24_192to384_22kto1k_ft.pth"

    # ---- Data ----
    # If you don't have a val set yet, set val_root=None; Lightning will skip val loop.
    data = WaferDataModule(
        train_root="wafermap/train",
        val_root=None,                 # e.g., "wafermap/val"
        in_chans=in_chans,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    data.setup()
    print(f"[Info] num_classes from dataset: {data.num_classes} (will use {num_classes})")

    # ---- Model ----
    backbone = build_swin_v2(
        img_size=img_size,
        patch_size=4,
        in_chans=in_chans,
        num_classes=num_classes,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=24,
        ape=False,
        drop_path_rate=0.2,
        patch_norm=True,
        pretrained_ckpt=ckpt_path,
    )
    lit_model = LitWaferClassifier(backbone, lr=lr, weight_decay=weight_decay, t_max=t_max)

    ckpt_cb = ModelCheckpoint(
        monitor="val_loss",        # or "val_loss"
        mode="min",               # use "min" if you monitor val_loss
        save_top_k=1,
        filename="swinv2-{epoch:02d}-{val_acc:.4f}"
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")

    early_stop_cb = EarlyStopping(
        monitor="val_loss",     # or "val_acc"
        mode="min",             # use "max" if you monitor val_acc
        patience=5,             # number of epochs with no improvement to wait
        min_delta=0.0,          # minimum change to qualify as an improvement
        verbose=True,
        check_finite=True,      # stop if NaN/inf appears
        # stopping_threshold=None,       # optional hard threshold to stop early
        # divergence_threshold=None,     # optional divergence stop
    )

    # Create TensorBoard logger
    tb_logger = TensorBoardLogger(save_dir="lightning_logs", name="swinv2")

    # ---- Trainer ----
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        log_every_n_steps=1,          # keep your logging cadence
        logger=tb_logger,              # << add logger
        callbacks=[ckpt_cb, lr_cb, early_stop_cb],    # << add callbacks
    )

    # ---- Fit ----
    # if data.val_dataloader() is None:
    #     trainer.fit(lit_model, train_dataloaders=data.train_dataloader())
    # else:
    #     trainer.fit(lit_model, train_dataloaders=data.train_dataloader(), val_dataloaders=data.val_dataloader())
    trainer.fit(lit_model, datamodule=data)
    

if __name__ == "__main__":
    main()
