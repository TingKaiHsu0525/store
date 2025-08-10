# test_lightning.py
import os
import csv
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms

from loader import WaferMapDataset        # your dataset
from models.swin_transformer_v2 import SwinTransformerV2
from test3 import LitWaferClassifier, build_swin_v2  # reuse your train code


def build_transforms(img_size=384, in_chans=3):
    """Build PIL->Tensor transforms for test/eval."""
    mean = [0.485] * in_chans
    std  = [0.229] * in_chans
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return tf


def make_test_loader(test_root, img_size=384, in_chans=3, batch_size=32, num_workers=4):
    """Create a test DataLoader."""
    tf = build_transforms(img_size=img_size, in_chans=in_chans)
    ds = WaferMapDataset(test_root, transform=tf, in_chans=in_chans)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return ds, loader


def run_test(
    ckpt_path,
    test_root,
    img_size=384,
    in_chans=3,
    num_classes=7,
    batch_size=32,
    num_workers=4,
    output_csv="test_predictions.csv",
):
    """Load checkpoint, run inference on test set, save CSV with filename, label_idx, pred_idx, prob."""
    # 1) Build backbone to match training config
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
        pretrained_ckpt=None,  # do not load pretrain here; checkpoint has all weights
    )

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) Wrap with LightningModule and load weights
    lit_model = LitWaferClassifier(backbone)
    # lit_model = lit_model.load_from_checkpoint(ckpt_path, model=backbone, strict=True)
    lit_model = LitWaferClassifier.load_from_checkpoint(
        ckpt_path,
        model=backbone,          # we ignored 'model' when saving hparams, so we must pass it here
        strict=True,
        map_location="cpu"       # or your device; remove if you want it to auto-map
    )

    # 3) Data
    ds, loader = make_test_loader(
        test_root=test_root,
        img_size=img_size,
        in_chans=in_chans,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # 4) Lightning Trainer for test (optional)
    # trainer = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1, precision="16-mixed" if torch.cuda.is_available() else 32)

    # If you implemented test_step/test_dataloader in your DataModule, you could use:
    # trainer.test(lit_model, datamodule=data_module, ckpt_path=ckpt_path)
    # Here we do manual inference for CSV export.

    lit_model.to(device)
    lit_model.eval()
    lit_model.freeze()

    results = []
    softmax = torch.nn.Softmax(dim=1)

    # 5) Run inference loop
    idx = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = lit_model(images)
            probs = softmax(logits)
            max_probs, preds = torch.max(probs, dim=1)

            bsz = images.size(0)
            correct += (preds == labels).sum().item()
            total += bsz

            for i in range(bsz):
                fname = os.path.basename(ds.samples[idx][0]) if hasattr(ds, "samples") else f"sample_{idx:06d}"
                label_idx = int(labels[i].item())
                pred_idx = int(preds[i].item())
                prob = float(max_probs[i].item())
                results.append([fname, label_idx, pred_idx, f"{prob:.6f}"])
                idx += 1

    # 計算 accuracy
    accuracy = correct / total if total > 0 else 0.0
    print(f"[Info] Test Accuracy: {accuracy:.4f}")
    

    # 6) Save CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label_idx", "pred_idx", "prob"])
        writer.writerows(results)

    print(f"[Info] Wrote predictions to {output_csv}")
    return output_csv


if __name__ == "__main__":
    # Example usage
    ckpt_path = "lightning_logs/swinv2/version_7/checkpoints/swinv2-epoch=23-val_acc=0.9839.ckpt"  # change to your best ckpt
    test_root = "wafermap/test"  # folder with class subfolders
    run_test(
        ckpt_path=ckpt_path,
        test_root=test_root,
        img_size=384,
        in_chans=3,          # set 6 if you trained with 6 channels
        num_classes=7,
        batch_size=32,
        num_workers=4,
        output_csv="test_pred.csv",
    )
