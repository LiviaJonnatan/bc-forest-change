"""Stage 5 - change-detection model.

Design: early-fusion U-Net. The two years' bands plus their difference are
stacked on the channel axis and fed to a U-Net whose encoder is a pretrained
ResNet from segmentation_models_pytorch. Early fusion with an explicit
difference channel is a strong, simple baseline for bitemporal change
detection and trains in minutes on a free Colab GPU; Siamese encoders can
come later if the baseline plateaus.

Loss: cross-entropy with class weights (change pixels are rare) plus Dice on
the change classes. Metrics: per-class IoU and F1 on the forest mask only.
Torch is an optional dependency; this module is only imported by `train`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import confusion, metrics_from_confusion

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object


class ChipDataset(Dataset):
    """Loads chips from the index; returns (x, y) with x = [prev, curr, curr-prev]."""

    def __init__(self, index: pd.DataFrame, band_mean: np.ndarray, band_std: np.ndarray,
                 augment: bool = False):
        self.index = index.reset_index(drop=True)
        self.mean = band_mean[:, None, None].astype(np.float32)
        self.std = band_std[:, None, None].astype(np.float32)
        self.augment = augment

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        z = np.load(self.index.path[i])
        prev = np.nan_to_num((z["prev"] - self.mean) / self.std)
        curr = np.nan_to_num((z["curr"] - self.mean) / self.std)
        x = np.concatenate([prev, curr, curr - prev], axis=0)
        y = z["label"].astype(np.int64)
        if self.augment:
            if np.random.rand() < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
            if np.random.rand() < 0.5:
                x, y = x[:, ::-1, :], y[::-1, :]
            k = np.random.randint(4)
            x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))


def band_stats(index: pd.DataFrame, n_sample: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Per-band mean/std from a sample of training chips."""
    sample = index[index.split == "train"].sample(min(n_sample, (index.split == "train").sum()),
                                                   random_state=0)
    acc = []
    for p in sample.path:
        acc.append(np.load(p)["curr"].reshape(-1, 256 * 256))
    arr = np.concatenate(acc, axis=1)
    return np.nanmean(arr, axis=1), np.nanstd(arr, axis=1) + 1e-6


def build_model(in_channels: int, n_classes: int, encoder: str = "resnet18"):
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name=encoder, encoder_weights="imagenet",
                    in_channels=in_channels, classes=n_classes)


def dice_loss(logits, target, n_classes, skip_class=0, eps=1e-6):
    probs = torch.softmax(logits, dim=1)
    onehot = torch.nn.functional.one_hot(target, n_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (probs * onehot).sum(dims)
    denom = probs.sum(dims) + onehot.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)
    keep = [c for c in range(n_classes) if c != skip_class]
    return 1 - dice[keep].mean()


def train(index_path: Path, out_dir: Path, n_classes: int, class_names: list[str],
          epochs: int = 15, batch_size: int = 16, lr: float = 3e-4, encoder: str = "resnet18",
          device: str | None = None) -> pd.DataFrame:
    if torch is None:  # pragma: no cover
        raise ImportError("pip install 'bcchange[model]'")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    index = pd.read_parquet(index_path)
    mean, std = band_stats(index)
    np.savez(out_dir / "band_stats.npz", mean=mean, std=std)
    dl = {s: DataLoader(ChipDataset(index[index.split == s], mean, std, augment=(s == "train")),
                        batch_size=batch_size, shuffle=(s == "train"), num_workers=2)
          for s in ["train", "val", "test"]}
    in_ch = 3 * len(mean)
    model = build_model(in_ch, n_classes, encoder).to(device)
    # Class weights from pixel frequency in the training split.
    freq = np.array([index.loc[index.split == "train", c].mean()
                     for c in ["change_frac"]]).sum()
    w = torch.tensor([1.0] + [max(1.0, (1 - freq) / max(freq, 1e-3))] * (n_classes - 1),
                     dtype=torch.float32, device=device)
    ce = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(dl["train"]))
    best, log = -1, []
    for ep in range(epochs):
        model.train()
        for x, y in dl["train"]:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = ce(logits, y) + dice_loss(logits, y, n_classes)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        val = evaluate(model, dl["val"], n_classes, class_names, device)
        score = val.loc[val["class"] != class_names[0], "iou"].mean()
        log.append({"epoch": ep, "val_change_iou": score})
        print(f"epoch {ep:02d}  val mean change IoU {score:.3f}")
        if score > best:
            best = score
            torch.save(model.state_dict(), out_dir / "model.pt")
    model.load_state_dict(torch.load(out_dir / "model.pt", map_location=device))
    test = evaluate(model, dl["test"], n_classes, class_names, device)
    test.to_csv(out_dir / "test_metrics.csv", index=False)
    pd.DataFrame(log).to_csv(out_dir / "train_log.csv", index=False)
    return test


@torch.no_grad() if torch else (lambda f: f)
def evaluate(model, loader, n_classes, class_names, device) -> pd.DataFrame:
    model.eval()
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu().numpy()
        cm += confusion(pred, y.numpy(), n_classes)
    return metrics_from_confusion(cm, class_names)
