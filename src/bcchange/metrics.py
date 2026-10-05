"""Confusion-matrix metrics (torch-free so they can be tested anywhere)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def confusion(pred: np.ndarray, true: np.ndarray, n_classes: int, mask: np.ndarray | None = None):
    if mask is not None:
        pred, true = pred[mask > 0], true[mask > 0]
    return np.bincount(n_classes * true.ravel() + pred.ravel(), minlength=n_classes ** 2).reshape(n_classes, n_classes)


def metrics_from_confusion(cm: np.ndarray, names: list[str]) -> pd.DataFrame:
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    iou = tp / np.maximum(tp + fp + fn, 1)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / np.maximum(tp + fn, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    return pd.DataFrame({"class": names, "iou": iou, "precision": prec, "recall": rec, "f1": f1,
                         "support_px": cm.sum(1)})


