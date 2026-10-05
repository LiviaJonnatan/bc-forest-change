"""Stage 4 - cut (year t-1, year t) image pairs and year-t labels into chips.

Each chip stores: image_prev (C,H,W), image_curr (C,H,W), label (H,W), the
forest fraction, and a spatial block id. The block id is the FWA assessment
watershed containing the chip centre (falling back to a coarse grid when
watersheds are unavailable). Train/val/test splits are assigned at the block
level so that neighbouring, near-identical chips never straddle a split -
random chip splits in remote sensing leak badly and inflate accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from shapely.geometry import Point

from .config import Config


@dataclass
class ChipIndex:
    rows: list[dict]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def iter_windows(height: int, width: int, size: int, stride: int):
    for r in range(0, height - size + 1, stride):
        for c in range(0, width - size + 1, stride):
            yield r, c, Window(c, r, size, size)


def assign_blocks(centres: gpd.GeoSeries, watersheds: gpd.GeoDataFrame | None,
                  grid_m: float = 10_000) -> np.ndarray:
    """Block id per chip: watershed id if available, else 10 km grid cell."""
    if watersheds is not None and not watersheds.empty:
        joined = gpd.sjoin(gpd.GeoDataFrame(geometry=centres), watersheds[["geometry"]],
                           how="left", predicate="within")
        joined = joined[~joined.index.duplicated()]
        ids = joined["index_right"].fillna(-1).astype(int).values
        if (ids >= 0).mean() > 0.9:
            return ids
    xs, ys = centres.x.values, centres.y.values
    return (np.floor(xs / grid_m) * 100_000 + np.floor(ys / grid_m)).astype(int)


def split_blocks(block_ids: np.ndarray, fractions: dict[str, float], seed: int = 0) -> np.ndarray:
    """Assign every chip a split by shuffling *blocks*, not chips."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(block_ids)
    rng.shuffle(uniq)
    n = len(uniq)
    n_train = round(fractions["train"] * n)
    n_val = round(fractions["val"] * n)
    lookup = {}
    for i, b in enumerate(uniq):
        lookup[b] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    return np.array([lookup[b] for b in block_ids])


def make_chips(cfg: Config, year: int, image_prev: Path, image_curr: Path, label: Path,
               forest_mask: Path | None, watersheds: gpd.GeoDataFrame | None,
               out_dir: Path) -> ChipIndex:
    t = cfg["tiles"]
    size, stride, min_forest = t["size_px"], t["stride_px"], t["min_forest_frac"]
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, centres = [], []
    with rasterio.open(image_prev) as a, rasterio.open(image_curr) as b, rasterio.open(label) as lab_src:
        fm = rasterio.open(forest_mask) if forest_mask else None
        for r, c, win in iter_windows(lab_src.height, lab_src.width, size, stride):
            ff = fm.read(1, window=win).mean() if fm else 1.0
            if ff < min_forest:
                continue
            lab = lab_src.read(1, window=win)
            xa, xb = a.read(window=win), b.read(window=win)
            if np.isnan(xa).mean() > 0.2 or np.isnan(xb).mean() > 0.2:
                continue  # too much cloud gap
            path = out_dir / f"chip_{year}_{r}_{c}.npz"
            np.savez_compressed(path, prev=xa.astype(np.float32), curr=xb.astype(np.float32),
                                label=lab.astype(np.uint8))
            cx, cy = lab_src.xy(r + size // 2, c + size // 2)
            centres.append(Point(cx, cy))
            rows.append({"path": str(path), "year": year, "row": r, "col": c,
                         "forest_frac": float(ff),
                         "change_frac": float((lab > 0).mean()),
                         "harvest_frac": float((lab == cfg.classes["harvest"]).mean()),
                         "fire_frac": float((lab == cfg.classes["fire"]).mean())})
        if fm:
            fm.close()
    if not rows:
        return ChipIndex([])
    centres = gpd.GeoSeries(centres, crs=cfg.crs)
    blocks = assign_blocks(centres, watersheds)
    splits = split_blocks(blocks, t["split"])
    for row, bl, sp in zip(rows, blocks, splits, strict=True):
        row["block"] = int(bl)
        row["split"] = sp
    return ChipIndex(rows)


def build_all(cfg: Config) -> Path:
    interim, proc = cfg.path("interim"), cfg.path("processed")
    ws_path = cfg.path("raw") / "watersheds.parquet"
    ws = gpd.read_parquet(ws_path) if ws_path.exists() else None
    fm = interim / "forest_mask.tif"
    frames = []
    for prev, curr in zip(cfg.years[:-1], cfg.years[1:], strict=True):
        idx = make_chips(cfg, curr, interim / f"s2_{prev}.tif", interim / f"s2_{curr}.tif",
                         interim / f"label_{curr}.tif", fm if fm.exists() else None, ws,
                         proc / "chips")
        frames.append(idx.to_frame())
    index = pd.concat(frames, ignore_index=True)
    out = proc / "chip_index.parquet"
    index.to_parquet(out)
    return out
