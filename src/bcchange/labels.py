"""Stage 3 - turn catalogue polygons into per-year label rasters.

For each year t, a pixel is labelled
    harvest  if a Consolidated Cutblock polygon with HARVEST_YEAR == t covers it,
    fire     if a fire perimeter with FIRE_YEAR == t covers it (fire wins ties),
    no_change otherwise.
A forest mask from VRI (BCLCS level 2 == 'T' for treed) restricts training and
scoring to the forested land base so ocean, alpine and cities are not counted
as "no change" successes.

These functions are pure: they take GeoDataFrames and a raster grid and return
numpy arrays, which is what makes them unit-testable without the network.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from rasterio.transform import Affine

from .config import Config


def grid_from_raster(path: Path) -> tuple[Affine, tuple[int, int], str]:
    with rasterio.open(path) as src:
        return src.transform, (src.height, src.width), src.crs.to_string()


def rasterize_year(polys: gpd.GeoDataFrame, year_col: str, year: int, value: int,
                   transform: Affine, shape: tuple[int, int], min_ha: float = 0.0) -> np.ndarray:
    """Burn polygons whose year_col == year into a uint8 array with `value`."""
    sel = polys[polys[year_col] == year]
    if min_ha > 0:
        sel = sel[sel.geometry.area >= min_ha * 10_000]
    out = np.zeros(shape, dtype=np.uint8)
    if sel.empty:
        return out
    return features.rasterize(((g, value) for g in sel.geometry), out_shape=shape,
                              transform=transform, fill=0, dtype="uint8", all_touched=False)


def forest_mask(vri: gpd.GeoDataFrame, transform: Affine, shape: tuple[int, int]) -> np.ndarray:
    """1 where VRI says treed (BCLCS_LEVEL_2 == 'T'), else 0."""
    treed = vri[vri["BCLCS_LEVEL_2"] == "T"]
    if treed.empty:
        return np.zeros(shape, dtype=np.uint8)
    return features.rasterize(((g, 1) for g in treed.geometry), out_shape=shape,
                              transform=transform, fill=0, dtype="uint8")


def build_label(cfg: Config, year: int, cutblocks: gpd.GeoDataFrame | None,
                fires: gpd.GeoDataFrame | None, transform: Affine,
                shape: tuple[int, int]) -> np.ndarray:
    cls = cfg.classes
    min_ha = cfg["labels"]["min_polygon_ha"]
    lab = np.zeros(shape, dtype=np.uint8)
    if cutblocks is not None and not cutblocks.empty:
        h = rasterize_year(cutblocks, "HARVEST_YEAR", year, cls["harvest"], transform, shape, min_ha)
        lab = np.where(h > 0, h, lab)
    if fires is not None and not fires.empty:
        f = rasterize_year(fires, "FIRE_YEAR", year, cls["fire"], transform, shape, min_ha)
        lab = np.where(f > 0, f, lab)  # fire overrides harvest
    return lab


def write_raster(arr: np.ndarray, path: Path, transform: Affine, crs: str) -> Path:
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                       count=1, dtype=arr.dtype, crs=crs, transform=transform,
                       compress="deflate", tiled=True) as dst:
        dst.write(arr, 1)
    return path


def build_all(cfg: Config) -> dict[int, Path]:
    raw = cfg.path("raw")
    cut = _read_opt(raw / "cutblocks.parquet")
    fire = _read_opt(raw / "fires.parquet")
    vri = _read_opt(raw / "vri.parquet")
    outs = {}
    ref = cfg.path("interim") / f"s2_{cfg.years[0]}.tif"
    transform, shape, crs = grid_from_raster(ref)
    if vri is not None:
        write_raster(forest_mask(vri, transform, shape), cfg.path("interim") / "forest_mask.tif", transform, crs)
    for y in cfg.years:
        lab = build_label(cfg, y, cut, fire, transform, shape)
        outs[y] = write_raster(lab, cfg.path("interim") / f"label_{y}.tif", transform, crs)
    return outs


def _read_opt(p: Path):
    return gpd.read_parquet(p) if p.exists() else None
