"""Shared fixtures: a tiny synthetic study area so tests run offline."""
import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from bcchange.config import Config

CRS = "EPSG:3005"
RES = 10
H = W = 512  # 5.12 km square


@pytest.fixture
def cfg(tmp_path):
    import yaml
    with open("config/southcoast.yaml") as f:
        conf = yaml.safe_load(f)
    conf["years"] = [2020, 2021]
    conf["tiles"]["size_px"] = 128
    conf["tiles"]["stride_px"] = 128
    for k in conf["paths"]:
        conf["paths"][k] = str(tmp_path / k)
    p = tmp_path / "config" / "test.yaml"
    p.parent.mkdir()
    with open(p, "w") as f:
        yaml.safe_dump(conf, f)
    return Config.load(p)


@pytest.fixture
def grid():
    return from_origin(1_200_000, 480_000, RES, RES), (H, W)


def _write(path, arr, transform, count=1):
    arr = arr if arr.ndim == 3 else arr[None]
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2],
                       count=arr.shape[0], dtype=arr.dtype, crs=CRS, transform=transform) as d:
        d.write(arr)
    return path


@pytest.fixture
def synthetic_scene(cfg, grid):
    """Two years of fake imagery, a cutblock in 2021, a fire in 2021, full forest mask."""
    transform, shape = grid
    rng = np.random.default_rng(0)
    base = rng.normal(0.2, 0.02, (8, H, W)).astype(np.float32)
    prev = base.copy()
    curr = base.copy()
    # cutblock: rows 100-200, cols 100-200 -> NIR drops
    curr[3, 100:200, 100:200] -= 0.1
    # fire: rows 300-400, cols 300-400 -> NBR drops hard
    curr[5, 300:400, 300:400] += 0.1
    interim = cfg.path("interim")
    _write(interim / "s2_2020.tif", prev, transform)
    _write(interim / "s2_2021.tif", curr, transform)
    _write(interim / "forest_mask.tif", np.ones((H, W), np.uint8), transform)

    def px_box(r0, r1, c0, c1):
        x0, y0 = transform * (c0, r1) if False else (transform.c + c0 * transform.a, transform.f + r1 * transform.e)
        x1, y1 = transform * (c1, r0) if False else (transform.c + c1 * transform.a, transform.f + r0 * transform.e)
        return box(x0, y0, x1, y1)

    cut = gpd.GeoDataFrame({"HARVEST_YEAR": [2021, 2019]},
                           geometry=[px_box(100, 200, 100, 200), px_box(10, 30, 10, 30)], crs=CRS)
    fire = gpd.GeoDataFrame({"FIRE_YEAR": [2021]}, geometry=[px_box(300, 400, 300, 400)], crs=CRS)
    vri = gpd.GeoDataFrame({"BCLCS_LEVEL_2": ["T", "T"], "PROJ_AGE_1": [60, 200]},
                           geometry=[px_box(0, 256, 0, W), px_box(256, H, 0, W)], crs=CRS)
    raw = cfg.path("raw")
    cut.to_parquet(raw / "cutblocks.parquet")
    fire.to_parquet(raw / "fires.parquet")
    vri.to_parquet(raw / "vri.parquet")
    return {"cut": cut, "fire": fire, "vri": vri, "transform": transform, "shape": shape}
