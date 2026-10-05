"""Stage 2 - build one cloud-masked summer median composite per year.

Source: Sentinel-2 L2A on Microsoft Planetary Computer (free, no key needed
for search; `planetary_computer.sign` handles asset URLs). Scenes are filtered
by cloud cover, cloud/shadow/snow pixels are removed with the Scene
Classification Layer (SCL), and the per-pixel median over the season is taken.
Output: one Cloud-Optimised GeoTIFF per year in BC Albers at 10 m with bands
B02,B03,B04,B08,B11,B12 plus NDVI and NBR as extra bands.

The South Coast is cloudy; the June-September window and a 40% scene cap are
there to get enough clear observations. If a year's composite still has
large gaps, widen the window in the config rather than lowering the mask.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from .config import Config

# SCL classes to drop: 0 nodata, 1 saturated, 3 cloud shadow, 8/9 cloud, 10 cirrus, 11 snow
SCL_BAD = (0, 1, 3, 8, 9, 10, 11)


def add_indices(da: xr.DataArray) -> xr.DataArray:
    """Append NDVI and NBR bands to a (band, y, x) reflectance array."""
    nir, red, swir2 = da.sel(band="B08"), da.sel(band="B04"), da.sel(band="B12")
    ndvi = ((nir - red) / (nir + red + 1e-6)).expand_dims(band=["NDVI"])
    nbr = ((nir - swir2) / (nir + swir2 + 1e-6)).expand_dims(band=["NBR"])
    return xr.concat([da, ndvi, nbr], dim="band")


def build_composite(cfg: Config, year: int, aoi_bounds_wgs84: tuple[float, float, float, float],
                    force: bool = False) -> Path:
    out = cfg.path("interim") / f"s2_{year}.tif"
    if out.exists() and not force:
        return out
    try:
        import planetary_computer as pc
        import pystac_client
        import stackstac
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install 'bcchange[fetch]' to enable imagery downloads") from e

    img = cfg["imagery"]
    s = cfg["season"]
    catalog = pystac_client.Client.open(img["stac_url"], modifier=pc.sign_inplace)
    items = catalog.search(
        collections=[img["collection"]],
        bbox=aoi_bounds_wgs84,
        datetime=f"{year}-{s['start']}/{year}-{s['end']}",
        query={"eo:cloud_cover": {"lt": cfg["max_scene_cloud_pct"]}},
    ).item_collection()
    if len(items) == 0:
        raise RuntimeError(f"No Sentinel-2 scenes for {year}")

    stack = stackstac.stack(items, assets=img["bands"], epsg=int(cfg.crs.split(":")[1]),
                            resolution=cfg["resolution_m"], bounds_latlon=aoi_bounds_wgs84,
                            dtype="float32", fill_value=np.nan, chunksize=2048)
    scl = stack.sel(band="SCL")
    good = ~scl.isin(SCL_BAD)
    refl = stack.sel(band=[b for b in img["bands"] if b != "SCL"]).where(good)
    comp = refl.median(dim="time", skipna=True)
    comp = add_indices(comp)
    comp = comp.rio.write_crs(cfg.crs)
    comp.rio.to_raster(out, driver="COG", compress="deflate")
    return out


def build_all(cfg: Config, aoi_bounds_wgs84, force: bool = False) -> dict[int, Path]:
    return {y: build_composite(cfg, y, aoi_bounds_wgs84, force=force) for y in cfg.years}
