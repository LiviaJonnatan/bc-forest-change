"""Stage 1 - fetch vector layers from the BC Geographic Warehouse (WFS).

All layers are pulled through the public WFS at openmaps.gov.bc.ca using the
`bcdata` client, clipped to the area of interest, reprojected to BC Albers
and cached as GeoParquet under data/raw/. Nothing here needs an account.

If `bcdata` cannot reach a layer, `fetch_layer` raises with the layer name so
the failure is obvious, and the rest of the pipeline can run on whichever
layers did download (see labels.py for the fire/harvest fallbacks).
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from .config import Config


def _bcdata():
    try:
        import bcdata  # noqa: WPS433
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install 'bcchange[fetch]' to enable WFS downloads") from e
    return bcdata


def fetch_aoi(cfg: Config) -> gpd.GeoDataFrame:
    """Return the study-area polygon (region, or district if configured)."""
    bcdata = _bcdata()
    aoi = cfg["aoi"]
    if aoi.get("district_name"):
        layer, field, name = cfg["layers"]["districts"], "DISTRICT_NAME", aoi["district_name"]
    else:
        layer, field, name = cfg["layers"]["regions"], "REGION_NAME", aoi["region_name"]
    gdf = bcdata.get_data(layer, query=f"{field} = '{name}'", as_gdf=True).to_crs(cfg.crs)
    if gdf.empty:
        raise ValueError(f"No feature found: {layer} where {field}='{name}'")
    return gdf.dissolve()[["geometry"]]


def fetch_layer(cfg: Config, key: str, aoi: gpd.GeoDataFrame, query: str | None = None,
                force: bool = False) -> gpd.GeoDataFrame:
    """Fetch one configured layer clipped to the AOI; cache to GeoParquet."""
    out = cfg.path("raw") / f"{key}.parquet"
    if out.exists() and not force:
        return gpd.read_parquet(out)
    bcdata = _bcdata()
    layer = cfg["layers"][key]
    gdf = bcdata.get_data(layer, query=query, bounds=list(aoi.total_bounds),
                          bounds_crs=cfg.crs, as_gdf=True).to_crs(cfg.crs)
    gdf = gpd.clip(gdf, aoi)
    if key == "cutblocks" and "HARVEST_YEAR" not in gdf:
        gdf["HARVEST_YEAR"] = gdf["HARVEST_START_YEAR_CALENDAR"].astype("Int64")
    if key == "fires":
        cal = pd.to_datetime(gdf["FIRE_DATE"], errors="coerce").dt.year
        gdf["FIRE_YEAR"] = cal.fillna(gdf["FIRE_YEAR"]).astype("Int64")
        gdf = (gdf.sort_values("VERSION_NUMBER")
                  .drop_duplicates(subset=["FIRE_NUMBER", "FIRE_YEAR"], keep="last"))
    if key == "streams":
        gdf = gdf[gdf["WATERBODY_KEY"].isna() | (gdf["WATERBODY_KEY"] == 0)]
    gdf.to_parquet(out)
    return gdf


def fetch_all(cfg: Config, force: bool = False) -> dict[str, Path]:
    """Download every label and context layer for the AOI."""
    aoi = fetch_aoi(cfg)
    aoi.to_parquet(cfg.path("raw") / "aoi.parquet")
    y0 = min(cfg.years) - 1
    plan = {
        "cutblocks": f"HARVEST_START_YEAR_CALENDAR >= {y0}",
        "fires": f"FIRE_YEAR >= {y0 - 1}",
        "vri": "BCLCS_LEVEL_2 = 'T'",
        "streams": "STREAM_ORDER >= 2",   # keep the FWA pull manageable
        "watersheds": None,
        "ogma": None,
        "parks": None,
    }
    done = {}
    for key, q in plan.items():
        try:
            fetch_layer(cfg, key, aoi, query=q, force=force)
            done[key] = cfg.path("raw") / f"{key}.parquet"
        except Exception as e:  # keep going; report at the end
            print(f"[vectors] {key}: FAILED ({e})")
    return done
