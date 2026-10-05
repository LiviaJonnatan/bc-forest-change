"""Stage 6 - turn a change raster into the tables a ministry would ask for.

Inputs: a predicted (or labelled) change raster for a year, plus the vector
context layers. Outputs, all in hectares:

  * change by VRI projected age class (is loss concentrated in old stands?)
  * change inside riparian buffers around FWA streams
  * change inside Old Growth Management Areas and parks
  * detected change with no overlapping cutblock or fire polygon that year
    ("undocumented change": label lag, unrecorded clearing, or model error)

Everything is vectorised from the raster once (`polygonize_change`) and then
overlaid, so each table is a GeoPandas overlay rather than a pixel loop.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio import features
from shapely.geometry import shape

from .config import Config

HA = 10_000.0
AGE_BINS = [0, 40, 80, 140, 250, 10_000]
AGE_LABELS = ["<40", "40-80", "80-140", "140-250", "250+"]


def polygonize_change(raster: Path, class_value: int, crs: str) -> gpd.GeoDataFrame:
    with rasterio.open(raster) as src:
        arr = src.read(1)
        mask = arr == class_value
        geoms = [shape(g) for g, v in features.shapes(arr, mask=mask, transform=src.transform)]
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
    gdf["ha"] = gdf.geometry.area / HA
    return gdf


def _overlay_ha(change: gpd.GeoDataFrame, zone: gpd.GeoDataFrame, zone_col: str | None = None) -> pd.DataFrame:
    if zone.empty or change.empty:
        return pd.DataFrame()
    ov = gpd.overlay(change[["geometry"]], zone, how="intersection", keep_geom_type=False)
    ov["ha"] = ov.geometry.area / HA
    if zone_col:
        return ov.groupby(zone_col, as_index=False)["ha"].sum()
    return pd.DataFrame({"ha": [ov["ha"].sum()]})


def by_age_class(change: gpd.GeoDataFrame, vri: gpd.GeoDataFrame) -> pd.DataFrame:
    v = vri[["geometry", "PROJ_AGE_1"]].copy()
    v["age_class"] = pd.cut(v["PROJ_AGE_1"].fillna(0), AGE_BINS, labels=AGE_LABELS, right=False)
    out = _overlay_ha(change, v[["geometry", "age_class"]], "age_class")
    total = out["ha"].sum() if not out.empty else 0
    if total:
        out["share"] = out["ha"] / total
    return out


def in_riparian(change: gpd.GeoDataFrame, streams: gpd.GeoDataFrame, buffer_m: float) -> pd.DataFrame:
    buf = gpd.GeoDataFrame(geometry=[streams.geometry.buffer(buffer_m).union_all()], crs=streams.crs)
    inside = _overlay_ha(change, buf)
    return pd.DataFrame({"zone": ["riparian"], "ha": [inside["ha"].iloc[0] if not inside.empty else 0.0],
                         "total_change_ha": [change["ha"].sum()]})


def in_protected(change: gpd.GeoDataFrame, ogma: gpd.GeoDataFrame | None,
                 parks: gpd.GeoDataFrame | None) -> pd.DataFrame:
    rows = []
    for name, layer in [("OGMA", ogma), ("parks", parks)]:
        if layer is None or layer.empty:
            continue
        ov = _overlay_ha(change, layer[["geometry"]])
        rows.append({"zone": name, "ha": ov["ha"].iloc[0] if not ov.empty else 0.0})
    return pd.DataFrame(rows)


def undocumented(change: gpd.GeoDataFrame, cutblocks: gpd.GeoDataFrame | None,
                 fires: gpd.GeoDataFrame | None, year: int, tolerance_years: int = 1) -> gpd.GeoDataFrame:
    """Predicted change polygons not covered by any cutblock/fire within +-tolerance years."""
    docs = []
    if cutblocks is not None:
        docs.append(cutblocks[cutblocks.HARVEST_YEAR.between(year - tolerance_years, year + tolerance_years)][["geometry"]])
    if fires is not None:
        docs.append(fires[fires.FIRE_YEAR.between(year - tolerance_years, year + tolerance_years)][["geometry"]])
    if not docs:
        return change
    documented = gpd.GeoDataFrame(pd.concat(docs, ignore_index=True), crs=change.crs)
    if documented.empty:
        return change
    left = gpd.overlay(change, documented, how="difference", keep_geom_type=False)
    left["ha"] = left.geometry.area / HA
    return left[left["ha"] >= 0.5]


def run(cfg: Config, change_raster: Path, year: int, class_name: str = "harvest") -> dict[str, pd.DataFrame]:
    raw = cfg.path("raw")

    def rd(k):
        f = raw / f"{k}.parquet"
        return gpd.read_parquet(f) if f.exists() else None

    change = polygonize_change(change_raster, cfg.classes[class_name], cfg.crs)
    out = {"total": pd.DataFrame({"year": [year], "class": [class_name], "ha": [change["ha"].sum()]})}
    if (vri := rd("vri")) is not None:
        out["by_age_class"] = by_age_class(change, vri)
    if (streams := rd("streams")) is not None:
        out["riparian"] = in_riparian(change, streams, cfg["labels"]["riparian_buffer_m"])
    out["protected"] = in_protected(change, rd("ogma"), rd("parks"))
    und = undocumented(change, rd("cutblocks"), rd("fires"), year)
    out["undocumented"] = pd.DataFrame({"year": [year], "n_polygons": [len(und)], "ha": [und["ha"].sum()]})
    odir = cfg.path("outputs") / f"analysis_{year}"
    odir.mkdir(exist_ok=True)
    for k, df in out.items():
        df.to_csv(odir / f"{k}.csv", index=False)
    und.to_file(odir / "undocumented.gpkg", driver="GPKG") if len(und) else None
    return out
