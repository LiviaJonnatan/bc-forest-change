# bc-forest-change

> **Status: work in progress.** The pipeline scaffold is complete and tested offline, and all vector layers have been downloaded for the Sunshine Coast Natural Resource District. Imagery compositing and model training have not yet been run at full scale; see [Compute requirements](#compute-requirements) for what that takes.

Annual forest-change detection for British Columbia's South Coast from Sentinel-2 imagery, labelled with the province's own harvest and wildfire records, with an analysis layer that reports where the change is happening in terms a resource-stewardship analyst would use.

## What it does

Given a study area and a range of years, the pipeline:

1. Downloads the provincial vector layers that define the study area, the labels (harvest cutblocks, wildfire perimeters), and the policy context (forest inventory, streams, watersheds, old-growth management areas, parks).
2. Builds one cloud-masked summer median composite of Sentinel-2 imagery per year.
3. Rasterizes the harvest and fire polygons onto the imagery grid to produce per-year label maps (0 no change, 1 harvest, 2 fire), plus a forest mask so that water, alpine, and urban land are excluded from training and scoring.
4. Cuts (year t−1, year t) image pairs and year-t labels into 256×256 chips and assigns them to train/validation/test splits by watershed, so that neighbouring chips never leak across splits.
5. Trains an early-fusion U-Net with a pretrained ResNet encoder to classify change, and reports per-class IoU and F1 on the held-out watersheds.
6. Converts a predicted change map into policy tables: hectares of change by stand age class, change inside riparian buffers and protected areas, and detected change with no matching cutblock or fire record ("undocumented change").

The question behind it: where did forest cover change in the South Coast each year since 2018, how much of it is recorded in provincial systems, and how much falls in riparian buffers, old-growth management areas, and old stands?

## Why this design

- **Administrative labels, not model labels.** Training targets come from the Consolidated Cutblocks and historical fire perimeter datasets, maintained by the Ministry of Forests and the BC Wildfire Service, rather than from a prior global change product. The one exception is noted in the data table below.
- **Spatial-block evaluation.** Random chip splits in remote sensing overstate accuracy because adjacent chips are nearly identical. Splits here are by Freshwater Atlas assessment watershed.
- **Label noise is a finding.** Cutblock dates lag reality, and some cutblock polygons are themselves derived from satellite change detection. Stage 6's undocumented-change table is where that surfaces, and it is the most policy-relevant output.
- **Idempotent stages.** Every stage skips work whose outputs exist, so a run that fails on the cloudiest year can be resumed.

## Data sources

All provincial layers are from the [BC Data Catalogue](https://catalogue.data.gov.bc.ca) under the Open Government Licence – British Columbia, pulled through the public WFS endpoint at `openmaps.gov.bc.ca` via the [`bcdata`](https://github.com/smnorris/bcdata) client. No account is required.

| Layer | Catalogue object | Used for | Notes from the live schema |
|---|---|---|---|
| Natural Resource Districts / Regions | `WHSE_ADMIN_BOUNDARIES.ADM_NR_DISTRICTS_SPG`, `ADM_NR_REGIONS_SPG` | Study-area boundary | Generalized geometry (`_SPG`). The district polygon includes marine area; intersect with watersheds for land only. |
| Harvested Areas of BC (Consolidated Cutblocks) | `WHSE_FOREST_VEGETATION.VEG_CONSOLIDATED_CUT_BLOCKS_SP` | Harvest labels | Year field is `HARVEST_START_YEAR_CALENDAR`, normalised to `HARVEST_YEAR` on download. `DATA_SOURCE` distinguishes RESULTS (administrative), VRI, and *Satellite Imagery – Change Detection*; the last is model-derived and is kept as an attribute so results can be reported with and without it. |
| BC Wildfire Fire Perimeters – Historical | `WHSE_LAND_AND_NATURAL_RESOURCE.PROT_HISTORICAL_FIRE_POLYS_SP` | Fire labels | `FIRE_YEAR` is the fiscal year (April–March); the calendar year is recomputed from `FIRE_DATE`. Multiple `VERSION_NUMBER`s per fire are deduplicated to the latest. |
| VRI Forest Vegetation Composite Rank 1 (2024) | `WHSE_FOREST_VEGETATION.VEG_COMP_LYR_R1_POLY` | Forest mask (`BCLCS_LEVEL_2 = 'T'`), stand age (`PROJ_AGE_1`), BEC zone | 7.15 M polygons province-wide; downloaded filtered to treed polygons. **Caveat:** the 2024 vintage is updated for depletions, so stands harvested within the study window already show post-harvest ages. The age-class analysis needs a pre-2018 VRI vintage. |
| Freshwater Atlas Stream Network | `WHSE_BASEMAPPING.FWA_STREAM_NETWORKS_SP` | Riparian buffers | Downloaded at `STREAM_ORDER >= 2`; arcs inside waterbodies (`WATERBODY_KEY` set) are dropped so buffers are not drawn through lakes. |
| Freshwater Atlas Assessment Watersheds | `WHSE_BASEMAPPING.FWA_ASSESSMENT_WATERSHEDS_POLY` | Spatial blocks for the train/val/test split | |
| Old Growth Management Areas (legal) | `WHSE_LAND_USE_PLANNING.RMP_OGMA_LEGAL_CURRENT_SVW` | Protected-area overlay | |
| Parks, Ecological Reserves and Protected Areas | `WHSE_TANTALIS.TA_PARK_ECORES_PA_SVW` | Protected-area overlay | |

Imagery is Copernicus Sentinel-2 Level-2A surface reflectance, accessed through the [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com) STAC API (free; no key required for search). Bands B02, B03, B04, B08, B11, B12 plus derived NDVI and NBR; the Scene Classification Layer masks cloud, shadow, cirrus and snow.

### Current data pull (Sunshine Coast NRD, 2017–2024)

| Layer | Features | Area |
|---|---:|---:|
| District AOI (incl. marine) | 1 | 1,904,769 ha |
| Watersheds (land) | 366 | 1,547,090 ha |
| VRI, treed | 76,845 | 808,393 ha |
| Cutblocks | 2,573 | 18,067 ha |
| Fire perimeters | 214 | 54,243 ha |
| OGMAs | 2,561 | 54,414 ha |
| Parks | 51 | 59,822 ha |
| Streams (order ≥ 2) | 21,595 arcs | |

Harvest is a rare class, roughly 0.3 % of the treed area per year. Training subsamples no-change chips rather than using them all.

## Compute requirements

The vector download runs on a laptop in a few minutes (VRI may need the paged `bcdata dump` route if the WFS times out). The remaining stages are heavier.

| Stage | What it needs | Estimate for the full district |
|---|---|---|
| 2 Composites | Bandwidth and disk. Each year is ~190 M pixels × 8 bands of float32, about 6 GB uncompressed before COG compression; the median requires streaming 15–40 scenes per year through dask. | ~40 GB disk for 7 years; several hours per year on a home connection. |
| 4 Chips | Disk only. | ~3,000 chips per year pair before filtering; a few GB compressed. |
| 5 Training | A GPU. A ResNet-18 U-Net on 256×256 chips trains in roughly 10–20 minutes per epoch on a free Colab T4 for a district-sized chip set. | 15 epochs ≈ 3–5 GPU-hours. |
| 6 Analysis | CPU and GeoPandas. | Minutes. |

**Recommended first run:** a 20 km × 20 km test box around a harvest-heavy area for two years (set `aoi.test_bbox` and `years` in the config). That is a few hundred MB of imagery, finishes in minutes on a laptop, and verifies the whole chain end to end before committing GPU time.

## Install and run

```bash
uv venv && uv pip install -e ".[dev,fetch]"      # or: pip install -e ".[dev,fetch]"
uv run pytest -q                                 # 8 offline tests on synthetic data

uv run bcchange fetch-vectors                    # stage 1
uv run bcchange build-composites                 # stage 2  (set test_bbox first)
uv run bcchange make-labels                      # stage 3
uv run bcchange make-chips                       # stage 4
uv run bcchange train --epochs 15                # stage 5  (GPU)
uv run bcchange analyze 2023 outputs/pred_2023.tif   # stage 6
```

Configuration is in `config/southcoast.yaml`. Set `aoi.district_name` to a district for a smaller study area, or leave it null for the whole South Coast region.

## Repository layout

```
src/bcchange/
  config.py     YAML config loader
  vectors.py    stage 1: WFS download, clip, field normalisation, GeoParquet cache
  imagery.py    stage 2: STAC search, SCL masking, yearly median composite
  labels.py     stage 3: rasterize cutblocks/fires by year; VRI forest mask
  tiles.py      stage 4: chip extraction; watershed-block split
  model.py      stage 5: dataset, U-Net, training loop (torch optional)
  metrics.py    confusion-matrix IoU/F1 (torch-free)
  analysis.py   stage 6: policy overlays and undocumented-change detection
  cli.py        `bcchange` command-line entry point
tests/          offline tests on a synthetic study area
config/         study-area and pipeline settings
```

## Licence

Code: MIT. Data: Open Government Licence – British Columbia; Copernicus Sentinel data terms.
