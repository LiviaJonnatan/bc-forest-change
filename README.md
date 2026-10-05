# bc-forest-change

Annual forest-change detection for British Columbia's South Coast from Sentinel-2 imagery, trained and validated against the province's own harvest and wildfire records, with a policy analysis layer that asks where the change is happening.

**Question.** Where did forest cover change in the South Coast Natural Resource Region each year since 2018, how much of it is documented in the Consolidated Cutblocks and fire-perimeter records, and how much falls in riparian buffers, old-growth management areas, and old stands?

**Why it is more than a CV demo.** The labels are administrative polygons from the BC Data Catalogue, not another model's output, and the final tables are the ones a resource-stewardship analyst produces: hectares by stand age, change inside protected zones, and detections with no matching record.

## Pipeline

| Stage | Command | Input | Output |
|---|---|---|---|
| 1 Vectors | `bcchange fetch-vectors` | BC Geographic Warehouse WFS (`bcdata`) | `data/raw/*.parquet` (AOI, cutblocks, fires, VRI, streams, watersheds, OGMA, parks) |
| 2 Imagery | `bcchange build-composites` | Sentinel-2 L2A via Planetary Computer STAC | `data/interim/s2_<year>.tif` cloud-masked summer medians, 10 m, BC Albers, 8 bands incl. NDVI and NBR |
| 3 Labels | `bcchange make-labels` | stage 1 + stage 2 grid | `data/interim/label_<year>.tif` (0 none, 1 harvest, 2 fire) and `forest_mask.tif` from VRI |
| 4 Chips | `bcchange make-chips` | stages 2–3 | `data/processed/chips/*.npz` and `chip_index.parquet` with watershed-block train/val/test split |
| 5 Model | `bcchange train` | stage 4 | early-fusion U-Net (pretrained ResNet encoder), `outputs/model/test_metrics.csv` |
| 6 Analysis | `bcchange analyze <year> <raster>` | prediction raster + stage 1 | `outputs/analysis_<year>/*.csv`, undocumented-change polygons as GeoPackage |

Every stage is idempotent and resumable. Configuration lives in `config/southcoast.yaml`; switch `aoi.district_name` to `"Sunshine Coast Natural Resource District"` for a smaller first run.

## Install

```bash
pip install -e ".[dev]"            # core + tests (runs offline on synthetic data)
pip install -e ".[fetch]"          # bcdata, pystac-client, stackstac for stages 1–2
pip install -e ".[model]"          # torch, segmentation-models-pytorch for stage 5
pytest                             # 8 offline tests
```

## Design decisions worth knowing

- **Labels are per-year, not cumulative.** A pixel is "harvest in 2021" only if a cutblock with `HARVEST_YEAR == 2021` covers it. Fire overrides harvest on overlap. Polygons under 0.5 ha are dropped.
- **Forest mask from VRI.** Training and scoring are restricted to `BCLCS_LEVEL_2 == 'T'`, so ocean, alpine and cities are not counted as easy "no change" wins.
- **Spatial-block split.** Chips are assigned to train/val/test by FWA assessment watershed (falling back to a 10 km grid). Random chip splits in remote sensing leak through spatial autocorrelation and overstate accuracy.
- **Early fusion with a difference channel.** Input is `[year t-1, year t, t - (t-1)]` stacked on channels; a plain U-Net on that is a strong baseline and trains in minutes on a free GPU. A Siamese encoder is the obvious next step if it plateaus.
- **Cloud.** The South Coast is the cloudiest region in BC. Composites use June–September, a 40 % scene cloud cap, SCL masking, and a per-pixel median; chips with more than 20 % gaps are dropped.
- **Label noise is a result, not a nuisance.** Cutblock dates can lag reality. Stage 6's "undocumented change" table (detections with no cutblock or fire within ±1 year) is where that shows up, and it is the most policy-relevant output.

## Data sources

All provincial layers are from the [BC Data Catalogue](https://catalogue.data.gov.bc.ca) under the Open Government Licence – BC, pulled through the public WFS so no account is needed: Harvested Areas of BC (Consolidated Cutblocks), BC Wildfire Fire Perimeters – Historical, VRI Forest Vegetation Composite Rank 1, Freshwater Atlas streams and assessment watersheds, Old Growth Management Areas, Parks and Protected Areas, Natural Resource Region and District boundaries. Imagery is Copernicus Sentinel-2 L2A via Microsoft Planetary Computer.

If a label layer is unavailable, the NTEMS/NFIS Canada-wide harvest and fire rasters (Canadian Forest Service, 30 m, 1985–2020) are a drop-in substitute; see `labels.py`.

## Roadmap

1. First run on Sunshine Coast district, 2018–2024, harvest + fire.
2. Add built-up/agricultural conversion as a third class in the Fraser Valley using NDVI persistence, scored against the Agricultural Land Reserve.
3. Siamese encoder and temporal (multi-year) input.
4. Serve predictions as map tiles behind a small web map.
