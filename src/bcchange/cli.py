"""Command-line entry point: `bcchange <stage>`.

    bcchange fetch-vectors      stage 1  BC Data Catalogue layers -> data/raw/*.parquet
    bcchange build-composites   stage 2  Sentinel-2 yearly composites -> data/interim/s2_*.tif
    bcchange make-labels        stage 3  label rasters + forest mask -> data/interim/
    bcchange make-chips         stage 4  chips + spatial-block split -> data/processed/
    bcchange train              stage 5  model, metrics -> outputs/model/
    bcchange analyze            stage 6  policy tables for one year -> outputs/analysis_<year>/
    bcchange run-all            stages 1-4 (5 needs a GPU session; 6 needs a prediction)

Every stage is idempotent: it skips work whose outputs already exist unless
--force is passed, so a failed run can be resumed.
"""
from __future__ import annotations

from pathlib import Path

import typer

from .config import Config

app = typer.Typer(add_completion=False, help=__doc__)
CFG = typer.Option("config/southcoast.yaml", "--config", "-c")


@app.command("fetch-vectors")
def fetch_vectors(config: str = CFG, force: bool = False):
    from . import vectors
    cfg = Config.load(config)
    done = vectors.fetch_all(cfg, force=force)
    typer.echo(f"fetched: {sorted(done)}")


@app.command("build-composites")
def build_composites(config: str = CFG, force: bool = False):
    import geopandas as gpd

    from . import imagery
    cfg = Config.load(config)
    aoi = gpd.read_parquet(cfg.path("raw") / "aoi.parquet").to_crs("EPSG:4326")
    paths = imagery.build_all(cfg, tuple(aoi.total_bounds), force=force)
    typer.echo(f"composites: {list(paths.values())}")


@app.command("make-labels")
def make_labels(config: str = CFG):
    from . import labels
    cfg = Config.load(config)
    typer.echo(f"labels: {list(labels.build_all(cfg).values())}")


@app.command("make-chips")
def make_chips(config: str = CFG):
    import pandas as pd

    from . import tiles
    cfg = Config.load(config)
    idx = pd.read_parquet(tiles.build_all(cfg))
    typer.echo(idx.groupby(["year", "split"]).size().unstack(fill_value=0).to_string())
    typer.echo(f"\nchips with any change: {(idx.change_frac > 0).mean():.1%}")


@app.command("train")
def train(config: str = CFG, epochs: int = 15, batch_size: int = 16, encoder: str = "resnet18"):
    from . import model
    cfg = Config.load(config)
    out = cfg.path("outputs") / "model"
    out.mkdir(exist_ok=True)
    names = [k for k, _ in sorted(cfg.classes.items(), key=lambda kv: kv[1])]
    test = model.train(cfg.path("processed") / "chip_index.parquet", out, len(names), names,
                       epochs=epochs, batch_size=batch_size, encoder=encoder)
    typer.echo(test.round(3).to_string(index=False))


@app.command("analyze")
def analyze(year: int, change_raster: Path, config: str = CFG, class_name: str = "harvest"):
    from . import analysis
    cfg = Config.load(config)
    out = analysis.run(cfg, change_raster, year, class_name)
    for k, df in out.items():
        typer.echo(f"\n== {k} ==\n{df.round(1).to_string(index=False)}")


@app.command("run-all")
def run_all(config: str = CFG, force: bool = False):
    fetch_vectors(config, force)
    build_composites(config, force)
    make_labels(config)
    make_chips(config)


if __name__ == "__main__":
    app()
