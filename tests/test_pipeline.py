import numpy as np
import pandas as pd

from bcchange import analysis, labels, tiles
from bcchange.metrics import confusion, metrics_from_confusion


def test_label_rasterization(cfg, synthetic_scene):
    s = synthetic_scene
    lab = labels.build_label(cfg, 2021, s["cut"], s["fire"], s["transform"], s["shape"])
    assert lab[150, 150] == cfg.classes["harvest"]
    assert lab[350, 350] == cfg.classes["fire"]
    assert lab[20, 20] == 0           # 2019 cutblock must not appear in 2021 label
    assert lab[450, 50] == 0
    # areas: 100x100 px at 10 m = 100 ha each
    assert (lab == 1).sum() == 100 * 100
    assert (lab == 2).sum() == 100 * 100


def test_fire_overrides_harvest(cfg, synthetic_scene):
    s = synthetic_scene
    fire = s["fire"].copy()
    fire.loc[0, "geometry"] = s["cut"].geometry[0]  # same footprint
    lab = labels.build_label(cfg, 2021, s["cut"], fire, s["transform"], s["shape"])
    assert lab[150, 150] == cfg.classes["fire"]


def test_forest_mask(cfg, synthetic_scene):
    s = synthetic_scene
    fm = labels.forest_mask(s["vri"], s["transform"], s["shape"])
    assert fm.mean() == 1.0


def test_build_all_labels_writes_rasters(cfg, synthetic_scene):
    outs = labels.build_all(cfg)
    assert set(outs) == {2020, 2021}
    assert (cfg.path("interim") / "forest_mask.tif").exists()


def test_chips_and_block_split(cfg, synthetic_scene):
    labels.build_all(cfg)
    idx = pd.read_parquet(tiles.build_all(cfg))
    assert len(idx) == 16                         # 512/128 squared, one year pair
    assert set(idx.split) <= {"train", "val", "test"}
    # Split is by block: every block maps to exactly one split
    assert (idx.groupby("block").split.nunique() == 1).all()
    # Chips covering the cutblock/fire report change
    assert (idx.change_frac > 0).sum() >= 2
    z = np.load(idx.path[0])
    assert z["prev"].shape == (8, 128, 128) and z["label"].shape == (128, 128)


def test_split_blocks_fractions():
    blocks = np.repeat(np.arange(100), 5)
    sp = tiles.split_blocks(blocks, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=1)
    frac = pd.Series(sp).value_counts(normalize=True)
    assert abs(frac["train"] - 0.7) < 0.05


def test_metrics():
    pred = np.array([0, 1, 1, 2, 0, 2])
    true = np.array([0, 1, 0, 2, 0, 1])
    cm = confusion(pred, true, 3)
    m = metrics_from_confusion(cm, ["no", "harvest", "fire"]).set_index("class")
    assert m.loc["harvest", "recall"] == 0.5
    assert m.loc["fire", "precision"] == 0.5


def test_analysis_tables(cfg, synthetic_scene):
    labels.build_all(cfg)
    out = analysis.run(cfg, cfg.path("interim") / "label_2021.tif", 2021, "harvest")
    assert abs(out["total"].ha.iloc[0] - 100) < 0.5
    age = out["by_age_class"].set_index("age_class")
    assert abs(age.loc["40-80", "ha"] - 100) < 0.5       # cutblock sits in the 60-yr stand
    assert out["undocumented"].ha.iloc[0] < 0.5          # fully covered by the cutblock polygon
