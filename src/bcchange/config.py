"""Load the YAML config and expose typed paths."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Config:
    raw: dict
    root: Path

    @classmethod
    def load(cls, path: str | Path = "config/southcoast.yaml") -> Config:
        path = Path(path)
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(raw=raw, root=path.resolve().parent.parent)

    def __getitem__(self, key: str):
        return self.raw[key]

    def path(self, key: str) -> Path:
        p = self.root / self.raw["paths"][key]
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def crs(self) -> str:
        return self.raw["crs"]

    @property
    def years(self) -> list[int]:
        return list(self.raw["years"])

    @property
    def classes(self) -> dict[str, int]:
        return dict(self.raw["labels"]["classes"])
