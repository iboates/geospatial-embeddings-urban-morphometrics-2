"""Volume metric.

Footprint area multiplied by building height. Height is resolved in priority
order: OSM 'height' tag → 'building:levels' × 3 m → default 6 m.
All volumes are in cubic metres (equal-area CRS for footprint area).
"""

from pathlib import Path

import pandas as pd

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series
from urban_morphometrics.metrics.features import write_features


@register("volume")
def compute(ctx: CellContext, num_quantiles: int, features_dir: Path | None = None) -> dict:
    """Compute building volume statistics for the focal cell.

    Volume = footprint area (m²) × height (m). Height resolution priority:
    OSM 'height' tag, then 'building:levels' × 3 m, then 6 m default.
    """
    b = ctx.buildings_with_height
    if b.empty:
        return aggregate_series(pd.Series([], dtype=float), "volume", num_quantiles)

    volume = b.geometry.area * b["height"]
    if features_dir is not None:
        write_features(b[["geometry"]].assign(volume=volume), features_dir / "volume.gpkg")
    return aggregate_series(volume, "volume", num_quantiles)
