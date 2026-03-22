"""Shared walls metric.

Length of wall shared between adjacent (touching) buildings. Also computes a
ratio: shared wall length as a fraction of each building's total perimeter.
High values indicate terraced or attached housing; zero means a free-standing building.

Requires neighbourhood context so buildings at cell edges can touch buildings
outside the focal cell.
"""

from pathlib import Path

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series
from urban_morphometrics.metrics.features import write_features

_EMPTY = pd.Series(dtype=float)


@register("shared_walls")
def compute(ctx: CellContext, num_quantiles: int, features_dir: Path | None = None) -> dict:
    """Shared wall length and shared-wall-to-perimeter ratio per focal building."""
    b = ctx.buildings_ea
    if b.empty:
        return {
            **aggregate_series(_EMPTY, "shared_walls", num_quantiles),
            **aggregate_series(_EMPTY, "shared_walls_ratio", num_quantiles),
        }

    all_b = ctx.focal_plus_neighbourhood_buildings
    sw_all = momepy.shared_walls(all_b)

    focal_sw = sw_all.reindex(b.index)
    perimeter = b.geometry.length
    # Clip to [0, 1] — values marginally above 1 can occur due to floating-point
    focal_ratio = (focal_sw / perimeter).clip(0, 1)

    if features_dir is not None:
        write_features(
            b[["geometry"]].assign(shared_walls=focal_sw, shared_walls_ratio=focal_ratio),
            features_dir / "shared_walls.gpkg",
        )

    return {
        **aggregate_series(focal_sw, "shared_walls", num_quantiles),
        **aggregate_series(focal_ratio, "shared_walls_ratio", num_quantiles),
    }
