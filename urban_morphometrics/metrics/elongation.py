"""Elongation metric.

Ratio of the shorter to the longer side of the minimum bounding rectangle.
Values near 1 indicate compact/square footprints; low values indicate elongated
buildings such as row houses or linear strips.
"""

from pathlib import Path

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series
from urban_morphometrics.metrics.features import write_features


@register("elongation")
def compute(ctx: CellContext, num_quantiles: int, features_dir: Path | None = None) -> dict:
    """Shorter/longer side ratio of the minimum bounding rectangle per building (0–1)."""
    b = ctx.buildings_ea
    empty = pd.Series(dtype=float)
    if b.empty:
        return {
            **aggregate_series(empty, "elongation", num_quantiles),
            **aggregate_series(empty, "elongation_joined", num_quantiles),
        }
    values = momepy.elongation(b)

    d = ctx.dissolved_buildings_ea
    joined_values = momepy.elongation(d) if not d.empty else empty

    if features_dir is not None:
        write_features(b[["geometry"]].assign(elongation=values), features_dir / "elongation.gpkg")
        write_features(d[["geometry"]].assign(elongation_joined=joined_values), features_dir / "elongation_joined.gpkg")

    result = aggregate_series(values, "elongation", num_quantiles)
    result.update(aggregate_series(joined_values, "elongation_joined", num_quantiles))
    return result
