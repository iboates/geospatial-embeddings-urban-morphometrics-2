"""Equivalent rectangular index (ERI) metric.

Ratio comparing the building's area and perimeter to an equivalent rectangle.
High ERI → rectangular footprints (row houses, slabs); low ERI → complex
footprints (L-shapes, courtyards).
"""

from pathlib import Path

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series
from urban_morphometrics.metrics.features import write_features


@register("equivalent_rectangular_index")
def compute(ctx: CellContext, num_quantiles: int, features_dir: Path | None = None) -> dict:
    """ERI per building — compares area and perimeter to an equivalent rectangle."""
    b = ctx.buildings_ea
    empty = pd.Series(dtype=float)
    if b.empty:
        return {
            **aggregate_series(empty, "equivalent_rectangular_index", num_quantiles),
            **aggregate_series(empty, "equivalent_rectangular_index_joined", num_quantiles),
        }
    values = momepy.equivalent_rectangular_index(b)

    d = ctx.dissolved_buildings_ea
    joined_values = momepy.equivalent_rectangular_index(d) if not d.empty else empty

    if features_dir is not None:
        write_features(b[["geometry"]].assign(equivalent_rectangular_index=values), features_dir / "equivalent_rectangular_index.gpkg")
        write_features(d[["geometry"]].assign(equivalent_rectangular_index_joined=joined_values), features_dir / "equivalent_rectangular_index_joined.gpkg")

    result = aggregate_series(values, "equivalent_rectangular_index", num_quantiles)
    result.update(aggregate_series(joined_values, "equivalent_rectangular_index_joined", num_quantiles))
    return result
