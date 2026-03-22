"""Fractal dimension metric.

2 * log(perimeter/4) / log(area). ~1.0 = very simple (square); 1.05–1.15 =
moderately articulated; >1.20 = complex/fragmented. Computed twice:
- `fractal_dimension`: per raw individual building
- `fractal_dimension_joined`: per dissolved structure (touching buildings merged)
"""

from pathlib import Path

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series
from urban_morphometrics.metrics.features import write_features


@register("fractal_dimension")
def compute(ctx: CellContext, num_quantiles: int, features_dir: Path | None = None) -> dict:
    """2·log(perimeter/4)/log(area) per raw building and per dissolved structure."""
    b = ctx.buildings_ea
    empty = pd.Series(dtype=float)
    if b.empty:
        return {
            **aggregate_series(empty, "fractal_dimension", num_quantiles),
            **aggregate_series(empty, "fractal_dimension_joined", num_quantiles),
        }

    raw_values = momepy.fractal_dimension(b)
    if features_dir is not None:
        write_features(b[["geometry"]].assign(fractal_dimension=raw_values), features_dir / "fractal_dimension.gpkg")
    result = aggregate_series(raw_values, "fractal_dimension", num_quantiles)

    dissolved = ctx.dissolved_buildings_ea
    joined_values = momepy.fractal_dimension(dissolved) if not dissolved.empty else pd.Series(dtype=float)
    if features_dir is not None:
        write_features(dissolved[["geometry"]].assign(fractal_dimension_joined=joined_values), features_dir / "fractal_dimension_joined.gpkg")
    result.update(aggregate_series(joined_values, "fractal_dimension_joined", num_quantiles))

    return result
