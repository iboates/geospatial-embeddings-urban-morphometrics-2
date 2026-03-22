"""Facade ratio metric.

Ratio of building area to perimeter (area / perimeter). Computed twice:
- `facade_ratio`: per raw individual building
- `facade_ratio_joined`: per dissolved structure (touching buildings merged)

Higher values indicate more compact structures; lower values indicate elongated
or highly articulated perimeters relative to their area.
"""

from pathlib import Path

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series
from urban_morphometrics.metrics._utils import dissolve_touching
from urban_morphometrics.metrics.features import write_features


@register("facade_ratio")
def compute(ctx: CellContext, num_quantiles: int, features_dir: Path | None = None) -> dict:
    """Area / perimeter per raw building and per dissolved (joined) structure."""
    b = ctx.buildings_ea
    empty = pd.Series(dtype=float)
    if b.empty:
        return {
            **aggregate_series(empty, "facade_ratio", num_quantiles),
            **aggregate_series(empty, "facade_ratio_joined", num_quantiles),
        }

    raw_values = momepy.facade_ratio(b)
    if features_dir is not None:
        write_features(b[["geometry"]].assign(facade_ratio=raw_values), features_dir / "facade_ratio.gpkg")
    result = aggregate_series(raw_values, "facade_ratio", num_quantiles)

    dissolved = dissolve_touching(b)
    joined_values = momepy.facade_ratio(dissolved)
    if features_dir is not None:
        write_features(dissolved[["geometry"]].assign(facade_ratio_joined=joined_values), features_dir / "facade_ratio_joined.gpkg")
    result.update(aggregate_series(joined_values, "facade_ratio_joined", num_quantiles))

    return result
