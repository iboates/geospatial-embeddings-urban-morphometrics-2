"""Cell alignment metric.

Deviation between each building's orientation and the orientation of its
morphological tessellation cell. Low values mean buildings are aligned with
their local Voronoi partition (typical of grid-planned areas); high values
indicate buildings rotated relative to the urban fabric around them.

Requires neighbourhood context so tessellation cells at the cell boundary are
correctly computed.
"""

from pathlib import Path

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series
from urban_morphometrics.metrics.features import write_features


@register("cell_alignment")
def compute(ctx: CellContext, num_quantiles: int, features_dir: Path | None = None) -> dict:
    """Deviation between building orientation and its tessellation cell orientation (°)."""
    b = ctx.buildings_ea
    if b.empty:
        return aggregate_series(pd.Series(dtype=float), "cell_alignment", num_quantiles)

    tess = ctx.tessellation
    if tess is None or tess.empty:
        return aggregate_series(pd.Series(dtype=float), "cell_alignment", num_quantiles)

    all_b = ctx.focal_plus_neighbourhood_buildings
    building_orient = momepy.orientation(all_b)
    tess_orient = momepy.orientation(tess)

    # cell_alignment does (left - right).abs(); pandas aligns by index automatically,
    # so missing tessellation entries produce NaN and are excluded from aggregation.
    values = momepy.cell_alignment(building_orient, tess_orient)
    focal_values = values.reindex(b.index)
    if features_dir is not None:
        write_features(b[["geometry"]].assign(cell_alignment=focal_values), features_dir / "cell_alignment.gpkg")
    return aggregate_series(focal_values, "cell_alignment", num_quantiles)
