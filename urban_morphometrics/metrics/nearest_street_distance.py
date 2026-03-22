"""Nearest street distance metric.

Minimum distance (m) from each focal building's centroid to the nearest vehicle
street segment. Captures how accessible (or isolated) each building is from the
road network.

Requires neighbourhood context so buildings near the cell boundary can find
streets that lie just outside the cell.
"""

from pathlib import Path

import pandas as pd

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series
from urban_morphometrics.metrics.features import write_features


@register("nearest_street_distance")
def compute(ctx: CellContext, num_quantiles: int, features_dir: Path | None = None) -> dict:
    """Minimum distance from each focal building centroid to the nearest street (m)."""
    b = ctx.buildings_ea
    if b.empty:
        return aggregate_series(pd.Series(dtype=float), "nearest_street_distance", num_quantiles)

    streets = ctx.focal_plus_neighbourhood_vehicle_highways
    if streets.empty:
        return aggregate_series(pd.Series(dtype=float), "nearest_street_distance", num_quantiles)

    streets_union = streets.geometry.union_all()
    distances = b.centroid.distance(streets_union)
    if features_dir is not None:
        write_features(b[["geometry"]].assign(nearest_street_distance=distances), features_dir / "nearest_street_distance.gpkg")
    return aggregate_series(distances, "nearest_street_distance", num_quantiles)
