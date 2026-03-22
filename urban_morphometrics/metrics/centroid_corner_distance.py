"""Centroid-corner distance (CCD) metric.

Mean and standard deviation of distances from the building centroid to each
vertex. Captures both size and irregularity of the footprint. Computed twice:
- Raw buildings: prefixes `ccd_mean` and `ccd_std`
- Dissolved structures: prefixes `ccd_mean_joined` and `ccd_std_joined`

momepy.centroid_corner_distance returns a DataFrame with 'mean' and 'std' columns;
each column is aggregated independently.
"""

from pathlib import Path
import warnings

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series
from urban_morphometrics.metrics.features import write_features


def _aggregate_ccd(gdf, prefix: str, num_quantiles: int, features_dir: Path | None = None) -> dict:
    # Two known warnings are suppressed here:
    # 1. "Mean of empty slice" / "Degrees of freedom <= 0": fired by np.nanmean/nanstd
    #    when a building has no vertices that deviate >10° from 180° (momepy's corner
    #    filter removes all of them). The result is NaN, which aggregate_series drops.
    # 2. "invalid value encountered in arccos": fired when floating-point arithmetic
    #    produces a cosine slightly outside [-1, 1] for collinear vertices. The NaN
    #    result is treated as a non-corner by momepy's boolean logic, which is correct.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        warnings.filterwarnings("ignore", "Degrees of freedom <= 0", RuntimeWarning)
        warnings.filterwarnings("ignore", "invalid value encountered in arccos", RuntimeWarning)
        ccd = momepy.centroid_corner_distance(gdf)
    if features_dir is not None:
        write_features(
            gdf[["geometry"]].assign(**{f"{prefix}_mean": ccd["mean"], f"{prefix}_std": ccd["std"]}),
            features_dir / f"{prefix}.gpkg",
        )
    result = aggregate_series(ccd["mean"], f"{prefix}_mean", num_quantiles)
    result.update(aggregate_series(ccd["std"], f"{prefix}_std", num_quantiles))
    return result


@register("centroid_corner_distance")
def compute(ctx: CellContext, num_quantiles: int, features_dir: Path | None = None) -> dict:
    """Mean and std of centroid-to-vertex distances per raw building and dissolved structure."""
    b = ctx.buildings_ea
    empty = pd.Series(dtype=float)
    if b.empty:
        return {
            **aggregate_series(empty, "ccd_mean", num_quantiles),
            **aggregate_series(empty, "ccd_std", num_quantiles),
            **aggregate_series(empty, "ccd_mean_joined", num_quantiles),
            **aggregate_series(empty, "ccd_std_joined", num_quantiles),
        }

    result = _aggregate_ccd(b, "ccd", num_quantiles, features_dir)

    dissolved = ctx.dissolved_buildings_ea
    result.update(_aggregate_ccd(dissolved, "ccd_joined", num_quantiles, features_dir) if not dissolved.empty else {
        **aggregate_series(pd.Series(dtype=float), "ccd_mean_joined", num_quantiles),
        **aggregate_series(pd.Series(dtype=float), "ccd_std_joined", num_quantiles),
    })

    return result
