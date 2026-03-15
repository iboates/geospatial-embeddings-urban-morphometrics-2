"""Street profile metrics.

Cross-sectional profile of each street segment, measured by casting perpendicular
ticks and recording distances to buildings on each side. Produces three families
of columns:

- street_profile_width:      mean distance between facing buildings (m)
- street_profile_openness:   proportion of ticks with no buildings on one or both sides
- street_profile_hw_ratio:   height-to-width ratio (building height / street width)

Requires neighbourhood context so streets near the cell boundary sample buildings
across the boundary correctly. Heights are resolved for all buildings (focal and
neighbourhood) using the same OSM-tag priority logic as the volume metric:
height tag → building:levels × 3 → 6 m default.
"""

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.height import resolve_heights
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series


@register("street_profile")
def compute(ctx: CellContext, num_quantiles: int) -> dict:
    """Width, openness, and height-to-width ratio per focal street segment."""
    streets = ctx.focal_plus_neighbourhood_vehicle_highways
    focal_streets = ctx.vehicle_highways_ea

    empty_width = aggregate_series(pd.Series(dtype=float), "street_profile_width", num_quantiles)
    empty_open = aggregate_series(pd.Series(dtype=float), "street_profile_openness", num_quantiles)
    empty_hw = aggregate_series(pd.Series(dtype=float), "street_profile_hw_ratio", num_quantiles)

    if streets.empty or focal_streets.empty:
        return {**empty_width, **empty_open, **empty_hw}

    all_b = ctx.focal_plus_neighbourhood_buildings
    if all_b.empty:
        return {**empty_width, **empty_open, **empty_hw}

    all_b_with_height = resolve_heights(all_b)
    height = all_b_with_height["height"]

    profile = momepy.street_profile(streets, all_b_with_height, height=height)

    focal_profile = profile.reindex(focal_streets.index)

    result = {}
    result.update(aggregate_series(focal_profile["width"], "street_profile_width", num_quantiles))
    result.update(aggregate_series(focal_profile["openness"], "street_profile_openness", num_quantiles))
    result.update(aggregate_series(focal_profile["hw_ratio"], "street_profile_hw_ratio", num_quantiles))
    return result
