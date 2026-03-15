"""Building height resolution utilities.

QuackOSM produces a 'height' column and a 'building:levels' column (colon preserved)
when loaded with keep_all_tags=False. Heights are resolved in priority order:
  1. Parsed 'height' column (metres, string-cleaned)
  2. 'building:levels' * 3 m/storey
  3. Default of 6 m (≈ 2 storeys)
"""

import numpy as np
import pandas as pd
import geopandas as gpd


_LEVELS_COL = "building:levels"
_METRES_PER_STOREY = 3.0
_DEFAULT_HEIGHT_M = 6.0


def _parse_height_value(val) -> float | None:
    """Parse an OSM height string to float metres.

    OSM height values are often strings like '12', '12.5', '12 m', '40 ft'.
    Returns None if the value cannot be parsed.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip().lower()
    if not s:
        return None
    # Strip common unit suffixes
    if s.endswith("ft") or s.endswith("'"):
        try:
            return float(s.rstrip("ft'").strip()) * 0.3048
        except ValueError:
            return None
    # Strip 'm' suffix if present
    s = s.rstrip("m").strip()
    try:
        return float(s)
    except ValueError:
        return None


def _parse_levels_value(val) -> float | None:
    """Parse an OSM building:levels string to float."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return float(str(val).strip())
    except ValueError:
        return None


def resolve_heights(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add a resolved numeric 'height' column to a buildings GeoDataFrame.

    Mutates a copy of the input. Resolution priority:
      1. Existing 'height' column (parsed to float metres)
      2. 'building:levels' column * 3 m/storey
      3. Default 6 m

    Args:
        buildings: GeoDataFrame of building polygons, as returned by QuackOSM.

    Returns:
        A copy with a numeric 'height' column guaranteed to have no NaN values.
    """
    buildings = buildings.copy()

    if "height" in buildings.columns:
        buildings["height"] = buildings["height"].apply(_parse_height_value)
    else:
        buildings["height"] = np.nan

    if _LEVELS_COL in buildings.columns:
        levels = buildings[_LEVELS_COL].apply(_parse_levels_value)
        missing = buildings["height"].isna()
        has_levels = levels.notna()
        buildings.loc[missing & has_levels, "height"] = (
            levels[missing & has_levels] * _METRES_PER_STOREY
        )

    buildings["height"] = buildings["height"].fillna(_DEFAULT_HEIGHT_M)
    return buildings
