"""OSM oneway tag parsing for highway GeoDataFrames.

QuackOSM produces 'highway' and 'oneway' as direct columns when loaded with
keep_all_tags=False. These utilities parse those values into a boolean oneway flag.
"""


def _default_oneway(highway_val, junction) -> bool:
    """Default oneway when tag absent: motorway and roundabout are one-way."""
    if junction and str(junction).lower() == "roundabout":
        return True
    if highway_val and str(highway_val).lower() in ("motorway", "motorway_link"):
        return True
    return False


def parse_oneway(oneway_val, highway_val, junction_val=None) -> bool:
    """Parse OSM oneway tag. Returns True if one-way only, False if bidirectional.

    OSM oneway=yes/true/1 -> one-way. oneway=no/false/0 or absent -> two-way.
    highway=motorway and junction=roundabout default to one-way.

    Args:
        oneway_val: Value of the OSM 'oneway' tag (string, float NaN, or None).
        highway_val: Value of the OSM 'highway' tag.
        junction_val: Value of the OSM 'junction' tag, if available.
    """
    if oneway_val is None or (isinstance(oneway_val, float) and oneway_val != oneway_val):
        return _default_oneway(highway_val, junction_val)
    val = str(oneway_val).strip().lower()
    if not val:
        return _default_oneway(highway_val, junction_val)
    if val in ("yes", "true", "1"):
        return True
    if val in ("no", "false", "0"):
        return False
    if val == "-1":
        return True  # one-way reverse; geometry direction unchanged
    return _default_oneway(highway_val, junction_val)


def apply_oneway(highways_gdf):
    """Add a boolean 'oneway' column to a highways GeoDataFrame.

    Overwrites any existing 'oneway' column with a parsed boolean version.

    Args:
        highways_gdf: GeoDataFrame with 'highway' column and optionally 'oneway'.

    Returns:
        A copy with a boolean 'oneway' column.
    """
    highways = highways_gdf.copy()
    oneway_vals = highways.get("oneway")
    highway_vals = highways["highway"]
    junction_vals = highways.get("junction")

    highways["oneway"] = [
        parse_oneway(
            oneway_vals.iloc[i] if oneway_vals is not None else None,
            highway_vals.iloc[i],
            junction_vals.iloc[i] if junction_vals is not None else None,
        )
        for i in range(len(highways))
    ]
    return highways
