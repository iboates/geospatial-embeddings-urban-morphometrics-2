# PROBLEMS.md — Metric Diagnostic Report

Each metric was run in isolation on a 35-cell H3 grid (resolution 9) over
Südstadt, Karlsruhe, with `use_cache=False`, `neighbourhood_distance=200`,
and `num_quantiles=4`.

## Summary

| Metric                     | Status | Issue |
|----------------------------|--------|-------|
| courtyard_area             | OK     | — |
| floor_area                 | OK     | — |
| longest_axis_length        | OK     | — |
| perimeter_wall             | OK     | — |
| volume                     | OK     | — |
| centroid_corner_distance   | WARN   | #1, #2 |
| circular_compactness       | OK     | — |
| compactness_weighted_axis  | OK     | — |
| convexity                  | OK     | — |
| corners                    | WARN   | #2 |
| courtyard_index            | OK     | — |
| elongation                 | OK     | — |
| equivalent_rectangular_index | OK   | — |
| facade_ratio               | OK     | — |
| form_factor                | OK     | — |
| fractal_dimension          | OK     | — |
| rectangularity             | OK     | — |
| shape_index                | OK     | — |
| square_compactness         | OK     | — |
| squareness                 | WARN   | #2 |

---

## Problem #1 — `centroid_corner_distance`: Empty-array statistics for buildings with no detected corners

**Metrics affected:** `centroid_corner_distance`

**Warning messages (454 occurrences each):**
```
RuntimeWarning: Mean of empty slice
  → numpy/lib/_nanfunctions_impl.py:1997
RuntimeWarning: Degrees of freedom <= 0 for slice.
  → numpy/lib/_nanfunctions_impl.py:1997
```
Both triggered from `momepy/functional/_shape.py:726`:
```python
return Series({"mean": np.nanmean(dists), "std": np.nanstd(dists)})
```

**Root cause:**

`momepy.centroid_corner_distance` filters building vertices to only those where
the interior angle deviates from 180° by more than `eps=10` (the default). For
each building, it calls `_true_angles_mask(pts, eps=10)` to find "true corners",
then computes distances from the centroid to only those corners.

If a building's exterior ring has no vertex deviating by more than 10° from
180° — which happens with degenerate OSM polygons such as triangles with very
shallow angles, nearly-collinear rings, or tiny slivers — then `dists` is an
empty array. `np.nanmean([])` and `np.nanstd([])` then fire their respective
warnings and return `NaN`.

With 454 occurrences across 35 cells (≈ 13 buildings per cell on average), this
is a common occurrence: a significant fraction of OSM building footprints
contain segments that look like corners but whose angles are all within 10° of
180°.

**Impact on results:**

The warnings are noisy but the output is correct. The `NaN` values for
individual buildings propagate into `_aggregate_ccd()`, which calls
`aggregate_series()`. That function calls `series.dropna()` before computing
statistics, so buildings without detectable corners are silently excluded from
the cell-level summary. The cell-level statistics reflect only buildings that
have at least one detected corner.

However, cells where **all** buildings have no detectable corners produce
all-NaN output columns, which is the correct fallback behaviour.

**Fix:**

Suppress the two known `RuntimeWarning` messages inside the
`centroid_corner_distance.py` metric wrapper, since the warnings are entirely
expected and the downstream `dropna()` handling is correct:

```python
import warnings

@register("centroid_corner_distance")
def compute(ctx: CellContext, num_quantiles: int) -> dict:
    b = ctx.buildings_ea
    ...
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        warnings.filterwarnings("ignore", "Degrees of freedom <= 0", RuntimeWarning)
        result = _aggregate_ccd(b, "ccd", num_quantiles)
        ...
    return result
```

---

## Problem #2 — `corners` / `squareness` / `centroid_corner_distance`: `invalid value encountered in arccos`

**Metrics affected:** `corners`, `squareness`, `centroid_corner_distance`

**Warning message:**
- `corners`: 8 occurrences
- `squareness`: 8 occurrences
- `centroid_corner_distance`: 17 occurrences (additional, on top of #1)

```
RuntimeWarning: invalid value encountered in arccos
  → momepy/functional/_shape.py:879
```

The line in question:
```python
cosine_angle = np.sum(ba * bc, axis=1) / (
    np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1)
)
angles = np.arccos(cosine_angle)   # ← line 879
```

**Root cause:**

The cosine of an angle is computed as a dot product divided by the product of
norms. Mathematically the result is in `[-1, 1]`, but floating-point arithmetic
can produce values very slightly outside this range (e.g., `1.0000000000000002`)
when the two edge vectors are nearly parallel (collinear vertices). `np.arccos`
is undefined outside `[-1, 1]` and returns `NaN` with this warning.

**Impact on results:**

The NaN propagates into `degrees = np.degrees(angles)`. A NaN in `degrees`
causes the comparisons `degrees <= 180 - eps` and `degrees >= 180 + eps` to
both return `False` (NaN comparisons are always `False` in NumPy). This means
the offending vertex is treated as **not a corner**, which is the correct
behaviour: the problematic cosines are exactly those where the angle should be
180° (perfectly collinear), so excluding the vertex as a non-corner is right.

The downstream `corners` count, `squareness` mean, and `centroid_corner_distance`
distances are therefore **not affected** by the NaN. The warning is purely
cosmetic.

**Fix:**

This is a bug inside momepy's `_true_angles_mask` helper (not our code). The
correct fix upstream would be:
```python
cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
angles = np.arccos(cosine_angle)
```

Until it is fixed upstream, suppress the warning in each affected metric wrapper:

```python
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", "invalid value encountered in arccos", RuntimeWarning
    )
    result = aggregate_series(momepy.corners(b), "corners", num_quantiles)
```

Apply the same suppression to `squareness.py` and inside `_aggregate_ccd` in
`centroid_corner_distance.py`.
