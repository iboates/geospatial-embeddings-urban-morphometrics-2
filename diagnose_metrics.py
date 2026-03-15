"""Diagnostic script: runs each metric individually without cache and collects output."""

import io
import logging
import sys
import traceback
import warnings
from contextlib import redirect_stderr, redirect_stdout

from srai.regionalizers import H3Regionalizer, geocode_to_region_gdf

from urban_morphometrics.main import compute_urban_morphometrics
from urban_morphometrics.metrics import REGISTRY

PBF_URL = "https://download.geofabrik.de/europe/germany/baden-wuerttemberg/karlsruhe-regbez-latest.osm.pbf"
OUTPUT_FOLDER = "/mnt/c/Users/Isaac/Downloads/urban_morphometrics"
RUN_NAME_PREFIX = "_diag"


def get_study_area():
    region_gdf = geocode_to_region_gdf("Südstadt, Karlsruhe, Germany")
    regionalizer = H3Regionalizer(resolution=9)
    regions_gdf = regionalizer.transform(region_gdf)
    print(f"Study area: {len(regions_gdf)} cells")
    return regions_gdf


def run_metric(name: str, regions_gdf):
    """Run a single metric and return (stdout, stderr, warnings_list, exception)."""
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    caught_warnings = []
    exception = None

    with redirect_stdout(captured_stdout), redirect_stderr(captured_stderr):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Also capture logging
            log_buffer = io.StringIO()
            handler = logging.StreamHandler(log_buffer)
            handler.setLevel(logging.DEBUG)
            root_logger = logging.getLogger()
            root_logger.addHandler(handler)
            try:
                compute_urban_morphometrics(
                    study_area_gdf=regions_gdf,
                    pbf_path=PBF_URL,
                    run_name=f"{RUN_NAME_PREFIX}_{name}",
                    neighbourhood_distance=200,
                    num_quantiles=4,
                    metrics=[name],
                    debug=False,
                    output_folder=OUTPUT_FOLDER,
                    use_cache=False,
                )
            except Exception as e:
                exception = e
                traceback.print_exc(file=captured_stderr)
            finally:
                root_logger.removeHandler(handler)
            caught_warnings = list(w)

    return (
        captured_stdout.getvalue(),
        captured_stderr.getvalue() + log_buffer.getvalue(),
        caught_warnings,
        exception,
    )


def main():
    regions_gdf = get_study_area()

    metric_names = list(REGISTRY.keys())
    print(f"\nMetrics to test ({len(metric_names)}): {', '.join(metric_names)}\n")
    print("=" * 70)

    results = {}
    for name in metric_names:
        print(f"\n>>> Testing metric: {name}")
        print("-" * 50)
        stdout, stderr, caught_warnings, exc = run_metric(name, regions_gdf)

        has_issue = exc is not None or caught_warnings or "WARNING" in stderr or "ERROR" in stderr
        status = "FAIL" if exc is not None else ("WARN" if has_issue else "OK")
        print(f"Status: {status}")

        if stdout.strip():
            print(f"[stdout]\n{stdout.strip()}")
        if stderr.strip():
            # Filter to just WARNING/ERROR lines plus the last 5 for context
            lines = stderr.strip().splitlines()
            warn_err = [l for l in lines if "WARNING" in l or "ERROR" in l or "Traceback" in l or "Error" in l.split(":")[-1][:20]]
            if warn_err:
                print(f"[stderr/log relevant lines]")
                for l in warn_err:
                    print(f"  {l}")
        if caught_warnings:
            print(f"[Python warnings]")
            for w in caught_warnings:
                print(f"  {w.category.__name__}: {w.message} (at {w.filename}:{w.lineno})")
        if exc is not None:
            print(f"[Exception] {type(exc).__name__}: {exc}")

        results[name] = {
            "status": status,
            "stdout": stdout,
            "stderr": stderr,
            "warnings": caught_warnings,
            "exception": exc,
        }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, r in results.items():
        print(f"  {r['status']:4s}  {name}")

    return results


if __name__ == "__main__":
    main()
