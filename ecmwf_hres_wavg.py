#!/usr/bin/env python3
"""
ECMWF HRES Basin Weighted Average
===================================
Computes pyscissor-weighted spatial average of daily HRES variables
for every basin found under ./basins/, and exports a CSV per basin.

Directory structure expected:
    ./basins/
        bahadurabad/
            basin_bahadurabaad.json      ← basin shapefile (GeoJSON)
            input/
                    hres_bahadurabaad_20250615.csv   ← output

Input NetCDF:
    hres_data/daily_tmp_<YYYYMMDD>.nc

Output CSV columns:
    date | d2m | t2m | ssr | str | tp | sp | forecast_date

Usage:
    python ecmwf_hres_wavg.py 20250615
"""

import sys
import warnings
from pathlib import Path

import fiona
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import shape
from pyscissor import scissor

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

BASINS_DIR   = Path("basins")
HRES_DIR     = Path("hres_data")

# Variables to extract — in desired CSV column order
VAR_COLUMNS  = ["d2m", "t2m", "ssr", "str", "tp", "sp"]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def discover_basins() -> list[dict]:
    """
    Scan ./basins/ for subfolders that contain a basin_<name>.json shapefile.
    Returns a list of dicts: {name, shapefile_path, output_root}
    """
    basins = []
    if not BASINS_DIR.exists():
        print(f"❌ Basins directory not found: {BASINS_DIR.absolute()}")
        return basins

    for folder in sorted(BASINS_DIR.iterdir()):
        if not folder.is_dir():
            continue
        name = folder.name
        shp = folder / f"basin_{name}.json"
        if not shp.exists():
            print(f"⚠️  [{name}] shapefile not found ({shp}), skipping.")
            continue
        basins.append({
            "name"        : name,
            "shapefile"   : shp,
            "input_root"  : folder / "input",   # ./basins/{name}/input/
        })

    return basins


def load_basin_shape(shapefile: Path):
    """Load the first feature from a GeoJSON/shapefile as a Shapely geometry."""
    with fiona.open(str(shapefile)) as sf:
        feature = sf[0]
    return shape(feature["geometry"])


def build_weight_grid(shapely_geom, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    Build pyscissor weight grid for a given basin geometry and lat/lon arrays.
    Returns a 2-D masked array (lat × lon).
    """
    pys = scissor(shapely_geom, lats, lons)
    return pys.get_masked_weight()


def weighted_avg(data_2d: np.ndarray, weights: np.ndarray) -> float:
    """
    Compute weighted average of a 2-D spatial field using a masked weight grid.
    NaN-safe: ignores NaN cells in the data.
    """
    # Combine the pyscissor mask with any NaN mask in the data
    data_flat    = data_2d.flatten().astype(float)
    weights_flat = np.ma.filled(weights, 0.0).flatten()

    nan_mask          = np.isnan(data_flat)
    weights_flat[nan_mask] = 0.0

    total_weight = weights_flat.sum()
    if total_weight == 0:
        return np.nan

    return float(np.sum(data_flat * weights_flat) / total_weight)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(date_str: str):
    print("  ECMWF HRES Basin Weighted Average")
    print(f"  Forecast date : {date_str}")

    # ── Locate input NC file ─────────────────────────────────────────────────
    nc_path = HRES_DIR / f"daily_{date_str}.nc"
    if not nc_path.exists():
        print(f"❌ Input file not found: {nc_path.absolute()}")
        sys.exit(1)

    print(f"  Input NC      : {nc_path}")

    # ── Load dataset ─────────────────────────────────────────────────────────
    print("\n⏳ Loading NetCDF dataset...")
    ds = xr.open_dataset(nc_path)

    lats = ds["latitude"].values   # 1-D array
    lons = ds["longitude"].values  # 1-D array
    times = pd.DatetimeIndex(ds["time"].values)

    print(f"   Timesteps : {len(times)}  ({times[0].date()} → {times[-1].date()})")
    print(f"   Grid      : {len(lats)} lat × {len(lons)} lon")
    print(f"   Variables : {list(ds.data_vars)}")

    # Check which requested variables are actually present
    available_vars = [v for v in VAR_COLUMNS if v in ds.data_vars]
    missing_vars   = [v for v in VAR_COLUMNS if v not in ds.data_vars]
    if missing_vars:
        print(f"⚠️  Variables not in NC (will be NaN in CSV): {missing_vars}")

    # Pre-load all variable arrays into memory: dict of var → np.ndarray (time, lat, lon)
    print("   Pre-loading variable arrays...")
    var_arrays = {v: ds[v].values for v in available_vars}
    ds.close()

    # ── Discover basins ──────────────────────────────────────────────────────
    basins = discover_basins()
    if not basins:
        print("❌ No valid basins found. Exiting.")
        sys.exit(1)

    print(f"\n🗺️  Found {len(basins)} basin(s): {[b['name'] for b in basins]}")

    # ── Process each basin ───────────────────────────────────────────────────
    for basin in basins:
        name = basin["name"]
        print(f"\n{'─' * 60}")
        print(f"📍 Basin: {name}")
        print(f"   Shapefile: {basin['shapefile']}")

        # Build weight grid (computed once per basin)
        try:
            geom        = load_basin_shape(basin["shapefile"])
            weight_grid = build_weight_grid(geom, lats, lons)
            print(f"   Weight grid: {weight_grid.shape}  "
                  f"(non-zero cells: {np.count_nonzero(np.ma.filled(weight_grid, 0))})")
        except Exception as e:
            print(f"   ❌ Failed to build weight grid: {e}")
            continue

        # Build one row per timestep
        rows = []
        for t_idx, ts in enumerate(times):
            row = {
                "date"          : ts.strftime("%Y-%m-%d"),
                "forecast_date" : date_str,
            }

            for var in VAR_COLUMNS:
                if var in var_arrays:
                    field = var_arrays[var][t_idx]          # (lat, lon)
                    row[var] = weighted_avg(field, weight_grid)
                else:
                    row[var] = np.nan

            rows.append(row)

        # Assemble DataFrame with desired column order
        df = pd.DataFrame(rows, columns=["date"] + VAR_COLUMNS)

        # ── Save output ──────────────────────────────────────────────────────
        out_dir  = basin["input_root"]
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"hres_{name}_{date_str}.csv"
        df.to_csv(out_file, index=False, float_format="%.6f")
        print(f"   ✅ Saved → {out_file}")
    print("✅ All basins processed.")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ecmwf_hres_wavg.py <YYYYMMDD>")
        print("Example: python ecmwf_hres_wavg.py 20250615")
        sys.exit(1)

    date_arg = sys.argv[1]
    if len(date_arg) != 8 or not date_arg.isdigit():
        print(f"❌ Invalid date format: '{date_arg}'. Expected YYYYMMDD.")
        sys.exit(1)

    main(date_arg)
