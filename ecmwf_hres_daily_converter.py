#!/usr/bin/env python3
"""
ECMWF HRES Sub-daily to Daily Converter
========================================
Converts a merged HRES NetCDF file (3-hourly + 6-hourly timesteps)
into a daily NetCDF file.

Variable handling:
  Instantaneous (t2m, d2m, sp):
      → Daily mean of all available timesteps (simple mean regardless of 3h vs 6h period)

  Accumulated (tp, ssr, str):
      → De-accumulate (diff between consecutive steps; ECMWF accumulates from forecast start)
      → Drop t=0h (accumulation start, value is ~0)
      → Each de-accumulated interval is assigned to the DATE of its ending timestamp,
        EXCEPT intervals ending at 00:00 UTC which belong to the PREVIOUS calendar day
        (because 21:00→00:00 UTC interval "completes" the prior day)
      → Daily total = sum of all intervals assigned to that date



Usage Example:
    python ecmwf_hres_daily_converter.py 20250615
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

INSTANT_VARS = {"t2m", "d2m", "sp"}       # daily mean
ACCUM_VARS   = {"tp", "ssr", "str"}       # de-accumulate → daily sum


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def assign_accum_date(timestamps: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    For each de-accumulated interval (whose label is its ENDING timestamp),
    return the calendar date it belongs to.

    Rule:
        - If ending time is 00:00 UTC → belongs to the PREVIOUS calendar day
        - Otherwise              → belongs to its own calendar day
    """
    dates = []
    for ts in timestamps:
        if ts.hour == 0 and ts.minute == 0:
            dates.append((ts - pd.Timedelta(days=1)).date())
        else:
            dates.append(ts.date())
    return pd.DatetimeIndex(dates)


def deaccumulate(da: xr.DataArray) -> xr.DataArray:
    """
    De-accumulate an ECMWF accumulated variable.

    ECMWF accumulates from the forecast start (t=0h). We:
      1. Compute forward difference along time: val(T) - val(T-1)
      2. Drop the first timestep (t=0h) which has no prior step to diff against
         (its value represents total accumulation = 0 at forecast start).

    Returns a DataArray of interval values, still labelled by their ending timestamp.
    """
    # diff along time axis: result[i] = data[i] - data[i-1], first element is NaN
    diffed = da.diff(dim="time")
    # Drop the first original timestep (t=0h) — diff already removed it implicitly
    # (diff output has len = original_len - 1, aligned to times[1:])
    return diffed


def process_instantaneous(da: xr.DataArray) -> xr.DataArray:
    """Daily mean from all available sub-daily timesteps."""
    times = pd.DatetimeIndex(da.time.values)
    dates = times.normalize()  # floor to midnight UTC

    daily_slices = []
    unique_dates = sorted(set(dates))

    for d in unique_dates:
        mask = dates == d
        subset = da.isel(time=mask)
        daily_mean = subset.mean(dim="time", keep_attrs=True)
        daily_slices.append(daily_mean.expand_dims(time=[pd.Timestamp(d)]))

    return xr.concat(daily_slices, dim="time")


def process_accumulated(da: xr.DataArray) -> xr.DataArray:
    """De-accumulate, assign intervals to correct calendar day, then daily sum."""
    # Step 1: de-accumulate
    da_diff = deaccumulate(da)  # shape: (n_times - 1, lat, lon)

    # Step 2: assign each interval to its correct calendar day
    interval_times = pd.DatetimeIndex(da_diff.time.values)
    assigned_dates = assign_accum_date(interval_times)

    # Step 3: daily sum
    daily_slices = []
    unique_dates = sorted(set(assigned_dates))

    for d in unique_dates:
        mask = assigned_dates == d
        subset = da_diff.isel(time=mask)
        daily_sum = subset.sum(dim="time", keep_attrs=True)
        daily_slices.append(daily_sum.expand_dims(time=[pd.Timestamp(d)]))

    return xr.concat(daily_slices, dim="time")


# ──────────────────────────────────────────────────────────────────────────────
# Main converter
# ──────────────────────────────────────────────────────────────────────────────

def convert(input_path: Path, output_path: Path):
    print("  ECMWF HRES Daily Converter")
    print(f"📂 Input : {input_path}")
    print(f"📂 Output: {output_path}")

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\n⏳ Loading dataset...")
    ds = xr.open_dataset(input_path)

    times = pd.DatetimeIndex(ds.time.values)
    print(f"   Timesteps  : {len(times)}")
    print(f"   Date range : {times[0]}  →  {times[-1]}")
    print(f"   Variables  : {list(ds.data_vars)}")
    print(f"   Grid       : {dict(ds.dims)}")

    # ── Detect which variables are present ───────────────────────────────────
    present_instant = [v for v in ds.data_vars if v in INSTANT_VARS]
    present_accum   = [v for v in ds.data_vars if v in ACCUM_VARS]
    unknown         = [v for v in ds.data_vars if v not in INSTANT_VARS | ACCUM_VARS]

    if unknown:
        print(f"\n⚠️  Unknown variables (will be skipped): {unknown}")

    # ── Process each variable ─────────────────────────────────────────────────
    daily_vars = {}

    for var in present_instant:
        print(f"\n🔄 [{var}] instantaneous → daily mean ...")
        daily = process_instantaneous(ds[var])
        daily.attrs.update(ds[var].attrs)
        daily.attrs["cell_methods"] = "time: mean"
        daily_vars[var] = daily
        print(f"   ✅ {len(daily.time)} daily values")

    for var in present_accum:
        print(f"\n🔄 [{var}] accumulated → de-accumulate → daily sum ...")
        daily = process_accumulated(ds[var])
        daily.attrs.update(ds[var].attrs)
        daily.attrs["cell_methods"] = "time: sum"
        daily_vars[var] = daily
        print(f"   ✅ {len(daily.time)} daily values")

    ds.close()

    # ── Align time axes (all vars must share the same daily time axis) ────────
    print("\n🔗 Aligning time axes across variables...")

    # Find common dates across all processed variables
    all_dates = [set(pd.DatetimeIndex(v.time.values).normalize()) for v in daily_vars.values()]
    common_dates = sorted(set.intersection(*all_dates))
    common_times = pd.DatetimeIndex(common_dates)

    aligned = {}
    for var, da in daily_vars.items():
        da_times = pd.DatetimeIndex(da.time.values).normalize()
        mask = [t in common_dates for t in da_times]
        aligned[var] = da.isel(time=mask)

    n_days = len(common_times)
    print(f"   Common daily timesteps: {n_days}")
    print(f"   Date range: {common_times[0].date()}  →  {common_times[-1].date()}")

    # ── Build output dataset ──────────────────────────────────────────────────
    print("\n📦 Building output dataset...")
    ds_out = xr.Dataset(aligned)

    # Restore coordinate attributes
    ds_out["latitude"].attrs  = {"units": "degrees_north", "standard_name": "latitude"}
    ds_out["longitude"].attrs = {"units": "degrees_east",  "standard_name": "longitude"}
    ds_out["time"].attrs      = {
        "standard_name": "time",
        "long_name"    : "time",
        "axis"         : "T",
    }

    ds_out.attrs = {
        "title"      : "ECMWF HRES Daily Aggregated Data",
        "institution": "European Centre for Medium-Range Weather Forecasts",
        "Conventions": "CF-1.7",
    }

    # ── Write output ──────────────────────────────────────────────────────────
    print(f"\n💾 Writing output to {output_path} ...")
    encoding = {var: {"zlib": True, "complevel": 4} for var in ds_out.data_vars}
    ds_out.to_netcdf(output_path, format="NETCDF4", unlimited_dims=["time"], encoding=encoding)

    size_mb = output_path.stat().st_size / 1024**2
    print(f"✅ Done! Output file: {output_path}  ({size_mb:.1f} MB)")
    print(f"   Variables : {list(ds_out.data_vars)}")
    print(f"   Timesteps : {n_days} daily steps")
    print("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ecmwf_hres_daily_converter.py <YYYYMMDD> [output.nc]")
        print("Example: python ecmwf_hres_daily_converter.py 20250615")
        sys.exit(1)

    date_arg = sys.argv[1]
    
    # Construct input path from date
    hres_data_dir = Path(__file__).parent / "hres_data"
    input_path = hres_data_dir / f"tmp_{date_arg}.nc"
    
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        sys.exit(1)

    # Determine output path
    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
    else:
        output_path = hres_data_dir / f"daily_{date_arg}.nc"

    print(f"📂 Input:  {input_path}")
    print(f"📂 Output: {output_path}\n")
    
    convert(input_path, output_path)
