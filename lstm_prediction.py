#!/usr/bin/env python3
"""
LSTM Water Level Prediction - Multi-Basin Forecasting
Generates 15-day forecasts for all (or specified) basins using trained LSTM models and HRES data.

Usage:
    python lstm_prediction.py 20250601              # Process all basins
    python lstm_prediction.py 20250601 bahadurabaad muhuri  # Process specific basins

Input:
    - HRES forecast data: basins/{basin}/input/hres_{basin}_{date}.csv
    - Trained models: basins/{basin}/outputs/best_model_*.pt
    - Normalizer: basins/{basin}/outputs/normalizer_stats.json
    - Historical WL: updated_WL_data/WL_{basin}_daily.csv

Output:
    - Forecasts: basins/{basin}/output/forecast_{date}.csv
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

TARGET_COL = "water_level"
HINDCAST_LEN = 90
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FEATURE_COLS = [
    "tp", "t2m", "ssr", "str", "sp",
    "tp_roll3", "tp_roll7", "tp_roll14", "tp_roll30", "tp_roll60",
    "t2m_roll7", "t2m_roll14", "t2m_roll30",
    "tp_x_t2m",
    "wl_lag1", "wl_lag2", "wl_lag3", "wl_lag7", "wl_lag14",
    "wl_anom_30d",
    "doy_sin", "doy_cos", "month_sin", "month_cos",
]


# ══════════════════════════════════════════════════════════════════════════════
# NORMALIZER CLASS
# ══════════════════════════════════════════════════════════════════════════════

class Normalizer:
    def __init__(self):
        self.mean: dict = {}
        self.std:  dict = {}

    def fit(self, df: pd.DataFrame, cols: list):
        for c in cols:
            self.mean[c] = float(df[c].mean())
            self.std[c]  = float(max(df[c].std(), 1e-8))

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        for c in self.mean:
            if c in d.columns:
                d[c] = (d[c] - self.mean[c]) / self.std[c]
        return d

    def inverse_wl(self, arr) -> np.ndarray:
        return np.asarray(arr) * self.std[TARGET_COL] + self.mean[TARGET_COL]

    def load(self, path: str):
        with open(path) as f:
            d = json.load(f)
        self.mean, self.std = d["mean"], d["std"]


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build all routing-aware features."""
    d = df.copy()

    # Rolling precipitation
    for w in [3, 7, 14, 30, 60]:
        d[f"tp_roll{w}"] = d["tp"].rolling(w, min_periods=1).mean()

    # Rolling temperature
    for w in [7, 14, 30]:
        d[f"t2m_roll{w}"] = d["t2m"].rolling(w, min_periods=1).mean()

    # Interaction
    d["tp_x_t2m"] = d["tp_roll7"] * d["t2m_roll7"]

    # Autoregressive WL lags
    for lag in [1, 2, 3, 7, 14]:
        d[f"wl_lag{lag}"] = d[TARGET_COL].shift(lag)

    # WL anomaly
    wl_roll30 = d[TARGET_COL].rolling(30, min_periods=7).mean()
    d["wl_anom_30d"] = d[TARGET_COL] - wl_roll30

    # Cyclical encoding
    doy   = d.index.dayofyear.values
    month = d.index.month.values
    d["doy_sin"]   = np.sin(2 * np.pi * doy   / 365.25)
    d["doy_cos"]   = np.cos(2 * np.pi * doy   / 365.25)
    d["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    d["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    # Forward-fill any remaining NaN values (from lags on initial rows)
    d = d.fillna(method='ffill').fillna(method='bfill')

    return d


# ══════════════════════════════════════════════════════════════════════════════
# LSTM MODEL
# ══════════════════════════════════════════════════════════════════════════════

class RoutingLSTM(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 128,
                 n_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.Tanh(),
        )
        self.lstm = nn.LSTM(
            input_size  = hidden_size,
            hidden_size = hidden_size,
            num_layers  = n_layers,
            batch_first = True,
            dropout     = dropout if n_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x          = self.proj(x)
        out, _     = self.lstm(x)
        last       = self.drop(out[:, -1])
        return self.head(last).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def run_hres_forecast(hres_df, history_df, model_dir, norm, n_features,
                      hindcast_len=HINDCAST_LEN):
    """
    Generate 15-day water level forecast using HRES inputs and historical data.
    """
    # Validate required columns
    required_cols = ["tp", "t2m", "ssr", "str", "sp"]
    missing_cols = [c for c in required_cols if c not in hres_df.columns]
    if missing_cols:
        raise Exception(f"Missing columns in HRES data: {missing_cols}. Available: {list(hres_df.columns)}")
    
    # Convert units (HRES only)
    hres = hres_df.copy()
    if hres["t2m"].median() > 100:
        hres["t2m"] = hres["t2m"] - 273.15
    if hres["sp"].median() > 10000:
        hres["sp"] = hres["sp"] / 100.0

    hist = history_df.copy()

    # Load ensemble models
    models = []
    for pt in sorted(f for f in os.listdir(model_dir)
                     if f.startswith("best_model_") and f.endswith(".pt")):
        try:
            m = RoutingLSTM(n_features).to(DEVICE)
            state_dict = torch.load(
                os.path.join(model_dir, pt),
                map_location=DEVICE
            )
            m.load_state_dict(state_dict)
            m.eval()
            models.append(m)
            print(f"  ✓ Loaded {pt}")
        except Exception as e:
            print(f"  ⚠ Failed to load {pt}: {e}")
            continue

    if not models:
        raise Exception("No models loaded!")

    print(f"  Total ensemble size: {len(models)}\n")

    # Rolling inference
    running = hist.copy()
    results = []

    for step in range(len(hres)):
        lead = step + 1
        fore_date = hres.index[step]

        # Get last known water level for initialization
        last_wl = float(running[TARGET_COL].dropna().iloc[-1])

        # Append HRES day with last WL value (will be updated after prediction)
        new_row = hres.iloc[[step]][["tp", "t2m", "ssr", "str", "sp"]].copy()
        new_row[TARGET_COL] = last_wl
        running = pd.concat([running, new_row])

        # Engineer features
        full = engineer_features(running)

        # Get input window
        win_df = full[FEATURE_COLS].iloc[-hindcast_len:]
        win = win_df.values.astype(np.float32)

        if np.isnan(win).any():
            # Fallback (should rarely happen with forward-filling)
            pred_wl = last_wl
            pred_std = 0.0
            nan_count = np.isnan(win).sum()
            print(f"  Lead {lead:>2}  {fore_date.date()}  "
                  f"⚠ {nan_count} NaN values → WL={pred_wl:.3f} m")
        else:
            # Normalize
            win_norm_df = pd.DataFrame(win, columns=FEATURE_COLS)
            win_norm = norm.transform(win_norm_df)[FEATURE_COLS].values \
                           .astype(np.float32)

            x_t = torch.FloatTensor(win_norm).unsqueeze(0).to(DEVICE)
            preds_norm = []

            with torch.no_grad():
                for m in models:
                    preds_norm.append(float(m(x_t).cpu().item()))

            pred_wl = float(norm.inverse_wl(np.array([np.mean(preds_norm)]))[0])
            pred_std = float(np.std(preds_norm)) * norm.std[TARGET_COL]

            print(f"  Lead {lead:>2}  {fore_date.date()}  "
                  f"WL = {pred_wl:.3f} m  (±{pred_std:.3f} m)")

        # Update predicted water level in history
        running.iloc[-1, running.columns.get_loc(TARGET_COL)] = pred_wl

        results.append({
            "date": fore_date,
            "lead_day": lead,
            "predicted_wl_m": round(pred_wl, 4),
            "lower_m": round(pred_wl - 2 * pred_std, 4),
            "upper_m": round(pred_wl + 2 * pred_std, 4),
        })

    out_df = pd.DataFrame(results).set_index("date")
    return out_df


# ══════════════════════════════════════════════════════════════════════════════
# PROCESS BASIN
# ══════════════════════════════════════════════════════════════════════════════

def process_basin(basin_name, date_str, lstm_dir):
    """Process a single basin."""
    
    try:
        forecast_date = datetime.strptime(date_str, "%Y%m%d")
    except ValueError:
        print(f"❌ Invalid date format: {date_str}. Use YYYYMMDD")
        return False

    basin_dir = lstm_dir / "basins" / basin_name
    
    if not basin_dir.exists():
        print(f"❌ Basin directory not found: {basin_dir}")
        return False

    print(f"\n{'=' * 70}")
    print(f"  Basin: {basin_name.upper()}")
    print(f"  Date: {forecast_date.strftime('%Y-%m-%d')}")
    print(f"{'=' * 70}\n")

    # Setup paths
    input_dir = basin_dir / "input"
    output_dir = basin_dir / "output"
    model_dir = basin_dir / "outputs"
    norm_path = model_dir / "normalizer_stats.json"
    wl_data_dir = lstm_dir / "updated_WL_data"

    # Validate directories
    for d in [input_dir, model_dir, output_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Find input file
    input_files = list(input_dir.glob(f"hres_{basin_name}_{date_str}.csv"))
    if not input_files:
        input_files = list(input_dir.glob(f"*{date_str}*.csv"))

    if not input_files:
        print(f"❌ Input file not found in {input_dir}")
        print(f"   Expected: hres_{basin_name}_{date_str}.csv\n")
        return False

    hres_path = input_files[0]
    print(f"[1] Loading HRES forecast data")
    print(f"    {hres_path.name}")

    try:
        hres_df = pd.read_csv(hres_path, index_col=0, parse_dates=True)
        # Strip whitespace from column names
        hres_df.columns = hres_df.columns.str.strip()
        if len(hres_df) < 15:
            print(f"⚠️  Warning: Only {len(hres_df)} days of forecast data")
        else:
            hres_df = hres_df.iloc[:15]
        print(f"    ✓ {len(hres_df)} days loaded")
        print(f"    Columns: {', '.join(hres_df.columns)}\n")
    except Exception as e:
        print(f"❌ Failed to load HRES data: {e}\n")
        return False

    # Load historical water level
    print(f"[2] Loading historical water level data")
    
    wl_candidates = [
        wl_data_dir / f"WL_{basin_name}_daily.csv",
        wl_data_dir / f"WL_{basin_name.replace('_', '')}_daily.csv",
    ]
    
    wl_path = None
    for candidate in wl_candidates:
        if candidate.exists():
            wl_path = candidate
            break

    if not wl_path or not wl_path.exists():
        available = list(wl_data_dir.glob("*.csv"))
        print(f"❌ WL file not found")
        print(f"   Looked for: {wl_candidates[0]}")
        print(f"   Available: {[f.name for f in available]}\n")
        return False

    print(f"    {wl_path.name}")

    try:
        wl_all = pd.read_csv(wl_path)
        # Strip whitespace from column names
        wl_all.columns = wl_all.columns.str.strip()
        
        date_col = None
        for col in wl_all.columns:
            if col.lower() in ['date', 'dates']:
                date_col = col
                break
        
        if date_col is None:
            date_col = wl_all.columns[0]
        
        # Try multiple date formats
        for date_fmt in ['%m/%d/%Y', '%m/%d/%y', '%Y-%m-%d', '%d/%m/%Y']:
            try:
                wl_all[date_col] = pd.to_datetime(wl_all[date_col], format=date_fmt)
                break
            except:
                continue
        
        wl_all = wl_all.set_index(date_col)
        wl_all.index.name = 'date'
        
        if 'water_level' not in wl_all.columns:
            wl_all = wl_all.rename(columns={wl_all.columns[0]: 'water_level'})

        history_df = wl_all[['water_level']].tail(300).copy()
        history_df.columns = [TARGET_COL]
        
        # Forward-fill any NaN gaps in historical data
        history_df = history_df.fillna(method='ffill').fillna(method='bfill')
        
        last_date = history_df.index[-1]
        print(f"    ✓ {len(history_df)} days loaded (last: {last_date.date()})\n")
    except Exception as e:
        print(f"❌ Failed to load WL data: {e}\n")
        return False

    # Load normalizer
    print(f"[3] Loading model and normalizer")
    if not norm_path.exists():
        print(f"❌ Normalizer not found: {norm_path.name}")
        print(f"   Skipping basin (no trained model available)\n")
        return False
    
    try:
        norm = Normalizer()
        norm.load(norm_path)
        print(f"    ✓ Normalizer loaded")
        
        n_features = len(FEATURE_COLS)
        print(f"    Features: {n_features}\n")
    except Exception as e:
        print(f"❌ Failed to load normalizer: {e}\n")
        return False

    # Run forecast
    print(f"[4] Running forecast")
    try:
        forecast_df = run_hres_forecast(
            hres_df, history_df, model_dir, norm, n_features
        )
    except Exception as e:
        print(f"❌ Prediction failed: {e}\n")
        return False

    # Save output
    print(f"[5] Saving forecast")
    output_file = output_dir / f"forecast_{date_str}.csv"
    try:
        forecast_df.to_csv(output_file)
        print(f"    ✓ Saved: {output_file.name}\n")
    except Exception as e:
        print(f"❌ Failed to save forecast: {e}\n")
        return False

    # Summary
    print(f"{'=' * 70}")
    print(f"  FORECAST SUMMARY — {basin_name.upper()}")
    print(f"{'=' * 70}")
    print(forecast_df.to_string())
    print(f"\n  Output: {output_file}")
    print(f"{'=' * 70}\n")
    
    return True


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Multi-basin LSTM water level forecasting"
    )
    parser.add_argument("date", help="Date in YYYYMMDD format (e.g., 20250601)")
    parser.add_argument("basins", nargs="*", 
                       help="Basin names (if empty, process all basins)")
    args = parser.parse_args()

    lstm_dir = Path(__file__).parent
    basins_dir = lstm_dir / "basins"

    if not basins_dir.exists():
        print(f"❌ Basins directory not found: {basins_dir}")
        sys.exit(1)

    # Determine which basins to process
    if args.basins:
        basins_to_process = args.basins
    else:
        # Auto-discover all basins
        basins_to_process = sorted([d.name for d in basins_dir.iterdir() 
                                   if d.is_dir() and (d / "input").exists()])

    if not basins_to_process:
        print("❌ No basins found or specified")
        sys.exit(1)

    print(f"\n{'═' * 70}")
    print(f"  LSTM MULTI-BASIN WATER LEVEL FORECASTING")
    print(f"  Date: {args.date}")
    print(f"  Basins: {', '.join(basins_to_process)}")
    print(f"  Device: {DEVICE}")
    print(f"{'═' * 70}")

    # Process each basin
    results = {}
    for basin in basins_to_process:
        success = process_basin(basin, args.date, lstm_dir)
        results[basin] = "✓ Success" if success else "✗ Failed"

    # Summary
    print(f"\n{'═' * 70}")
    print(f"  OVERALL SUMMARY")
    print(f"{'═' * 70}")
    for basin, status in results.items():
        print(f"  {basin:20s} {status}")
    print(f"{'═' * 70}\n")


if __name__ == "__main__":
    main()
