"""
feni_muhuri_lstm.py
═══════════════════════════════════════════════════════════════════════
Feni / Muhuri River — Water Level Simulation & 15-day HRES Forecasting
Outlet gauge : Parshuram (Feni district, Bangladesh)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY THIS SCRIPT IS DIFFERENT FROM THE BRAHMAPUTRA VERSION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Brahmaputra-Jamuna                 Feni / Muhuri
  ─────────────────────────────────  ─────────────────────────────────
  Area        ~580 000 km²           ~2 770 km²  (Feni) / 1 800 km²
  Routing lag ~60 days (Tibet→BD)    3–7 days (Tripura hills→plain)
  Snowmelt    YES (major driver)     NO (fully tropical, 25 °C mean)
  Response    Slow, large inertia    FAST, flash-flood prone
  HINDCAST    90 days                30 days
  tp windows  3,7,14,30,60 days      2,3,5,7,10,14 days
  t2m windows 7,14,30 days           3,7 days (ET only, no snowmelt)
  WL lags     1,2,3,7,14 days        1,2,3,5,7 days
  WL anomaly  30-day window          7-day window (fast limb)
  HIDDEN_SIZE 128                    64
  DROPOUT     0.20                   0.25

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT "LEAD TIME" MEANS HERE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Training / evaluation (ERA5 reanalysis 1980–2025):
    The model reads a 30-day window of past features and predicts
    the water level on the NEXT day — this is a 0-day lead simulation.
    There is no forecast lead time during ERA5 training.

  HRES operational forecast:
    • Lead 1  = you predict WL for tomorrow using HRES day-1 inputs
    • Lead 2  = WL for day after tomorrow using HRES day-2 inputs
    • Lead 15 = WL 15 days from now using HRES day-15 inputs
    The predicted WL from each step is fed back as wl_lag1, wl_lag2 …
    for subsequent steps (autoregressive routing feedback).
    See run_hres_forecast() at the bottom of this file.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FEATURES (24 total)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ERA5 raw (5):
    tp, t2m, ssr, str, sp

  Rolling precipitation — fast basin routing (6):
    tp_roll2  : 2-day mean  (captures same-day + yesterday inflow)
    tp_roll3  : 3-day mean  (typical Feni time of concentration)
    tp_roll5  : 5-day mean
    tp_roll7  : 7-day mean  (weekly antecedent moisture)
    tp_roll10 : 10-day mean (medium antecedent state)
    tp_roll14 : 14-day mean (longer dry/wet spell memory)

  Rolling temperature — evapotranspiration only, NO snowmelt (2):
    t2m_roll3 : 3-day mean
    t2m_roll7 : 7-day mean

  Rain × temperature interaction (1):
    tp_x_t2m  : tp_roll3 × t2m_roll3 (warm rain → faster surface runoff)

  Autoregressive WL lags (5):
    wl_lag1 … wl_lag3  : carry channel storage / soil moisture state
    wl_lag5, wl_lag7   : weekly memory

  Rising / falling limb indicator (1):
    wl_anom_7d : WL − 7-day rolling mean
                 positive = rising limb (flood building)
                 negative = falling limb (recession)

  Cyclical temporal encoding (4):
    doy_sin, doy_cos, month_sin, month_cos
    (captures monsoon seasonality without discontinuities)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SPLIT  :  70 % train  /  30 % test  (strictly chronological)
          Val = last 10 % of train period, for early stopping only
          30-day gap between every boundary (= HINDCAST_LEN)

MODEL  :  RoutingLSTM
          Input projection  Linear(23 → 64) + Tanh
          LSTM              1 layer, hidden = 64
          Dropout           0.25
          Output head       Linear(64 → 1)
          Ensemble          3 independently seeded models

LOSS   :  MSE  (converted to RMSE in m for reporting)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUTS (written to OUT_DIR)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    best_model_0.pt … best_model_2.pt
    normalizer_stats.json
    feature_cols.json
    all_predictions.csv
    plots/
        01_training_curves.png
        02_hydrograph_training.png
        02_hydrograph_test.png
        03_scatter_training.png
        03_scatter_test.png
        04_seasonal_error.png
        05_feature_correlation.png
        06_lag_analysis.png

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Usage:
    python feni_muhuri_lstm.py

HRES inference after training:
    See run_hres_forecast() at the bottom of this file.
    Supply a 15-row HRES DataFrame with columns:
        tp [mm/day], t2m [K], ssr [J/m²], str [J/m²], sp [Pa]
    Temperature must be in Kelvin — converted internally to °C.
"""

import os
import json
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  PATHS  ← change these two lines for your machine
# ══════════════════════════════════════════════════════════════════════════════
BASIN_DIR = Path(__file__).resolve().parent
ERA5_CSV = BASIN_DIR / "ERA5_Feni_All.csv"
WL_CSV   = BASIN_DIR / "WL_SW212_daily.csv"
OUT_DIR  = BASIN_DIR / "model"

# ══════════════════════════════════════════════════════════════════════════════
#  BASIN-SPECIFIC HYPERPARAMETERS
#  All values below are tuned for the Feni/Muhuri small fast-response basin.
#  DO NOT copy these from the Brahmaputra version.
# ══════════════════════════════════════════════════════════════════════════════

HINDCAST_LEN = 30      # days of ERA5 history the LSTM reads per sample
                       # 30 days = 4× the max expected routing time (7 days)
                       # for this ~2770 km² basin

TARGET_COL   = "water_level"

# Model architecture — smaller than Brahmaputra (simpler basin dynamics)
HIDDEN_SIZE  = 64      # LSTM hidden units
N_LAYERS     = 1       # single-layer LSTM (sufficient for small basin)
DROPOUT      = 0.25    # slightly higher than Brahmaputra to prevent overfit

# Training
TRAIN_FRAC   = 0.70    # 70 % chronological train, 30 % test
VAL_FRAC     = 0.10    # 10 % of training period used for validation
BATCH_SIZE   = 256
LR           = 5e-4
MAX_EPOCHS   = 200
PATIENCE     = 25      # early-stopping patience
GRAD_CLIP    = 1.0
N_ENSEMBLE   = 3       # independently seeded; prediction = ensemble mean

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ══════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1 — DATA LOADING & UNIT CONVERSION
# ─────────────────────────────────────────────────────────────────────────────

def load_raw(era5_path: str, wl_path: str) -> pd.DataFrame:
    """
    Load ERA5 and water-level CSVs, align on a complete daily index,
    apply unit conversions, and linearly interpolate any missing WL days.

    Unit conversions applied:
        t2m : Kelvin → °C   (check: if median > 100 it is Kelvin)
        sp  : Pa     → hPa  (check: if median > 10000 it is Pa)
        tp  : already mm/day — no conversion needed
        ssr / str : J m⁻²  — kept as-is, z-scored during normalisation
    """
    era5 = pd.read_csv(era5_path, parse_dates=["date"]).set_index("date")
    wl   = pd.read_csv(wl_path,   parse_dates=["date"]).set_index("date")

    # Unit conversions (auto-detect by median value)
    if era5["t2m"].median() > 100:
        era5["t2m"] = era5["t2m"] - 273.15    # K → °C
    if era5["sp"].median() > 10000:
        era5["sp"]  = era5["sp"]  / 100.0     # Pa → hPa

    # Align on a gap-free daily index spanning the full data range
    start = min(era5.index.min(), wl.index.min())
    end   = max(era5.index.max(), wl.index.max())
    full_idx = pd.date_range(start, end, freq="D")

    era5 = era5.reindex(full_idx)
    wl   = wl.reindex(full_idx)

    df = era5.copy()
    df[TARGET_COL] = wl[TARGET_COL]
    df.index.name  = "date"

    # Interpolate missing WL (linear over time — safe for gaps ≤ ~30 days)
    n_miss = int(df[TARGET_COL].isna().sum())
    if n_miss > 0:
        df[TARGET_COL] = df[TARGET_COL].interpolate(
            method="time", limit=45)           # cap at 45-day gaps
        still = int(df[TARGET_COL].isna().sum())
        print(f"  WL: interpolated {n_miss} missing days "
              f"({still} remain after limit=45)")

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2 — FEATURE ENGINEERING  (fast-response basin)
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all 23 routing-aware features for the Feni/Muhuri basin.

    Design rationale:
    ─────────────────
    The Feni/Muhuri drains Tripura hills (steep gradient) into the
    Chittagong coastal plain.  Rain in the upper catchment reaches
    Parshuram gauge within 3–7 days.

    • Short rolling windows (2–14 days) capture this fast routing.
    • No 30-day or 60-day windows — those would add noise, not signal.
    • No long temperature rolls — basin is tropical (25 °C mean),
      no glacier or seasonal snowmelt contribution.
    • wl_anom_7d (7-day anomaly) detects rising/falling limb faster
      than the 30-day window used for the large Brahmaputra basin.

    All rolling operations look BACKWARDS only — zero future leakage.
    """
    d = df.copy()

    # ── Rolling precipitation (fast routing lags) ─────────────────────
    # 2-day : today + yesterday → same-day response for intense storms
    # 3-day : Feni typical time of concentration
    # 5,7   : antecedent moisture state from recent wet spell
    # 10,14 : longer memory for sustained monsoon events
    for w in [2, 3, 5, 7, 10, 14]:
        d[f"tp_roll{w}"] = d["tp"].rolling(w, min_periods=1).mean()

    # ── Rolling temperature (ET proxy only, NO snowmelt) ──────────────
    # Short windows: warm spell → higher ET → drier soil → less runoff
    # (and vice versa — wet cold spell → saturated soil → more runoff)
    for w in [3, 7]:
        d[f"t2m_roll{w}"] = d["t2m"].rolling(w, min_periods=1).mean()

    # ── Rain × temperature interaction ───────────────────────────────
    # Warm rain generates faster infiltration-excess runoff than cool
    # drizzle.  Product of 3-day rolling means captures this.
    d["tp_x_t2m"] = d["tp_roll3"] * d["t2m_roll3"]

    # ── Autoregressive WL lags ────────────────────────────────────────
    # These encode the integrated catchment state (soil moisture,
    # channel storage, groundwater) with r > 0.97 at lag-7 for most
    # small Bangladesh gauges.
    for lag in [1, 2, 3, 5, 7]:
        d[f"wl_lag{lag}"] = d[TARGET_COL].shift(lag)

    # ── WL 7-day anomaly (rising / falling limb) ─────────────────────
    # For a fast basin the 7-day window is more informative than 30 days.
    # Positive anomaly = river rising (flood building).
    # Negative anomaly = recession limb.
    wl_roll7 = d[TARGET_COL].rolling(7, min_periods=3).mean()
    d["wl_anom_7d"] = d[TARGET_COL] - wl_roll7

    # ── Cyclical temporal encoding ────────────────────────────────────
    # Encodes monsoon seasonality without a discontinuity at year-end.
    doy   = d.index.dayofyear.values
    month = d.index.month.values
    d["doy_sin"]   = np.sin(2 * np.pi * doy   / 365.25)
    d["doy_cos"]   = np.cos(2 * np.pi * doy   / 365.25)
    d["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    d["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    return d


# ── 24 features in the exact order the LSTM expects ──────────────────────────
FEATURE_COLS = [
    # ERA5 raw (5)
    "tp", "t2m", "ssr", "str", "sp",
    # fast routing precipitation windows (6)
    "tp_roll2", "tp_roll3", "tp_roll5", "tp_roll7", "tp_roll10", "tp_roll14",
    # temperature — ET only, no snowmelt (2)
    "t2m_roll3", "t2m_roll7",
    # warm-rain interaction (1)
    "tp_x_t2m",
    # autoregressive WL lags (5)
    "wl_lag1", "wl_lag2", "wl_lag3", "wl_lag5", "wl_lag7",
    # rising / falling limb indicator (1)
    "wl_anom_7d",
    # cyclical temporal encoding (4)
    "doy_sin", "doy_cos", "month_sin", "month_cos",
]
# Total: 24 features
# TARGET_COL (water_level) is NOT in FEATURE_COLS — it is the label.
# The wl_lag* columns are shifted copies, separate from the target.


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3 — NORMALISER  (Z-score, fitted on training data only)
# ─────────────────────────────────────────────────────────────────────────────

class Normalizer:
    """
    Z-score normalizer.
    Fitted on the training set ONLY — prevents data leakage into val/test.
    Saved to JSON so the same scaling is reused at HRES inference time.
    """

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
        """Denormalise water level predictions back to metres."""
        return np.asarray(arr) * self.std[TARGET_COL] + self.mean[TARGET_COL]

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"mean": self.mean, "std": self.std}, f, indent=2)

    def load(self, path: str):
        with open(path) as f:
            d = json.load(f)
        self.mean, self.std = d["mean"], d["std"]


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4 — CHRONOLOGICAL SPLIT  (70 / 10 / 20 with leakage gaps)
# ─────────────────────────────────────────────────────────────────────────────

def chronological_split(df: pd.DataFrame):
    """
    Pure time-ordered split — NO shuffling.

    train  : first 70 % of rows
    val    : next  ~7 % of rows (10 % of training period)
    test   : last  30 % of rows (completely held-out)

    A HINDCAST_LEN-row gap is cut between every boundary so that
    no test/val date can appear inside a training hindcast window.
    """
    n       = len(df)
    gap     = HINDCAST_LEN                        # 30 days
    n_test  = int(n * (1 - TRAIN_FRAC))           # 30 %
    n_val   = int(n * TRAIN_FRAC * VAL_FRAC)      # 10 % of 70 % ≈ 7 %

    test_start = n - n_test
    val_end    = test_start - gap
    val_start  = val_end - n_val
    train_end  = val_start - gap

    train = df.iloc[:train_end].copy()
    val   = df.iloc[val_start:val_end].copy()
    test  = df.iloc[test_start:].copy()

    print(f"\n  Train : {train.index[0].date()} → {train.index[-1].date()}"
          f"  ({len(train):,} days, {len(train)/365.25:.1f} yrs)")
    print(f"  Val   : {val.index[0].date()} → {val.index[-1].date()}"
          f"  ({len(val):,} days)")
    print(f"  Test  : {test.index[0].date()} → {test.index[-1].date()}"
          f"  ({len(test):,} days, {len(test)/365.25:.1f} yrs)")

    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 5 — SEQUENCE GENERATION  (single-step sliding window)
# ─────────────────────────────────────────────────────────────────────────────

def make_sequences(df_norm: pd.DataFrame,
                   hindcast_len: int = HINDCAST_LEN):
    """
    Sliding-window single-step sequence builder.

    For each valid position i:
        X[i] = df_norm[FEATURE_COLS].iloc[i : i+hindcast_len]   shape [T, 23]
        y[i] = df_norm[TARGET_COL].iloc[i + hindcast_len]       scalar

    Rows with ANY NaN (from lag/rolling features near split boundaries)
    are silently skipped.

    Returns
    ───────
    X     : float32  [N, hindcast_len, 23]
    y     : float32  [N]
    dates : list of pd.Timestamp  — the PREDICTED day for each sequence
    """
    n_f  = len(FEATURE_COLS)
    arr  = df_norm[FEATURE_COLS + [TARGET_COL]].values.astype(np.float32)
    idx  = df_norm.index.tolist()

    Xl, yl, dl = [], [], []
    for i in range(len(arr) - hindcast_len):
        win = arr[i: i + hindcast_len,  :n_f]
        tgt = arr[i + hindcast_len,      n_f]
        if np.isnan(win).any() or np.isnan(tgt):
            continue
        Xl.append(win); yl.append(tgt); dl.append(idx[i + hindcast_len])

    X = np.array(Xl, dtype=np.float32)
    y = np.array(yl, dtype=np.float32)
    print(f"    {len(X):,} sequences  |  X{X.shape}  y{y.shape}")
    return X, y, dl


def to_loader(X: np.ndarray,
              y: np.ndarray,
              shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.FloatTensor(X).to(DEVICE),
        torch.FloatTensor(y).to(DEVICE),
    )
    return DataLoader(ds, batch_size=BATCH_SIZE,
                      shuffle=shuffle, drop_last=False)


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 6 — MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────

class RoutingLSTM(nn.Module):
    """
    Single-step LSTM water level simulator for small fast-response basins.

    Architecture
    ────────────
    Input projection : Linear(n_features → hidden_size) + Tanh
        Projects raw features into a latent space before feeding the LSTM.
        This improves convergence vs. injecting raw heterogeneous features.

    LSTM             : 1 layer, hidden_size = 64
        Reads the projected 30-day sequence and distils it into a
        hidden state that encodes catchment memory.

    Dropout          : 0.25 applied to the last LSTM hidden state
        Prevents over-fitting on the ~10 000-sample training set.

    Output head      : Linear(hidden_size → 1)
        Maps the catchment memory state to next-day water level.
    """

    def __init__(self,
                 n_features:  int,
                 hidden_size: int   = HIDDEN_SIZE,
                 n_layers:    int   = N_LAYERS,
                 dropout:     float = DROPOUT):
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
        """
        x : [B, hindcast_len, n_features]
        returns : [B]  (predicted WL, normalised)
        """
        x          = self.proj(x)            # [B, T, H]
        out, _     = self.lstm(x)            # [B, T, H]
        last       = self.drop(out[:, -1])   # [B, H]  ← last timestep only
        return self.head(last).squeeze(-1)   # [B]

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 7 — TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(model: nn.Module,
              loader: DataLoader,
              optimizer,
              train: bool) -> float:
    """Run one epoch; return mean MSE loss."""
    model.train() if train else model.eval()
    total, n = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for Xb, yb in loader:
            pred = model(Xb)
            loss = nn.functional.mse_loss(pred, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
            total += loss.item() * len(yb)
            n     += len(yb)
    return total / n


def train_member(seed:       int,
                 tr_loader:  DataLoader,
                 vl_loader:  DataLoader,
                 n_features: int,
                 save_path:  str):
    """Train one ensemble member. Returns (history_dict, best_val_rmse)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = RoutingLSTM(n_features).to(DEVICE)
    print(f"  Trainable parameters : {model.n_params():,}")

    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", patience=8, factor=0.5,
                min_lr=1e-6)

    best_val, no_impr = float("inf"), 0
    hist = {"train_rmse": [], "val_rmse": []}

    print(f"  {'Ep':>4} | {'Train RMSE (norm)':>18} | "
          f"{'Val RMSE (norm)':>16} | {'LR':>9}")
    print(f"  {'─'*56}")

    for ep in range(1, MAX_EPOCHS + 1):
        tr_loss = run_epoch(model, tr_loader, opt, train=True)
        vl_loss = run_epoch(model, vl_loader, opt, train=False)
        tr_rmse = tr_loss ** 0.5
        vl_rmse = vl_loss ** 0.5

        hist["train_rmse"].append(tr_rmse)
        hist["val_rmse"].append(vl_rmse)
        sched.step(vl_loss)
        lr_now = opt.param_groups[0]["lr"]

        tag = ""
        if vl_loss < best_val:
            best_val, no_impr = vl_loss, 0
            torch.save(model.state_dict(), save_path)
            tag = "  ✓"
        else:
            no_impr += 1

        if ep % 10 == 0 or tag:
            print(f"  {ep:>4} | {tr_rmse:>18.6f} | {vl_rmse:>16.6f} |"
                  f" {lr_now:>9.2e}{tag}")

        if no_impr >= PATIENCE:
            print(f"\n  Early stop at epoch {ep}  "
                  f"(best val RMSE = {best_val**0.5:.5f} normalised)")
            break

    return hist, best_val ** 0.5


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 8 — METRICS
# ─────────────────────────────────────────────────────────────────────────────

def _clean(obs, sim):
    o = np.asarray(obs, dtype=float)
    s = np.asarray(sim, dtype=float)
    m = ~(np.isnan(o) | np.isnan(s))
    return o[m], s[m]

def nse(obs, sim) -> float:
    o, s = _clean(obs, sim)
    return float(1 - np.sum((o - s)**2) / np.sum((o - o.mean())**2))

def kge(obs, sim) -> float:
    o, s = _clean(obs, sim)
    r     = float(pearsonr(o, s)[0])
    alpha = s.std()  / o.std()   if o.std()   > 0 else np.nan
    beta  = s.mean() / o.mean()  if o.mean() != 0 else np.nan
    return float(1 - ((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)**0.5)

def rmse(obs, sim) -> float:
    o, s = _clean(obs, sim)
    return float(np.sqrt(np.mean((o - s)**2)))

def mae(obs, sim) -> float:
    o, s = _clean(obs, sim)
    return float(np.mean(np.abs(o - s)))

def pbias(obs, sim) -> float:
    o, s = _clean(obs, sim)
    return float(100 * (s - o).sum() / o.sum())

def r2(obs, sim) -> float:
    o, s = _clean(obs, sim)
    return float(pearsonr(o, s)[0] ** 2)

def print_metrics(obs, sim, label: str):
    print(f"\n  ── {label} ──────────────────────────────────────")
    print(f"  NSE   : {nse(obs,sim):>8.4f}"
          f"   (>0.75 good | >0.90 excellent)")
    print(f"  KGE   : {kge(obs,sim):>8.4f}"
          f"   (>0.75 good | >0.90 excellent)")
    print(f"  R²    : {r2(obs,sim):>8.4f}")
    print(f"  RMSE  : {rmse(obs,sim):>8.4f} m")
    print(f"  MAE   : {mae(obs,sim):>8.4f} m")
    print(f"  PBias : {pbias(obs,sim):>8.2f} %"
          f"   (|<10 %| = acceptable)")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 9 — PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_training(histories, plot_dir):
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["steelblue", "darkorange", "green"]
    for i, h in enumerate(histories):
        c = colors[i % 3]
        ax.plot(h["train_rmse"], lw=1,   ls="--", alpha=0.45,
                color=c, label=f"Train {i+1}")
        ax.plot(h["val_rmse"],   lw=2,   alpha=0.9,
                color=c, label=f"Val {i+1}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE (normalised space)")
    ax.set_title("Ensemble Training & Validation Loss — Feni/Muhuri")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "01_training_curves.png"), dpi=130)
    plt.close()
    print("  Saved → 01_training_curves.png")


def plot_hydrograph(dates, obs, pred, label, plot_dir):
    fig, axes = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    ax = axes[0]
    ax.plot(dates, obs,  color="black",     lw=1.2, label="Observed")
    ax.plot(dates, pred, color="steelblue", lw=0.9,
            alpha=0.85, label="Simulated (ensemble mean)")
    ax.set_ylabel("Water Level (m)")
    ax.set_title(
        f"Feni / Muhuri at Parshuram — {label}\n"
        f"NSE={nse(obs,pred):.3f}  KGE={kge(obs,pred):.3f}  "
        f"RMSE={rmse(obs,pred):.3f} m  R²={r2(obs,pred):.4f}"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    resid = np.asarray(pred) - np.asarray(obs)
    clrs  = ["tomato" if r > 0 else "steelblue" for r in resid]
    axes[1].bar(dates, resid, color=clrs, width=1, alpha=0.6)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_ylabel("Residual (m)")
    axes[1].grid(True, alpha=0.3)

    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    fname = f"02_hydrograph_{label.lower().replace(' ', '_')}.png"
    plt.savefig(os.path.join(plot_dir, fname), dpi=130)
    plt.close()
    print(f"  Saved → {fname}")


def plot_scatter(obs, pred, label, plot_dir):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(obs, pred, s=2, alpha=0.2,
               color="steelblue", rasterized=True)
    lo = min(min(obs), min(pred)) - 0.2
    hi = max(max(obs), max(pred)) + 0.2
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="1 : 1")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Observed WL (m)")
    ax.set_ylabel("Simulated WL (m)")
    ax.set_title(
        f"Scatter — {label}\n"
        f"NSE={nse(obs,pred):.3f}  R²={r2(obs,pred):.4f}  "
        f"RMSE={rmse(obs,pred):.3f} m"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f"03_scatter_{label.lower().replace(' ', '_')}.png"
    plt.savefig(os.path.join(plot_dir, fname), dpi=130)
    plt.close()
    print(f"  Saved → {fname}")


def plot_seasonal_error(dates, obs, pred, plot_dir):
    err    = np.abs(np.asarray(pred) - np.asarray(obs))
    months = np.array([d.month for d in dates])
    MNAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly = [err[months == m] for m in range(1, 13)]

    fig, ax = plt.subplots(figsize=(11, 4))
    bp = ax.boxplot(monthly, labels=MNAMES, patch_artist=True,
                    showfliers=False,
                    medianprops={"color": "black", "lw": 1.5})
    cmap = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, 12))
    for patch, c in zip(bp["boxes"], cmap):
        patch.set_facecolor(c)
    ax.set_ylabel("|Error| (m)")
    ax.set_title("Seasonal Absolute Error Distribution — Test Period"
                 "\nFeni / Muhuri at Parshuram")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "04_seasonal_error.png"), dpi=130)
    plt.close()
    print("  Saved → 04_seasonal_error.png")


def plot_feature_correlation(df_feat, plot_dir):
    """
    Bar chart of Pearson r between each engineered feature and water_level.
    For Muhuri you should see tp_roll3/5/7 dominate (fast routing),
    NOT tp_roll60 (which dominated for Brahmaputra).
    """
    corrs = {}
    for col in FEATURE_COLS:
        if col in df_feat.columns:
            tmp = df_feat[[col, TARGET_COL]].dropna()
            if len(tmp) > 100:
                corrs[col] = float(pearsonr(tmp[col], tmp[TARGET_COL])[0])

    cols_s = sorted(corrs, key=lambda c: abs(corrs[c]), reverse=True)
    vals   = [corrs[c] for c in cols_s]

    fig, ax = plt.subplots(figsize=(13, 5))
    colors = ["steelblue" if v >= 0 else "tomato" for v in vals]
    ax.barh(cols_s[::-1], vals[::-1], color=colors[::-1])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Pearson r  with  water_level")
    ax.set_title("Feature–Target Correlation — Feni/Muhuri\n"
                 "(fast basin: short rolling windows should dominate)")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "05_feature_correlation.png"), dpi=130)
    plt.close()
    print("  Saved → 05_feature_correlation.png")


def plot_lag_analysis(df_raw, plot_dir):
    """
    Plot WL autocorrelation and tp→WL cross-correlation vs lag.
    This is the diagnostic that confirms HINDCAST_LEN = 30 days is correct.
    """
    wl = df_raw[TARGET_COL].dropna()
    tp = df_raw["tp"]

    lags      = list(range(0, 32))
    wl_auto   = []    # WL autocorrelation
    tp_cross  = []    # rolling-precip vs WL cross-correlation

    for lag in lags:
        if lag == 0:
            wl_auto.append(1.0)
        else:
            r = pearsonr(wl.iloc[lag:].values, wl.iloc[:-lag].values)[0]
            wl_auto.append(r)

        # rolling sum up to this lag
        roll = tp.rolling(max(lag, 1), min_periods=1).mean()
        tmp  = pd.concat([roll, df_raw[TARGET_COL]], axis=1).dropna()
        r2_  = pearsonr(tmp.iloc[:, 0], tmp.iloc[:, 1])[0]
        tp_cross.append(r2_)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(lags, wl_auto, marker="o", color="steelblue", lw=2)
    axes[0].axvline(7,  color="tomato",    ls="--", lw=1, label="7d")
    axes[0].axvline(14, color="darkorange",ls="--", lw=1, label="14d")
    axes[0].axvline(30, color="green",     ls="--", lw=1,
                    label=f"HINDCAST={HINDCAST_LEN}d")
    axes[0].set_xlabel("Lag (days)")
    axes[0].set_ylabel("Autocorrelation r")
    axes[0].set_title("WL Autocorrelation vs Lag\n"
                      "(confirms WL lags are strong features)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(lags, tp_cross, marker="s", color="darkorange", lw=2)
    axes[1].axvline(3,  color="tomato",    ls="--", lw=1, label="3d")
    axes[1].axvline(7,  color="steelblue", ls="--", lw=1, label="7d")
    axes[1].axvline(HINDCAST_LEN, color="green", ls="--", lw=1,
                    label=f"HINDCAST={HINDCAST_LEN}d")
    axes[1].set_xlabel("Rolling window size (days)")
    axes[1].set_ylabel("Pearson r  (rolling tp vs WL)")
    axes[1].set_title("Rolling Precip → WL Correlation vs Window Size\n"
                      "(peak window = optimal routing lag)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Lag Analysis — Feni / Muhuri at Parshuram",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "06_lag_analysis.png"),
                dpi=130, bbox_inches="tight")
    plt.close()
    print("  Saved → 06_lag_analysis.png")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 10 — HRES 15-DAY ROLLING INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def run_hres_forecast(
    hres_df:      pd.DataFrame,
    history_df:   pd.DataFrame,
    model_dir:    str,
    norm_path:    str,
    feat_path:    str,
    n_features:   int,
    hindcast_len: int = HINDCAST_LEN,
) -> pd.DataFrame:
    """
    Produce a 15-day water level forecast using HRES inputs.

    Parameters
    ──────────
    hres_df : pd.DataFrame
        15-row DataFrame indexed by forecast dates (tomorrow … day+15).
        Required columns: tp [mm/day], t2m [K], ssr [J/m²],
                          str [J/m²], sp [Pa]
        Temperature MUST be in Kelvin — converted internally to °C.

    history_df : pd.DataFrame
        At least (hindcast_len + 14) days of recent ERA5 + observed WL
        ending on the day BEFORE the first HRES forecast date.
        Columns: tp, t2m [K], ssr, str, sp, water_level [m]

    model_dir : str   — folder containing best_model_0.pt … best_model_N.pt
    norm_path : str   — path to normalizer_stats.json
    feat_path : str   — path to feature_cols.json
    n_features: int   — len(FEATURE_COLS), must match trained model

    How rolling autoregressive inference works
    ───────────────────────────────────────────
    Step 1 (lead=1):
      • Append HRES row-0 (ERA5 values only, WL unknown) to history.
      • Recompute ALL engineered features on the extended history.
        Rolling windows and lags use REAL historical WL up to today.
      • Take last hindcast_len rows as input → run model → pred WL(T+1).

    Step 2 (lead=2):
      • Fill WL(T+1) = prediction from step 1.
      • Append HRES row-1 to history.
      • Recompute features — wl_lag1 now uses the step-1 prediction.
      • Run model → pred WL(T+2).

    This propagates routing through the WL lag features:
      wl_lag1 at step k = predicted WL at step k-1
      wl_anom_7d at step k reflects the predicted trend

    Returns
    ───────
    pd.DataFrame indexed by date with columns:
        lead_day, predicted_wl_m, lower_m, upper_m
        (lower/upper = ±2σ across ensemble members)
    """
    # ── load saved artifacts ──────────────────────────────────────────
    norm = Normalizer()
    norm.load(norm_path)
    with open(feat_path) as f:
        feat_cols = json.load(f)["feature_cols"]

    # ── convert HRES units (K→°C, Pa→hPa) ────────────────────────────
    hres = hres_df.copy()
    if hres["t2m"].median() > 100:
        hres["t2m"] = hres["t2m"] - 273.15
    if hres["sp"].median() > 10000:
        hres["sp"]  = hres["sp"]  / 100.0

    # ── convert history units ─────────────────────────────────────────
    hist = history_df.copy()
    if hist["t2m"].median() > 100:
        hist["t2m"] = hist["t2m"] - 273.15
    if hist["sp"].median() > 10000:
        hist["sp"]  = hist["sp"]  / 100.0

    # ── load ensemble models ──────────────────────────────────────────
    models = []
    for pt in sorted(f for f in os.listdir(model_dir) if f.endswith(".pt")):
        m = RoutingLSTM(n_features).to(DEVICE)
        m.load_state_dict(
            torch.load(os.path.join(model_dir, pt), map_location=DEVICE))
        m.eval()
        models.append(m)
    print(f"\n  Loaded {len(models)} ensemble model(s) for HRES inference")

    # ── rolling 15-step autoregressive inference ──────────────────────
    running = hist.copy()
    results = []

    for step in range(len(hres)):
        lead      = step + 1
        fore_date = hres.index[step]

        # Append this HRES day (WL unknown at forecast time)
        new_row = hres.iloc[[step]][
            ["tp", "t2m", "ssr", "str", "sp"]].copy()
        new_row[TARGET_COL] = np.nan
        running = pd.concat([running, new_row])

        # Recompute ALL features on the full (extended) running history
        full = engineer_features(running)

        # Input window = last hindcast_len rows
        win = full[feat_cols].iloc[-hindcast_len:].values.astype(np.float32)

        if np.isnan(win).any():
            # Fallback: carry forward last known WL
            pred_wl  = float(running[TARGET_COL].dropna().iloc[-1])
            pred_std = 0.0
            print(f"    Lead {lead:>2}  {fore_date.date()}  "
                  f"⚠ NaN in window — using last WL={pred_wl:.3f} m")
        else:
            # Normalise the window
            win_df   = pd.DataFrame(win, columns=feat_cols)
            win_norm = norm.transform(win_df)[feat_cols].values \
                           .astype(np.float32)

            x_t         = torch.FloatTensor(win_norm).unsqueeze(0).to(DEVICE)
            preds_norm  = []
            with torch.no_grad():
                for m in models:
                    preds_norm.append(float(m(x_t).cpu().item()))

            pred_wl  = float(
                norm.inverse_wl(np.array([np.mean(preds_norm)]))[0])
            pred_std = float(np.std(preds_norm)) * norm.std[TARGET_COL]
            print(f"    Lead {lead:>2}  {fore_date.date()}"
                  f"  WL = {pred_wl:.3f} m  (±{pred_std:.3f} m)")

        # Feed prediction back into running history for the next step
        running.iloc[-1,
                     running.columns.get_loc(TARGET_COL)] = pred_wl

        results.append({
            "date"           : fore_date,
            "lead_day"       : lead,
            "predicted_wl_m" : round(pred_wl,            4),
            "lower_m"        : round(pred_wl - 2*pred_std, 4),
            "upper_m"        : round(pred_wl + 2*pred_std, 4),
        })

    out_df = pd.DataFrame(results).set_index("date")
    print("\n  ── HRES 15-day water level forecast ──")
    print(out_df.to_string())
    return out_df


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 11 — MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t0       = time.time()
    plot_dir = os.path.join(OUT_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    print(f"\n{'═'*62}")
    print(f"  Feni / Muhuri River — Routing-Aware LSTM")
    print(f"  Outlet  : Parshuram gauge")
    print(f"  Device  : {DEVICE}")
    print(f"  Outputs : {OUT_DIR}")
    print(f"{'═'*62}")

    # ── Step 1 : load data ────────────────────────────────────────────
    print("\n[1] Loading raw data")
    df_raw = load_raw(ERA5_CSV, WL_CSV)
    print(f"  Rows       : {len(df_raw):,}")
    print(f"  Date range : {df_raw.index[0].date()} → "
          f"{df_raw.index[-1].date()}")
    print(f"  WL range   : {df_raw[TARGET_COL].min():.3f} – "
          f"{df_raw[TARGET_COL].max():.3f} m")
    print(f"  tp range   : {df_raw['tp'].min():.2f} – "
          f"{df_raw['tp'].max():.2f} mm/d")
    print(f"  t2m range  : {df_raw['t2m'].min():.1f} – "
          f"{df_raw['t2m'].max():.1f} °C  (after K→°C)")

    # ── Step 2 : feature engineering ─────────────────────────────────
    print("\n[2] Engineering routing-aware features")
    df = engineer_features(df_raw)
    print(f"  Feature count : {len(FEATURE_COLS)}  (HINDCAST_LEN={HINDCAST_LEN}d)")
    print(f"  Features      : {FEATURE_COLS}")

    # Diagnostic plots before training
    print("\n  Generating diagnostic plots...")
    plot_feature_correlation(df, plot_dir)
    plot_lag_analysis(df_raw, plot_dir)

    # ── Step 3 : chronological split ─────────────────────────────────
    print("\n[3] Chronological split  (70 % train / 30 % test)")
    train, val, test = chronological_split(df)

    # ── Step 4 : normalise  ───────────────────────────────────────────
    print("\n[4] Fitting Z-score normalizer on training data only")
    norm = Normalizer()
    norm.fit(train, FEATURE_COLS + [TARGET_COL])
    norm.save(os.path.join(OUT_DIR, "normalizer_stats.json"))
    with open(os.path.join(OUT_DIR, "feature_cols.json"), "w") as f:
        json.dump({"feature_cols": FEATURE_COLS}, f, indent=2)

    # Print first few normalizer stats
    print(f"\n  {'Feature':<16} {'Mean':>12} {'Std':>12}")
    print(f"  {'─'*42}")
    for c in list(FEATURE_COLS[:6]) + [TARGET_COL]:
        print(f"  {c:<16} {norm.mean[c]:>12.4f} {norm.std[c]:>12.4f}")
    print(f"  ... (all saved to normalizer_stats.json)")

    train_n = norm.transform(train)
    val_n   = norm.transform(val)
    test_n  = norm.transform(test)

    # ── Step 5 : sequences ────────────────────────────────────────────
    print(f"\n[5] Creating sliding-window sequences  "
          f"(hindcast = {HINDCAST_LEN} days)")
    print("  Train :"); X_tr, y_tr, d_tr = make_sequences(train_n)
    print("  Val   :"); X_v,  y_v,  d_v  = make_sequences(val_n)
    print("  Test  :"); X_te, y_te, d_te = make_sequences(test_n)

    n_feat = X_tr.shape[-1]
    print(f"\n  Input features : {n_feat}  (expected 24)")

    tr_loader = to_loader(X_tr, y_tr, shuffle=True)
    vl_loader = to_loader(X_v,  y_v,  shuffle=False)
    te_loader = to_loader(X_te, y_te, shuffle=False)

    # ── Step 6 : train ensemble ───────────────────────────────────────
    print(f"\n[6] Training {N_ENSEMBLE}-member ensemble  "
          f"(max {MAX_EPOCHS} epochs, patience={PATIENCE})")
    histories, mpaths = [], []

    for i in range(N_ENSEMBLE):
        print(f"\n── Ensemble member {i+1} / {N_ENSEMBLE} ──────────────────────")
        path = os.path.join(OUT_DIR, f"best_model_{i}.pt")
        h, bv = train_member(
            seed       = 42 + i * 100,
            tr_loader  = tr_loader,
            vl_loader  = vl_loader,
            n_features = n_feat,
            save_path  = path,
        )
        histories.append(h)
        mpaths.append(path)
        print(f"  Saved → {path}   best val RMSE (norm) = {bv:.5f}")

    plot_training(histories, plot_dir)

    # ── Step 7 : load best weights ────────────────────────────────────
    print("\n[7] Loading best weights and generating predictions")
    ensemble = []
    for path in mpaths:
        m = RoutingLSTM(n_feat).to(DEVICE)
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        m.eval()
        ensemble.append(m)

    def predict_split(loader):
        all_preds = [[] for _ in ensemble]
        all_true  = []
        with torch.no_grad():
            for Xb, yb in loader:
                for j, m in enumerate(ensemble):
                    all_preds[j].extend(m(Xb).cpu().numpy().tolist())
                all_true.extend(yb.cpu().numpy().tolist())
        pred_norm = np.mean(
            [np.array(p) for p in all_preds], axis=0)
        return (norm.inverse_wl(np.array(all_true)),
                norm.inverse_wl(pred_norm))

    obs_tr, prd_tr = predict_split(tr_loader)
    obs_v,  prd_v  = predict_split(vl_loader)
    obs_te, prd_te = predict_split(te_loader)

    # ── Step 8 : metrics ──────────────────────────────────────────────
    print("\n[8] Performance Metrics")
    print_metrics(obs_tr, prd_tr, "TRAINING SET")
    print_metrics(obs_v,  prd_v,  "VALIDATION SET")
    print_metrics(obs_te, prd_te,
                  "TEST SET  ← completely held-out, never seen in training")

    # ── Step 9 : plots ────────────────────────────────────────────────
    print("\n[9] Saving evaluation plots")
    plot_hydrograph(d_tr, obs_tr, prd_tr, "Training", plot_dir)
    plot_hydrograph(d_te, obs_te, prd_te, "Test",     plot_dir)
    plot_scatter(obs_tr, prd_tr, "Training", plot_dir)
    plot_scatter(obs_te, prd_te, "Test",     plot_dir)
    plot_seasonal_error(d_te, obs_te, prd_te, plot_dir)

    # ── Step 10 : save predictions CSV ───────────────────────────────
    print("\n[10] Saving all predictions to CSV")
    rows = []
    for split, dates, obs, pred in [
        ("train", d_tr, obs_tr, prd_tr),
        ("val",   d_v,  obs_v,  prd_v),
        ("test",  d_te, obs_te, prd_te),
    ]:
        for d, o, p in zip(dates, obs, pred):
            rows.append({
                "split"     : split,
                "date"      : d.date(),
                "obs_wl_m"  : round(float(o), 4),
                "pred_wl_m" : round(float(p), 4),
                "error_m"   : round(float(p - o), 4),
            })
    csv_path = os.path.join(OUT_DIR, "all_predictions.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  Saved → {csv_path}  ({len(rows):,} rows)")

    print(f"\n{'═'*62}")
    print(f"  Total runtime : {(time.time()-t0)/60:.1f} min")
    print(f"  All outputs   : {OUT_DIR}")
    print(f"{'═'*62}")

    # ── HRES usage reminder ───────────────────────────────────────────
    print("""
╔══════════════════════════════════════════════════════════════╗
║  HRES 15-DAY FORECAST — run after training                  ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  from feni_muhuri_lstm import run_hres_forecast              ║
║                                                              ║
║  # hres_df : 15-row DataFrame, index = forecast dates        ║
║  #  columns: tp [mm/day]  t2m [K]  ssr [J/m²]               ║
║  #           str [J/m²]   sp [Pa]                           ║
║  #  t2m must be in Kelvin → converted to °C internally       ║
║                                                              ║
║  # history_df : last 45+ days of ERA5 + observed WL          ║
║  #  columns: same as hres_df + water_level [m]               ║
║                                                              ║
║  fc = run_hres_forecast(                                     ║
║      hres_df    = hres_df,                                   ║
║      history_df = history_df,                                ║
║      model_dir  = r"...Feni_Muhuri_LSTM/model",              ║
║      norm_path  = r".../model/normalizer_stats.json",        ║
║      feat_path  = r".../model/feature_cols.json",            ║
║      n_features = 23,                                        ║
║  )                                                           ║
║  # Returns: date|lead_day|predicted_wl_m|lower_m|upper_m    ║
║  # lead 1  = tomorrow's WL                                   ║
║  # lead 15 = WL 15 days from now                             ║
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
