"""
jamuna_lstm.py  —  Routing-aware LSTM
Brahmaputra-Jamuna Water Level Simulation & Forecasting

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT "LEAD TIME" MEANS IN THIS SCRIPT
──────────────────────────────────────
This script trains on ERA5 reanalysis (1980–2025), where every input
variable is already "known" for every day.  There is NO forecast lead
time in training — the model is a SIMULATION model, not a forecast model.

It predicts WL for day T using features computed from ERA5 data of days
(T-90) … (T-1).  That is a 0-day lead time simulation.

When you later plug in HRES 15-day forecasts:
  - Day T+1  : you use HRES day-1 values → this becomes 1-day lead time
  - Day T+15 : you use HRES day-15 values → this becomes 15-day lead time

See run_hres_forecast() at the bottom for exactly how to do that.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ROUTING INSIGHT & FEATURE DESIGN
──────────────────────────────────
The Brahmaputra basin is ~580 000 km².  Rain falling in the upper
reaches takes days to weeks to arrive at the gauge.  Simple daily ERA5
values therefore have limited predictive power on their own.

Features are designed to encode this "memory":

  1. Rolling precip sums (3, 7, 14, 30, 60 days)
     → captures the cumulative upstream inflow that is still routing
       through the channel network.  60-day window matches the known
       ~2-month travel time from the Tibetan Plateau headwaters.

  2. Rolling temperature means (7, 14, 30 days)
     → captures snowmelt contribution: warm spell → delayed melt runoff.

  3. Autoregressive WL lags (1, 2, 3, 7, 14 days)
     → WL at lag-1 has r ≈ 0.999 with today's WL.  These carry the
       integrated catchment state (soil moisture, channel storage,
       groundwater) far better than any met variable alone.

  4. WL 30-day anomaly
     → is the river rising (+) or falling (−)?  Critical for flood peak
       timing.

  5. Rain × temperature interaction
     → warm rainfall runoff is much faster than cold-season drizzle.

  6. Cyclical temporal encoding (doy, month)
     → tells the model where in the monsoon cycle we are.

Total: 24 features fed into the LSTM.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MODEL: Single-step LSTM (not encoder-decoder)
─────────────────────────────────────────────
An encoder-decoder would need to auto-regressively feed predicted WL
into steps 2…15 during training, compounding errors.  Single-step is
more stable.  At HRES inference we simply roll the window 15 times.

Architecture:
  Input projection  : Linear(24 → 128) + Tanh
  LSTM              : 1 layer, hidden=128
  Dropout           : 0.2
  Output head       : Linear(128 → 1)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SPLIT: 70% train / 30% test  (pure chronological, no shuffling)
  Val set = last 10% of training period, used for early stopping only.
  A 90-day gap separates every split to prevent hindcast leakage.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Usage:
    python jamuna_lstm_v2.py

Outputs (written to OUT_DIR):
    best_model_0.pt … best_model_2.pt   ← 3 ensemble weights
    normalizer_stats.json               ← mean/std per feature
    feature_cols.json                   ← ordered feature list
    all_predictions.csv                 ← train + val + test predictions
    plots/
        01_training_curves.png
        02_hydrograph_training.png
        02_hydrograph_test.png
        03_scatter_training.png
        03_scatter_test.png
        04_seasonal_error.png
        05_feature_correlation.png
"""

import os
import json
import time
import warnings
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
#  PATHS  ← edit if your paths differ
# ══════════════════════════════════════════════════════════════════════════════
ERA5_CSV = r"/mnt/d/LSTM/bahadurabaad/ERA5_Jamuna_All.csv"
WL_CSV   = r"/mnt/d/LSTM/bahadurabaad/observed_wl.csv"
OUT_DIR  = r"/mnt/d/LSTM/bahadurabaad/outputs"

# ══════════════════════════════════════════════════════════════════════════════
#  HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
HINDCAST_LEN = 90    # days of history the LSTM reads per sample
                     # 90 days captures the ~60d Brahmaputra routing lag
                     # plus 30d buffer for rising/falling limb context

TARGET_COL   = "water_level"

# Model
HIDDEN_SIZE  = 128   # LSTM hidden units
N_LAYERS     = 1
DROPOUT      = 0.2

# Training
TRAIN_FRAC   = 0.70  # 70 % train, 30 % test  (chronological)
VAL_FRAC     = 0.10  # 10 % of training period used as validation
BATCH_SIZE   = 256
LR           = 5e-4
MAX_EPOCHS   = 200
PATIENCE     = 25    # early-stopping patience (epochs with no val improvement)
GRAD_CLIP    = 1.0
N_ENSEMBLE   = 3     # independently seeded models; prediction = their mean

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ══════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
#  1.  DATA LOADING & UNIT CONVERSION
# ─────────────────────────────────────────────────────────────────────────────

def load_raw(era5_path: str, wl_path: str) -> pd.DataFrame:
    """
    Load ERA5 and water-level CSVs, align on a complete daily index,
    apply unit conversions, and linearly interpolate the known missing
    WL days (305 days scattered across 2010-2014, max gap = 31 days).
    """
    era5 = pd.read_csv(era5_path, parse_dates=["date"]).set_index("date")
    wl   = pd.read_csv(wl_path,   parse_dates=["date"]).set_index("date")

    # Unit conversions
    # tp  → already mm/day in your file (median ~3, max ~49)
    # t2m → Kelvin (median ~272), convert to °C for feature engineering
    # sp  → Pa (median ~70000), convert to hPa
    # ssr, str → J m⁻², keep as-is (will be z-scored)
    if era5["t2m"].median() > 100:
        era5["t2m"] = era5["t2m"] - 273.15
    if era5["sp"].median() > 10000:
        era5["sp"]  = era5["sp"]  / 100.0

    # Reindex to a gapless daily calendar
    full_idx = pd.date_range("1980-01-01", "2025-12-31", freq="D")
    era5 = era5.reindex(full_idx)
    wl   = wl.reindex(full_idx)

    df = era5.copy()
    df[TARGET_COL] = wl[TARGET_COL]
    df.index.name  = "date"

    # Interpolate missing WL days (max gap 31 days → safe linear interp)
    n_miss = int(df[TARGET_COL].isna().sum())
    if n_miss:
        df[TARGET_COL] = df[TARGET_COL].interpolate(method="time", limit=62)
        still = int(df[TARGET_COL].isna().sum())
        print(f"  WL: interpolated {n_miss} missing days "
              f"({still} remain after limit=62)")

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  2.  FEATURE ENGINEERING  (routing-aware)
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all routing-aware features.
    All rolling windows look BACKWARDS only — no future leakage.
    """
    d = df.copy()

    # Rolling precipitation — captures upstream inflow routing
    for w in [3, 7, 14, 30, 60]:
        d[f"tp_roll{w}"] = d["tp"].rolling(w, min_periods=1).mean()

    # Rolling temperature — captures snowmelt contribution
    for w in [7, 14, 30]:
        d[f"t2m_roll{w}"] = d["t2m"].rolling(w, min_periods=1).mean()

    # Rain × temperature interaction (warm rain → faster runoff)
    d["tp_x_t2m"] = d["tp_roll7"] * d["t2m_roll7"]

    # Autoregressive WL lags — encode integrated catchment state
    for lag in [1, 2, 3, 7, 14]:
        d[f"wl_lag{lag}"] = d[TARGET_COL].shift(lag)

    # WL 30-day anomaly — rising (+) or falling (−) limb
    wl_roll30 = d[TARGET_COL].rolling(30, min_periods=7).mean()
    d["wl_anom_30d"] = d[TARGET_COL] - wl_roll30

    # Cyclical temporal encoding
    doy   = d.index.dayofyear.values
    month = d.index.month.values
    d["doy_sin"]   = np.sin(2 * np.pi * doy   / 365.25)
    d["doy_cos"]   = np.cos(2 * np.pi * doy   / 365.25)
    d["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    d["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    return d


# 24 features in the exact order the model expects
FEATURE_COLS = [
    "tp", "t2m", "ssr", "str", "sp",           # ERA5 raw (5)
    "tp_roll3", "tp_roll7", "tp_roll14",        # routing precip (5)
    "tp_roll30", "tp_roll60",
    "t2m_roll7", "t2m_roll14", "t2m_roll30",   # snowmelt temp (3)
    "tp_x_t2m",                                 # interaction (1)
    "wl_lag1", "wl_lag2", "wl_lag3",           # AR WL lags (5)
    "wl_lag7", "wl_lag14",
    "wl_anom_30d",                              # limb indicator (1)
    "doy_sin", "doy_cos",                       # temporal (4)
    "month_sin", "month_cos",
]
# Total: 24 features


# ─────────────────────────────────────────────────────────────────────────────
#  3.  NORMALISER  (Z-score, fitted on training data only)
# ─────────────────────────────────────────────────────────────────────────────

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

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"mean": self.mean, "std": self.std}, f, indent=2)

    def load(self, path: str):
        with open(path) as f:
            d = json.load(f)
        self.mean, self.std = d["mean"], d["std"]


# ─────────────────────────────────────────────────────────────────────────────
#  4.  CHRONOLOGICAL SPLIT  (70 / 10 / 20 with 90-day leakage gaps)
# ─────────────────────────────────────────────────────────────────────────────

def chronological_split(df: pd.DataFrame):
    """
    Splits the full DataFrame purely by row order (no shuffling).

    train  → first 70 % of rows
    val    → next  ~7 % of rows  (10 % of train period, for early stopping)
    test   → last  30 % of rows  (completely held-out)

    A HINDCAST_LEN-row gap is removed between every split boundary
    so that no test/val date can appear in a training hindcast window.
    """
    n       = len(df)
    gap     = HINDCAST_LEN
    n_test  = int(n * (1 - TRAIN_FRAC))
    n_val   = int(n * TRAIN_FRAC * VAL_FRAC)

    test_start  = n - n_test
    val_end     = test_start - gap
    val_start   = val_end - n_val
    train_end   = val_start - gap

    train = df.iloc[:train_end].copy()
    val   = df.iloc[val_start:val_end].copy()
    test  = df.iloc[test_start:].copy()

    for name, s in [("Train", train), ("Val", val), ("Test", test)]:
        print(f"  {name:5s}: {s.index[0].date()} → {s.index[-1].date()}"
              f"  ({len(s):,} days)")

    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
#  5.  SEQUENCE GENERATION  (single-step sliding window)
# ─────────────────────────────────────────────────────────────────────────────

def make_sequences(df_norm: pd.DataFrame, hindcast_len: int = HINDCAST_LEN):
    """
    For each valid position i produce:
      X[i] : df_norm[FEATURE_COLS].iloc[i : i+hindcast_len]  → [T, 24]
      y[i] : df_norm[TARGET_COL].iloc[i+hindcast_len]        → scalar

    Rows containing NaN (from lag features near the split boundary)
    are silently skipped.

    Returns
    ───────
    X     : float32 array  [N, hindcast_len, 24]
    y     : float32 array  [N]
    dates : list of Timestamps  (date of the PREDICTED day)
    """
    n_f  = len(FEATURE_COLS)
    arr  = df_norm[FEATURE_COLS + [TARGET_COL]].values.astype(np.float32)
    idx  = df_norm.index.tolist()

    Xl, yl, dl = [], [], []
    for i in range(len(arr) - hindcast_len):
        win = arr[i: i + hindcast_len, :n_f]
        tgt = arr[i + hindcast_len,    n_f]
        if np.isnan(win).any() or np.isnan(tgt):
            continue
        Xl.append(win); yl.append(tgt); dl.append(idx[i + hindcast_len])

    X = np.array(Xl, dtype=np.float32)
    y = np.array(yl, dtype=np.float32)
    print(f"    {len(X):,} sequences  X{X.shape}  y{y.shape}")
    return X, y, dl


def to_loader(X, y, shuffle) -> DataLoader:
    ds = TensorDataset(torch.FloatTensor(X).to(DEVICE),
                       torch.FloatTensor(y).to(DEVICE))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      drop_last=False)


# ─────────────────────────────────────────────────────────────────────────────
#  6.  MODEL
# ─────────────────────────────────────────────────────────────────────────────

class RoutingLSTM(nn.Module):
    """
    Single-step LSTM water level simulator.

    Reads a 90-day window of 24 routing-aware features,
    outputs next-day water level (in normalised space).
    """

    def __init__(self, n_features: int,
                 hidden_size: int   = HIDDEN_SIZE,
                 n_layers:    int   = N_LAYERS,
                 dropout:     float = DROPOUT):
        super().__init__()
        # Input projection improves convergence vs. raw feature injection
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
        """x: [B, T, n_features]  →  [B]"""
        x          = self.proj(x)           # [B, T, H]
        out, _     = self.lstm(x)           # [B, T, H]
        last       = self.drop(out[:, -1])  # [B, H]  — last timestep
        return self.head(last).squeeze(-1)  # [B]

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
#  7.  TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, train: bool) -> float:
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


def train_member(seed: int, tr_loader, vl_loader,
                 n_features: int, save_path: str):
    """Train one ensemble member. Returns (history, best_val_rmse)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = RoutingLSTM(n_features).to(DEVICE)
    print(f"  Trainable params: {model.n_params():,}")

    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, "min", patience=8, factor=0.5, min_lr=1e-6)

    best_val, no_impr = float("inf"), 0
    hist = {"train_rmse": [], "val_rmse": []}

    print(f"  {'Ep':>4} | {'Train RMSE (norm)':>18} | "
          f"{'Val RMSE (norm)':>16} | {'LR':>9}")
    print(f"  {'─'*55}")

    for ep in range(1, MAX_EPOCHS + 1):
        tr_loss = run_epoch(model, tr_loader, opt, train=True)
        vl_loss = run_epoch(model, vl_loader, opt, train=False)
        tr_rmse = tr_loss ** 0.5
        vl_rmse = vl_loss ** 0.5
        hist["train_rmse"].append(tr_rmse)
        hist["val_rmse"].append(vl_rmse)
        sched.step(vl_loss)
        lr = opt.param_groups[0]["lr"]

        tag = ""
        if vl_loss < best_val:
            best_val, no_impr = vl_loss, 0
            torch.save(model.state_dict(), save_path)
            tag = "  ✓"
        else:
            no_impr += 1

        if ep % 10 == 0 or tag:
            print(f"  {ep:>4} | {tr_rmse:>18.6f} | {vl_rmse:>16.6f} |"
                  f" {lr:>9.2e}{tag}")

        if no_impr >= PATIENCE:
            print(f"\n  Early stop ep={ep}  best val RMSE={best_val**0.5:.6f}")
            break

    return hist, best_val ** 0.5


# ─────────────────────────────────────────────────────────────────────────────
#  8.  METRICS
# ─────────────────────────────────────────────────────────────────────────────

def _clean(obs, sim):
    o, s = np.asarray(obs, float), np.asarray(sim, float)
    m = ~(np.isnan(o) | np.isnan(s))
    return o[m], s[m]

def nse(obs, sim):
    o, s = _clean(obs, sim)
    return float(1 - np.sum((o - s)**2) / np.sum((o - o.mean())**2))

def kge(obs, sim):
    o, s = _clean(obs, sim)
    r     = float(pearsonr(o, s)[0])
    alpha = s.std() / o.std()   if o.std()   > 0 else np.nan
    beta  = s.mean()/ o.mean()  if o.mean() != 0 else np.nan
    return float(1 - ((r-1)**2 + (alpha-1)**2 + (beta-1)**2)**0.5)

def rmse(obs, sim):
    o, s = _clean(obs, sim)
    return float(np.sqrt(np.mean((o - s)**2)))

def mae(obs, sim):
    o, s = _clean(obs, sim)
    return float(np.mean(np.abs(o - s)))

def pbias(obs, sim):
    o, s = _clean(obs, sim)
    return float(100 * (s - o).sum() / o.sum())

def r2(obs, sim):
    o, s = _clean(obs, sim)
    return float(pearsonr(o, s)[0]**2)

def print_metrics(obs, sim, label: str):
    print(f"\n  ── {label} ──────────────────────────")
    print(f"  NSE   : {nse(obs,sim):>8.4f}   (>0.75 good, >0.90 excellent)")
    print(f"  KGE   : {kge(obs,sim):>8.4f}   (>0.75 good, >0.90 excellent)")
    print(f"  R²    : {r2(obs,sim):>8.4f}")
    print(f"  RMSE  : {rmse(obs,sim):>8.4f} m")
    print(f"  MAE   : {mae(obs,sim):>8.4f} m")
    print(f"  PBias : {pbias(obs,sim):>8.2f} %   (|<10| = acceptable)")


# ─────────────────────────────────────────────────────────────────────────────
#  9.  PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_training(histories, plot_dir):
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["steelblue", "darkorange", "green"]
    for i, h in enumerate(histories):
        c = colors[i % 3]
        ax.plot(h["train_rmse"], lw=1, ls="--", alpha=0.45, color=c,
                label=f"Train {i+1}")
        ax.plot(h["val_rmse"],   lw=2, alpha=0.9,  color=c,
                label=f"Val {i+1}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("RMSE (normalised)")
    ax.set_title("Ensemble Training & Validation Loss")
    ax.legend(ncol=2, fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "01_training_curves.png"), dpi=130)
    plt.close()


def plot_hydrograph(dates, obs, pred, label, plot_dir):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    ax.plot(dates, obs,  "k",           lw=1.2, label="Observed")
    ax.plot(dates, pred, "steelblue",   lw=0.9, alpha=0.85,
            label="Simulated (ensemble)")
    ax.set_ylabel("Water Level (m)")
    ax.set_title(f"Brahmaputra-Jamuna WL — {label}\n"
                 f"NSE={nse(obs,pred):.3f}  KGE={kge(obs,pred):.3f}  "
                 f"RMSE={rmse(obs,pred):.3f} m  R²={r2(obs,pred):.4f}")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    resid = np.asarray(pred) - np.asarray(obs)
    clr   = ["tomato" if r > 0 else "steelblue" for r in resid]
    axes[1].bar(dates, resid, color=clr, width=1, alpha=0.6)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_ylabel("Residual (m)"); axes[1].grid(alpha=0.3)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fname = f"02_hydrograph_{label.lower().replace(' ', '_')}.png"
    plt.savefig(os.path.join(plot_dir, fname), dpi=130)
    plt.close()


def plot_scatter(obs, pred, label, plot_dir):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(obs, pred, s=1.5, alpha=0.15, color="steelblue",
               rasterized=True)
    lo = min(min(obs), min(pred)) - 0.3
    hi = max(max(obs), max(pred)) + 0.3
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="1:1")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Observed WL (m)"); ax.set_ylabel("Simulated WL (m)")
    ax.set_title(f"Scatter — {label}\n"
                 f"NSE={nse(obs,pred):.3f}  R²={r2(obs,pred):.4f}  "
                 f"RMSE={rmse(obs,pred):.3f} m")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    fname = f"03_scatter_{label.lower().replace(' ', '_')}.png"
    plt.savefig(os.path.join(plot_dir, fname), dpi=130)
    plt.close()


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
    ax.set_title("Seasonal Absolute Error — Test Period")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "04_seasonal_error.png"), dpi=130)
    plt.close()


def plot_feature_correlation(df_feat, plot_dir):
    """Bar chart: Pearson r between each feature and water_level."""
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
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Pearson r  with  water_level")
    ax.set_title("Feature–Target Correlation  (validates routing-lag design)")
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "05_feature_correlation.png"), dpi=130)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  10.  HRES 15-DAY ROLLING INFERENCE
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
    hres_df      : 15-row DataFrame indexed by forecast dates.
                   Required columns: tp [mm/day], t2m [K], ssr [J/m²],
                                     str [J/m²], sp [Pa]
                   Temperature MUST be in Kelvin — converted internally.

    history_df   : ≥ 150 days of recent ERA5 + observed WL ending on
                   the day BEFORE the first HRES forecast date.
                   Columns: tp, t2m [K], ssr, str, sp, water_level [m]

    model_dir    : folder containing best_model_0.pt … best_model_N.pt
    norm_path    : path to normalizer_stats.json
    feat_path    : path to feature_cols.json
    n_features   : len(FEATURE_COLS) — must match trained model

    How autoregressive rolling works
    ─────────────────────────────────
    Day 1 (lead=1):
      • Append HRES row-0 to history.
      • Recompute ALL rolling features (windows use true history).
      • Take last hindcast_len rows → run model → predicted WL(T+1).

    Day 2 (lead=2):
      • Fill WL(T+1) = prediction from day 1.
      • Append HRES row-1 to history.
      • Recompute features → run model → predicted WL(T+2).

    This propagates the predicted WL back through wl_lag1/2/3/7/14 and
    wl_anom_30d in subsequent steps — exactly the routing feedback you
    described.

    Returns
    ───────
    pd.DataFrame:  date | lead_day | predicted_wl_m | lower_m | upper_m
    """
    # Load saved artifacts
    norm = Normalizer()
    norm.load(norm_path)
    with open(feat_path) as f:
        feat_cols = json.load(f)["feature_cols"]

    # Convert HRES units  (K → °C,  Pa → hPa)
    hres = hres_df.copy()
    if hres["t2m"].median() > 100:
        hres["t2m"] = hres["t2m"] - 273.15
    if hres["sp"].median() > 10000:
        hres["sp"]  = hres["sp"]  / 100.0

    # Convert history units
    hist = history_df.copy()
    if hist["t2m"].median() > 100:
        hist["t2m"] = hist["t2m"] - 273.15
    if hist["sp"].median() > 10000:
        hist["sp"]  = hist["sp"]  / 100.0

    # Load ensemble
    models = []
    for pt in sorted(f for f in os.listdir(model_dir) if f.endswith(".pt")):
        m = RoutingLSTM(n_features).to(DEVICE)
        m.load_state_dict(
            torch.load(os.path.join(model_dir, pt), map_location=DEVICE))
        m.eval()
        models.append(m)
    print(f"\n  Loaded {len(models)} ensemble model(s) for HRES inference")

    # Rolling inference
    running = hist.copy()
    results = []

    for step in range(len(hres)):
        lead        = step + 1
        fore_date   = hres.index[step]

        # Append this HRES day — WL unknown yet
        new_row = hres.iloc[[step]][["tp", "t2m", "ssr", "str", "sp"]].copy()
        new_row[TARGET_COL] = np.nan
        running = pd.concat([running, new_row])

        # Recompute all features on the full (extended) history
        full = engineer_features(running)

        # Input window = last hindcast_len rows
        win_df = full[feat_cols].iloc[-hindcast_len:]
        win    = win_df.values.astype(np.float32)

        if np.isnan(win).any():
            # Fallback: carry forward last known WL
            pred_wl  = float(running[TARGET_COL].dropna().iloc[-1])
            pred_std = 0.0
            print(f"    Lead {lead:>2}  {fore_date.date()}  "
                  f"NaN in window — using last known WL={pred_wl:.3f} m")
        else:
            # Normalise window
            win_norm_df = pd.DataFrame(win, columns=feat_cols)
            win_norm    = norm.transform(win_norm_df)[feat_cols].values \
                              .astype(np.float32)

            x_t = torch.FloatTensor(win_norm).unsqueeze(0).to(DEVICE)
            preds_norm = []
            with torch.no_grad():
                for m in models:
                    preds_norm.append(float(m(x_t).cpu().item()))

            pred_wl  = float(norm.inverse_wl(np.array([np.mean(preds_norm)]))[0])
            pred_std = float(np.std(preds_norm)) * norm.std[TARGET_COL]
            print(f"    Lead {lead:>2}  {fore_date.date()}  "
                  f"WL = {pred_wl:.3f} m  (±{pred_std:.3f})")

        # Feed prediction back into running history for next step
        running.iloc[-1, running.columns.get_loc(TARGET_COL)] = pred_wl

        results.append({
            "date"           : fore_date,
            "lead_day"       : lead,
            "predicted_wl_m" : round(pred_wl,  4),
            "lower_m"        : round(pred_wl - 2 * pred_std, 4),
            "upper_m"        : round(pred_wl + 2 * pred_std, 4),
        })

    out_df = pd.DataFrame(results).set_index("date")
    print("\n  HRES 15-day forecast:")
    print(out_df.to_string())
    return out_df


# ─────────────────────────────────────────────────────────────────────────────
#  11.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t0       = time.time()
    plot_dir = os.path.join(OUT_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    print(f"\n{'═'*60}")
    print(f"  Brahmaputra-Jamuna Routing-Aware LSTM  v2")
    print(f"  Device  : {DEVICE}")
    print(f"  Outputs : {OUT_DIR}")
    print(f"{'═'*60}")

    # ── Step 1: load raw data ─────────────────────────────────────────────
    print("\n[1] Loading raw data")
    df_raw = load_raw(ERA5_CSV, WL_CSV)
    print(f"  Rows      : {len(df_raw):,}")
    print(f"  WL range  : {df_raw[TARGET_COL].min():.2f} – "
          f"{df_raw[TARGET_COL].max():.2f} m")
    print(f"  t2m range : {df_raw['t2m'].min():.1f} – "
          f"{df_raw['t2m'].max():.1f} °C  (after K→°C)")
    print(f"  tp range  : {df_raw['tp'].min():.2f} – "
          f"{df_raw['tp'].max():.2f} mm/d")

    # ── Step 2: engineer features ─────────────────────────────────────────
    print("\n[2] Engineering routing-aware features")
    df = engineer_features(df_raw)
    print(f"  Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    plot_feature_correlation(df, plot_dir)
    print("  → 05_feature_correlation.png")

    # ── Step 3: split ─────────────────────────────────────────────────────
    print("\n[3] Chronological split  (70 % train / 30 % test)")
    train, val, test = chronological_split(df)

    # ── Step 4: normalise ─────────────────────────────────────────────────
    print("\n[4] Fitting Z-score normalizer on training data only")
    norm = Normalizer()
    norm.fit(train, FEATURE_COLS + [TARGET_COL])
    norm.save(os.path.join(OUT_DIR, "normalizer_stats.json"))
    with open(os.path.join(OUT_DIR, "feature_cols.json"), "w") as f:
        json.dump({"feature_cols": FEATURE_COLS}, f, indent=2)

    print(f"  {'Feature':<16} {'Mean':>12} {'Std':>12}")
    print(f"  {'─'*42}")
    for c in FEATURE_COLS[:5] + [TARGET_COL]:
        print(f"  {c:<16} {norm.mean[c]:>12.4f} {norm.std[c]:>12.4f}")
    print(f"  ... (saved to normalizer_stats.json)")

    train_n = norm.transform(train)
    val_n   = norm.transform(val)
    test_n  = norm.transform(test)

    # ── Step 5: sequences ─────────────────────────────────────────────────
    print("\n[5] Creating sliding-window sequences "
          f"(hindcast={HINDCAST_LEN} days)")
    print("  Train:"); X_tr, y_tr, d_tr = make_sequences(train_n)
    print("  Val  :"); X_v,  y_v,  d_v  = make_sequences(val_n)
    print("  Test :"); X_te, y_te, d_te = make_sequences(test_n)

    n_feat = X_tr.shape[-1]
    print(f"\n  Input features  : {n_feat}")

    tr_loader = to_loader(X_tr, y_tr, shuffle=True)
    vl_loader = to_loader(X_v,  y_v,  shuffle=False)
    te_loader = to_loader(X_te, y_te, shuffle=False)

    # ── Step 6: train ensemble ─────────────────────────────────────────────
    print(f"\n[6] Training {N_ENSEMBLE}-member ensemble "
          f"(max {MAX_EPOCHS} epochs, patience={PATIENCE})")
    histories, mpaths = [], []
    for i in range(N_ENSEMBLE):
        print(f"\n── Ensemble member {i+1}/{N_ENSEMBLE} ─────────────────────")
        path = os.path.join(OUT_DIR, f"best_model_{i}.pt")
        h, bv = train_member(42 + i*100, tr_loader, vl_loader, n_feat, path)
        histories.append(h)
        mpaths.append(path)
        print(f"  Saved → {path}  (best val RMSE norm = {bv:.5f})")

    plot_training(histories, plot_dir)
    print("\n  → 01_training_curves.png")

    # ── Step 7: load best weights & predict ───────────────────────────────
    print("\n[7] Loading best weights & running predictions")
    ensemble = []
    for path in mpaths:
        m = RoutingLSTM(n_feat).to(DEVICE)
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        m.eval()
        ensemble.append(m)

    def predict_split(loader):
        all_preds, all_true = [[] for _ in ensemble], []
        with torch.no_grad():
            for Xb, yb in loader:
                for j, m in enumerate(ensemble):
                    all_preds[j].extend(m(Xb).cpu().numpy().tolist())
                all_true.extend(yb.cpu().numpy().tolist())
        pred_norm = np.mean([np.array(p) for p in all_preds], axis=0)
        return (norm.inverse_wl(np.array(all_true)),
                norm.inverse_wl(pred_norm))

    obs_tr, prd_tr = predict_split(tr_loader)
    obs_v,  prd_v  = predict_split(vl_loader)
    obs_te, prd_te = predict_split(te_loader)

    # ── Step 8: metrics ───────────────────────────────────────────────────
    print("\n[8] Performance Metrics")
    print_metrics(obs_tr, prd_tr, "TRAINING SET")
    print_metrics(obs_v,  prd_v,  "VALIDATION SET")
    print_metrics(obs_te, prd_te, "TEST SET  ← completely held-out")

    # ── Step 9: plots ─────────────────────────────────────────────────────
    print("\n[9] Saving plots")
    plot_hydrograph(d_tr, obs_tr, prd_tr, "Training",   plot_dir)
    plot_hydrograph(d_te, obs_te, prd_te, "Test",        plot_dir)
    plot_scatter(obs_tr, prd_tr, "Training", plot_dir)
    plot_scatter(obs_te, prd_te, "Test",     plot_dir)
    plot_seasonal_error(d_te, obs_te, prd_te, plot_dir)
    print("  → 02_hydrograph_training/test.png")
    print("  → 03_scatter_training/test.png")
    print("  → 04_seasonal_error.png")

    # ── Step 10: save CSV ─────────────────────────────────────────────────
    print("\n[10] Saving predictions CSV")
    rows = []
    for split, dates, obs, pred in [
        ("train", d_tr, obs_tr, prd_tr),
        ("val",   d_v,  obs_v,  prd_v),
        ("test",  d_te, obs_te, prd_te),
    ]:
        for d, o, p in zip(dates, obs, pred):
            rows.append({"split": split, "date": d.date(),
                         "obs_wl_m":  round(float(o), 4),
                         "pred_wl_m": round(float(p), 4),
                         "error_m":   round(float(p - o), 4)})
    csv_path = os.path.join(OUT_DIR, "all_predictions.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  → {csv_path}  ({len(rows):,} rows)")

    print(f"\n{'═'*60}")
    print(f"  Total runtime: {(time.time()-t0)/60:.1f} min")
    print(f"  All outputs  : {OUT_DIR}")
    print(f"{'═'*60}")

    # ── HRES usage reminder ───────────────────────────────────────────────
    print("""
╔══════════════════════════════════════════════════════════════╗
║  HRES 15-DAY FORECAST — how to use after training           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  from jamuna_lstm import run_hres_forecast                ║
║                                                              ║
║  # hres_df   : 15-row DataFrame, index = forecast dates      ║
║  #   columns : tp [mm/day]  t2m [K]  ssr [J/m²]             ║
║  #             str [J/m²]   sp [Pa]                         ║
║  #   t2m in Kelvin — converted to °C internally              ║
║                                                              ║
║  # history_df: last 150 days of ERA5 + observed WL           ║
║  #   columns : same as hres_df + water_level [m]             ║
║                                                              ║
║  fc = run_hres_forecast(                                     ║
║      hres_df    = hres_df,                                   ║
║      history_df = history_df,                                ║
║      model_dir  = r"/mnt/d/Jamuna_LSTM/outputs",             ║
║      norm_path  = r".../outputs/normalizer_stats.json",      ║
║      feat_path  = r".../outputs/feature_cols.json",          ║
║      n_features = 24,                                        ║
║  )                                                           ║
║                                                              ║
║  # fc columns: predicted_wl_m  lower_m  upper_m             ║
║  # lead 1  = tomorrow                                        ║
║  # lead 15 = 15 days from now                                ║
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()