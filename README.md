# LSTM Flood Forecasting Pipeline

Operational LSTM-based water-level forecasting workflow for multiple Bangladesh river basins/stations using:

- ECMWF HRES forecast data
- Basin-weighted meteorological forcing
- Observed water-level updates from BWDB and FFWC
- Basin-specific trained LSTM ensembles

The repository is organized to keep the project root cleaner:

- reusable operational entry-point scripts live in `scripts/`
- basin-specific trained artifacts live in `basins/<basin>/model/`

## What This Project Does

For a given forecast date, the project can:

1. Download ECMWF HRES forecast fields over a regional South Asia window.
2. Convert sub-daily HRES fields to daily gridded NetCDF.
3. Compute basin-weighted daily meteorological forcings for each basin polygon.
4. Update observed daily water-level records from BWDB and FFWC APIs.
5. Run trained LSTM models for each basin to generate 15-day water-level forecasts.

## Current Repository Layout

```text
LSTM/
├── basins/
│   ├── bahadurabaad/
│   ├── bhairab_bazar/
│   ├── cumilla/
│   ├── hardinge/
│   └── muhuri/
├── BWDB_WL_Data/                  # legacy/raw water-level data
├── docs/
│   ├── Global prediction of extreme floods.pdf
│   └── lstm_hyperparameter_manual.docx
├── hres_data/                     # downloaded + processed ECMWF HRES NetCDFs
├── scripts/
│   ├── ecmwf_hres_downloader.py
│   ├── ecmwf_hres_daily_converter.py
│   ├── ecmwf_hres_wavg.py
│   ├── update_WL.py
│   ├── lstm_prediction.py
│   └── run_hres.sh
├── updated_WL_data/               # merged station-wise daily observed WL CSVs
├── ecmwf_variables.csv            # variable metadata and unit conversion info
├── lstm.ipynb                     # notebook experimentation / prototyping
├── requirements.txt
└── README.md
```

## Basin Folder Layout

Each basin folder under `basins/` is self-contained and usually includes:

```text
basins/<basin_name>/
├── basin_<basin_name>.json        # basin geometry used for weighted averaging
├── input/                         # HRES basin-forcing CSVs by forecast date
├── output/                        # final forecast CSVs by forecast date
├── model/                         # trained model artifacts
│   ├── best_model_0.pt
│   ├── best_model_1.pt
│   ├── best_model_2.pt
│   ├── normalizer_stats.json
│   ├── feature_cols.json
│   ├── all_predictions.csv
│   └── plots/
├── ERA5_*.csv                     # historical basin-averaged meteorological data
├── *_lstm.py                      # basin-specific training script
└── observed_wl.csv or daily WL CSV
```

## Main Scripts

All operational entry points are under `scripts/`.

### `scripts/ecmwf_hres_downloader.py`

- Downloads ECMWF open-data HRES GRIB slices by variable and timestep.
- Crops to a regional bounding box.
- Merges selected variables into a single NetCDF file at `hres_data/tmp_<YYYYMMDD>.nc`.

Default variables:

- `tp`
- `2t`
- `2d`
- `ssr`
- `str`
- `sp`

### `scripts/ecmwf_hres_daily_converter.py`

- Reads merged sub-daily HRES NetCDF.
- Converts instantaneous variables (`t2m`, `d2m`, `sp`) to daily means.
- De-accumulates accumulated variables (`tp`, `ssr`, `str`) and aggregates to daily sums.
- Writes `hres_data/daily_<YYYYMMDD>.nc`.

### `scripts/ecmwf_hres_wavg.py`

- Reads the daily HRES NetCDF.
- Discovers basin polygons under `basins/`.
- Computes spatially weighted daily averages with `pyscissor`.
- Writes one forcing CSV per basin to:
  - `basins/<basin>/input/hres_<basin>_<YYYYMMDD>.csv`

### `scripts/update_WL.py`

- Updates station-wise daily observed water-level CSVs in `updated_WL_data/`.
- Uses BWDB as the primary source.
- Uses FFWC as a fallback/fill source where station IDs are available.
- Chooses one representative daily value using preferred times:
  - `09:00`
  - otherwise `12:00`
  - otherwise `06:00`

### `scripts/lstm_prediction.py`

- Runs 15-day water-level inference for all basins or selected basins.
- Loads:
  - basin-specific HRES forcing CSV
  - observed recent water levels
  - saved model weights from `basins/<basin>/model/`
  - saved normalizer statistics from `basins/<basin>/model/`
- Produces:
  - `basins/<basin>/output/forecast_<YYYYMMDD>.csv`

### `scripts/run_hres.sh`

- Linux helper that runs the HRES preprocessing sequence from the repo root.
- Calls:
  - `scripts/ecmwf_hres_downloader.py`
  - `scripts/ecmwf_hres_daily_converter.py`
  - `scripts/ecmwf_hres_wavg.py`

## Modeling Approach

The original project notes referenced an encoder-decoder LSTM inspired by Nearing et al. (2024). The code currently used for training and inference in this repository is a routing-aware single-step LSTM ensemble that is rolled forward autoregressively for 15 forecast days.

Core characteristics of the current implementation:

- Hindcast window: 90 days for the large-river setups, with basin-specific variants for smaller flashy basins
- Model type: single-step LSTM with autoregressive rollout during forecast
- Ensemble size: typically 3 models per basin
- Hidden size: typically 128 for the larger basins, with basin-specific adjustments
- Inputs: engineered meteorological, hydrological, and seasonal features

### Feature Groups

The forecast models use a combination of:

- Raw meteorological forcings:
  - `tp`
  - `t2m`
  - `ssr`
  - `str`
  - `sp`
- Rolling precipitation windows
- Rolling temperature windows
- Interaction terms
- Water-level lag features
- Water-level anomaly features
- Seasonal encodings

This setup is designed to capture routing memory, delayed runoff response, and strong autoregressive persistence in river stage.

## Data Flow

```text
ECMWF Open Data
  -> scripts/ecmwf_hres_downloader.py
  -> hres_data/tmp_<date>.nc
  -> scripts/ecmwf_hres_daily_converter.py
  -> hres_data/daily_<date>.nc
  -> scripts/ecmwf_hres_wavg.py
  -> basins/<basin>/input/hres_<basin>_<date>.csv

BWDB + FFWC APIs
  -> scripts/update_WL.py
  -> updated_WL_data/WL_<station>_daily.csv

Basin model artifacts + latest HRES input + updated observed WL
  -> scripts/lstm_prediction.py
  -> basins/<basin>/output/forecast_<date>.csv
```

## End-to-End Operational Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Note: the weighted-averaging script also depends on packages used in code but not explicitly listed in older versions of `requirements.txt`, especially:

- `fiona`
- `pyscissor`
- `cfgrib`
- `requests`

### 2. Download and preprocess HRES for a forecast date

```bash
python scripts/ecmwf_hres_downloader.py 20250615
python scripts/ecmwf_hres_daily_converter.py 20250615
python scripts/ecmwf_hres_wavg.py 20250615
```

Or use the helper shell script on Linux environments:

```bash
./scripts/run_hres.sh 20250615
```

### 3. Update observed water levels

```bash
python scripts/update_WL.py 20250615
```

Optional single-station update:

```bash
python scripts/update_WL.py 20250615 bahadurabaad
```

### 4. Run forecasts

Run all discovered basins:

```bash
python scripts/lstm_prediction.py 20250615
```

Run selected basins only:

```bash
python scripts/lstm_prediction.py 20250615 bahadurabaad hardinge muhuri
```

## Inputs and Outputs

### Forecast-date inputs

For each basin forecast run, the pipeline expects:

- Basin HRES forcing:
  - `basins/<basin>/input/hres_<basin>_<YYYYMMDD>.csv`
- Trained model weights:
  - `basins/<basin>/model/best_model_*.pt`
- Normalizer:
  - `basins/<basin>/model/normalizer_stats.json`
- Observed water level:
  - `updated_WL_data/WL_<basin>_daily.csv`

### Forecast output

Each forecast CSV includes:

- `date`
- `lead_day`
- `predicted_wl_m`
- `lower_m`
- `upper_m`

Saved to:

```text
basins/<basin>/output/forecast_<YYYYMMDD>.csv
```

## Training Assets

Training remains basin-specific rather than centralized in a shared module. Basin scripts such as:

- `basins/bahadurabaad/jamuna_lstm.py`
- `basins/hardinge/ganges_lstm.py`
- `basins/muhuri/muhuri_lstm.py`

handle:

- reading historical ERA5/basin CSVs
- engineering features
- chronological splitting
- normalization
- LSTM training
- ensemble export
- diagnostic plot generation

Typical training outputs written under `basins/<basin>/model/`:

- `best_model_0.pt`, `best_model_1.pt`, `best_model_2.pt`
- `normalizer_stats.json`
- `feature_cols.json`
- `all_predictions.csv`
- `plots/*.png`

## Known Notes

- The repo root now keeps operational scripts under `scripts/` to reduce clutter.
- Basin naming is not perfectly uniform across scripts and folders, so some station/basin mappings are handled with fallbacks.
- The forecast pipeline is operational and produces outputs, but forecast quality should still be checked basin by basin.
- Some sample forecast CSVs in the repo are nearly constant across all 15 lead days, which is worth validating during model QA.

## Reference

The project is inspired by:

> Nearing et al. (2024). Global prediction of extreme floods in ungauged watersheds. Nature, 627, 559-563.
