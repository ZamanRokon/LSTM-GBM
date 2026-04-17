#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional
import os
import sys
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parent.parent

# Station map (BWDB Station ID)
STATIONS = {
    "bahadurabaad": "SW46.9L",
    "hardinge": "SW90",
    "dalia": "SW291.5R",
    "muhuri": "SW212",
    "laurergarh": "SW131.5",
    "amalshid": "SW172",
    "amalshid_s": "SW172",
    "sheola": "SW173",
    "monu": "SW201",
    "jaflong": "SW233A",
    "sarighat": "SW251",
    "bijoypur": "SW262",
    "jaldhup": "SW265",
    "sylhet": "SW267",
    "sunamganj": "SW269",
    "lubachara": "SW326",
    "muslimpur": "SW333",
    "jariajhanjail": "SW36",
    "kamalganj": "SW67",
    "cumilla": "SW110",
}


# FFWC API station
FFWC_STATIONS = {
    "bahadurabad": 66,
    "hardingebridge": 43,
    "dalia": 80,
    "parshuram": 21,
    "laurergarh": 70,
    "amalshid": 60,
    "amalshid_s": 60,
    "sheola": 61,
    "monu": 47,
    "jaflong": 154,
    "sarighat": 68,
    "bijoypur": 121,
    # "jaldhup": None,   # no new FFWC ID
    "sylhet": 59,
    "sunamganj": 65,
    "lubachara": 145,
    "muslimpur": 122,
    "jariajhanjail": 63,
    "kamalganj": 86,
    "cumilla": 23,
}


# Helpers: choosing a daily rep
PREFERRED_TIMES = ["09:00:00", "12:00:00", "06:00:00"]

def choose_one_per_day(day_times: dict) -> Optional[float]:
    """Given dict of 'HH:MM:SS' -> WL, pick 09:00, else 12:00, else 06:00."""
    for t in PREFERRED_TIMES:
        if t in day_times:
            return day_times[t]
    return None

def stamp(date_str: str, time_str: str) -> str:
    """Format to index: YYYY-MM-DDTHH:MM:SS.000000+0000"""
    return f"{date_str}T{time_str}.000000+0000"


# Primary: BWDB
def fetch_bwdb(station_id: str, from_year: int, to_year: int, cookie: Optional[str] = None) -> list:
    """
    Returns list in the format returned by BWDB API.
    Endpoint returns JSON; each element typically looks like:
    ["01-Jan-25 06:00:00","5.09"]
    """
    url = "http://www.hydrology.bwdb.gov.bd/hydro/module/surface_water/wl_cont_graph_year_wise.php"
    payload = {"station_id": station_id, "from_year": str(from_year), "to_year": str(to_year)}
    headers = {"User-Agent": "Mozilla/5.0"}
    if cookie:
        headers["Cookie"] = cookie

    r = requests.post(url, headers=headers, data=payload, timeout=60)
    if r.status_code != 200:
        raise Exception(f"BWDB HTTP {r.status_code}")

    try:
        data = r.json()
    except Exception:
        raise Exception("BWDB: response is not valid JSON")

    if not isinstance(data, list):
        raise Exception("BWDB: API did not return a list")

    return data

def reduce_bwdb_to_daily_map(items: list) -> dict:
    """
    Convert BWDB list -> { 'YYYY-MM-DDTHH:MM:SS.000000+0000': wl, ... }
    Picks one daily value using preferred times: 09:00 > 12:00 > 06:00
    """
    days = defaultdict(dict)

    for row in items:
        try:
            ts, wl = row[0], row[1]
            dt = datetime.strptime(ts, "%d-%b-%y %H:%M:%S")
            d = dt.strftime("%Y-%m-%d")
            t = dt.strftime("%H:%M:%S")
            days[d][t] = float(wl)
        except Exception:
            continue

    out = {}
    for d, times in days.items():
        for t in PREFERRED_TIMES:
            if t in times:
                out[stamp(d, t)] = times[t]
                break

    return out


# Secondary: FFWC
def fetch_ffwc(station_id: int, from_date: str, to_date: str) -> dict:
    """
    Calls the new FFWC API:
    https://api.ffwc.gov.bd/data_load/seven-days-observed-waterlevel-by-station/<station_id>/?format=json
    Returns:
        { 'YYYY-MM-DDTHH:MM:SS.000000+0000': wl }
    """
    url = f"https://api.ffwc.gov.bd/data_load/seven-days-observed-waterlevel-by-station/{station_id}/?format=json"
    headers = {"User-Agent": "Mozilla/5.0"}

    r = requests.get(url, headers=headers, timeout=60)
    if r.status_code != 200:
        raise Exception(f"FFWC HTTP {r.status_code}")

    data = r.json()
    if not isinstance(data, list):
        raise Exception("FFWC response invalid")

    start_d = datetime.strptime(from_date, "%Y-%m-%d").date()
    end_d = datetime.strptime(to_date, "%Y-%m-%d").date()

    days = defaultdict(dict)

    for rec in data:
        try:
            dt = datetime.strptime(rec["wl_date"], "%Y-%m-%dT%H:%M:%S%z")
            d = dt.date()
            if d < start_d or d > end_d:
                continue

            d_str = dt.strftime("%Y-%m-%d")
            t_str = dt.strftime("%H:%M:%S")
            wl = float(rec["waterlevel"])
            days[d_str][t_str] = wl
        except Exception:
            continue

    out = {}
    for d, times in days.items():
        for t in PREFERRED_TIMES:
            if t in times:
                out[stamp(d, t)] = times[t]
                break

    return out


# CSV I/O
def read_existing_csv(path: str) -> pd.DataFrame:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            df = pd.read_csv(path, index_col="date")
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=["water_level"]).rename_axis("date")

def convert_timestamp_to_date_format(timestamp_str: str) -> str:
    """
    Convert ISO timestamp like '2025-01-01T09:00:00.000000+0000' to 'M/D/YYYY' format.
    """
    try:
        date_part = timestamp_str[:10]  # Extract YYYY-MM-DD
        dt = datetime.strptime(date_part, "%Y-%m-%d")
        return dt.strftime("%-m/%-d/%Y") if os.name != 'nt' else f"{dt.month}/{dt.day}/{dt.year}"
    except Exception:
        return timestamp_str

def update_csv(path: str, new_map: dict):
    df_old = read_existing_csv(path)
    
    # Convert timestamp keys to M/D/YYYY format
    converted_map = {}
    for k, v in new_map.items():
        date_key = convert_timestamp_to_date_format(k)
        converted_map[date_key] = v
    
    df_new = pd.DataFrame.from_dict(converted_map, orient="index", columns=["water_level"])
    df_new.index.name = "date"

    # merge, prefer new for same timestamps
    df_comb = df_old.combine_first(df_new)
    df_comb.update(df_new)
    
    # Sort by converting M/D/YYYY back to datetime for proper sorting
    def parse_date_str(date_str):
        try:
            parts = date_str.split('/')
            return datetime(int(parts[2]), int(parts[0]), int(parts[1]))
        except:
            return datetime.min
    
    df_comb = df_comb.sort_index(key=lambda x: x.map(parse_date_str))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_comb.to_csv(path)


# MAIN
def main():
    if len(sys.argv) < 2:
        print("Usage: python update_WL_merged.py <yyyymmdd> [station_name] [cookie]")
        sys.exit(1)

    run_str = sys.argv[1]
    station_filter = sys.argv[2] if len(sys.argv) > 2 else ""
    cookie = sys.argv[3] if len(sys.argv) > 3 else None

    try:
        run_date = datetime.strptime(run_str, "%Y%m%d")
    except Exception:
        print("Invalid date format. Use YYYYMMDD.")
        sys.exit(1)

    for name, sid in STATIONS.items():
        if station_filter and name.lower() != station_filter.lower():
            continue

        out_file = REPO_ROOT / "updated_WL_data" / f"WL_{name}_daily.csv"
        df_old = read_existing_csv(out_file)

        # last known daily date from existing CSV index
        last_date = None
        if not df_old.empty:
            try:
                last_idx = df_old.index[-1]  # e.g., "1/15/2025"
                # Convert M/D/YYYY back to YYYY-MM-DD for processing
                parts = last_idx.split('/')
                last_date = f"{parts[2]}-{parts[0].zfill(2)}-{parts[1].zfill(2)}"
            except Exception:
                last_date = None

        # Decide year range for BWDB pull
        from_year = 2010
        if last_date:
            try:
                from_year = int(last_date[:4])
            except Exception:
                pass
        to_year = run_date.year

        print(f"⏳ {name} ({sid}): BWDB {from_year}→{to_year}")

        # 1) Pull BWDB and reduce to daily
        bwdb_map = {}
        try:
            bwdb_raw = fetch_bwdb(sid, from_year, to_year, cookie)
            bwdb_map = reduce_bwdb_to_daily_map(bwdb_raw)
        except Exception as e:
            print(f"  ⚠️ BWDB error: {e}")

        # 2) Determine date window for FFWC
        if last_date:
            start_dt = datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)
        else:
            start_dt = datetime(from_year, 1, 1)

        end_dt = run_date
        ffwc_map = {}

        if start_dt <= end_dt:
            fd = start_dt.strftime("%Y-%m-%d")
            td = end_dt.strftime("%Y-%m-%d")

            ffwc_sid = FFWC_STATIONS.get(name)
            if ffwc_sid is None:
                print(f"  ⚠️ FFWC skipped: new API station ID not set for {name}")
            else:
                print(f"⏳ {name} (FFWC new id: {ffwc_sid}): FFWC {fd}→{td}")
                try:
                    ffwc_map = fetch_ffwc(ffwc_sid, fd, td)
                except Exception as e:
                    print(f"  ⚠️ FFWC error: {e}")

        # 3) Merge logic
        # BWDB is primary, FFWC only fills missing dates
        merged = dict(bwdb_map)
        bwdb_dates = {k[:10] for k in merged.keys()}

        for k, v in ffwc_map.items():
            d = k[:10]
            if d not in bwdb_dates:
                merged[k] = v

        # 4) Keep only truly new days if CSV already exists
        if last_date:
            merged = {k: v for k, v in merged.items() if k[:10] > last_date}

        # 5) Save
        if merged:
            update_csv(out_file, merged)
            print(f"✅ {name}: Added/updated {len(merged)} record(s).")
        else:
            print(f"ℹ️ {name}: No new data to add.")

if __name__ == "__main__":
    main()
