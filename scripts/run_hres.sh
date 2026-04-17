#!/bin/bash

# Usage: ./scripts/run_hres.sh YYYYMMDD
# Example: ./scripts/run_hres.sh 20250615

# Exit immediately if a command fails
set -e

# Check if date argument is provided
if [ -z "$1" ]; then
  echo "Error: No date argument provided."
  echo "Usage: $0 YYYYMMDD"
  exit 1
fi

DATE_ARG=$1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Activate the lhasa environment
source /home/rock_ubuntu/miniconda3/bin/activate lhasa

# Run the scripts sequentially with the same date argument
cd "$REPO_ROOT"
python scripts/ecmwf_hres_downloader.py "$DATE_ARG"
python scripts/ecmwf_hres_daily_converter.py "$DATE_ARG"
python scripts/ecmwf_hres_wavg.py "$DATE_ARG"

echo "All scripts completed successfully for date $DATE_ARG"
