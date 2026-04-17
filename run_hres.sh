#!/bin/bash

# Usage: ./run_ecmwf.sh YYYYMMDD
# Example: ./run_ecmwf.sh 20250615

# Exit immediately if a command fails
set -e

# Check if date argument is provided
if [ -z "$1" ]; then
  echo "Error: No date argument provided."
  echo "Usage: $0 YYYYMMDD"
  exit 1
fi

DATE_ARG=$1

# Activate the lhasa environment
source /home/rock_ubuntu/miniconda3/bin/activate lhasa

# Run the scripts sequentially with the same date argument
python ecmwf_hres_downloader.py "$DATE_ARG"
python ecmwf_hres_daily_converter.py "$DATE_ARG"
python ecmwf_hres_wavg.py "$DATE_ARG"

echo "All scripts completed successfully for date $DATE_ARG"