#!/usr/bin/env python3
"""
ECMWF HRES Data Downloader
Downloads, processes, and converts ECMWF HRES data to NetCDF with regional crop.
"""

import os
import sys
import json
import subprocess
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd
import xarray as xr
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import warnings

warnings.filterwarnings('ignore')

REPO_ROOT = Path(__file__).resolve().parent.parent


class ECMWFDownloader:
    def __init__(self, date_str, variables, lon_range=(70, 100), lat_range=(20, 35)):
        """
        Initialize ECMWF downloader.
        
        Args:
            date_str: Date in YYYYMMDD format
            variables: List of variable names (e.g., ['tp', '2t', 'msl'])
            lon_range: (min, max) longitude
            lat_range: (min, max) latitude
        """
        self.date = date_str
        self.time = "00z"
        self.variables = variables
        self.lon_min, self.lon_max = lon_range
        self.lat_min, self.lat_max = lat_range
        
        # Spatial steps for 00z: 0-144h by 3h, 150-360h by 6h
        self.steps = list(range(0, 145, 3)) + list(range(150, 361, 6))
        
        self.base_url = f"https://storage.googleapis.com/ecmwf-open-data/{self.date}/{self.time}/ifs/0p25/oper"
        
        # Setup directories
        self.main_dir = REPO_ROOT / "hres_data"
        self.index_dir = self.main_dir / "index_files"
        self.tmp_dir = self.main_dir / "tmp"
        self.out_dir = self.main_dir
        
        for d in [self.index_dir, self.tmp_dir, self.out_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Variable metadata
        self.var_metadata = self._load_var_metadata()
        
        print("=" * 60)
        print(" ECMWF HRES Downloader (Python)")
        print("=" * 60)
        print(f"📅 Date: {self.date} | Time: {self.time}")
        print(f"📊 Variables: {', '.join(variables)}")
        print(f"🗺️  Region: Lon {self.lon_min}°-{self.lon_max}°, Lat {self.lat_min}°-{self.lat_max}°")
        print(f"⏱️  Steps: {len(self.steps)} timesteps")
        print("=" * 60)
    
    def _load_var_metadata(self):
        """Load variable metadata from CSV."""
        metadata = {}
        csv_path = REPO_ROOT / "ecmwf_variables.csv"
        
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                metadata[row['shortName']] = {
                    'long_name': row['long_name'],
                    'units': row['unit'],
                    'conversion_factor': float(row['conversion_factor']),
                    'aggregation_rule': row['aggregation_rule'],
                }
        return metadata
    
    def download_index_files(self):
        """Download index files for all timesteps."""
        print("\n" + "=" * 60)
        print("STEP 1: Downloading index files")
        print("=" * 60)
        
        downloaded = 0
        failed = 0
        
        for step in self.steps:
            url = f"{self.base_url}/{self.date}000000-{step}h-oper-fc.index"
            out_path = self.index_dir / f"{self.date}000000-{step}h-oper-fc.index"
            
            # Skip if already exists
            if out_path.exists():
                print(f"⏩ {step:3d}h - already exists")
                downloaded += 1
                continue
            
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200 and '"param"' in response.text:
                    with open(out_path, 'w') as f:
                        f.write(response.text)
                    print(f"✅ {step:3d}h")
                    downloaded += 1
                else:
                    print(f"⚠️  {step:3d}h - invalid content")
                    failed += 1
            except Exception as e:
                print(f"❌ {step:3d}h - {str(e)[:40]}")
                failed += 1
        
        print(f"\n✅ Downloaded: {downloaded}/{len(self.steps)} index files")
        return downloaded > 0
    
    def extract_variable_metadata(self, variable):
        """Extract metadata for a variable from all index files."""
        print(f"\n📋 Extracting JSON records for {variable}...")
        
        records = []
        index_files = list(self.index_dir.glob("*.index"))
        
        if not index_files:
            print(f"❌ No index files found")
            return []
        
        for idx_file in index_files:
            try:
                with open(idx_file, 'r') as f:
                    for line in f:
                        if f'"param": "{variable}"' in line:
                            records.append(json.loads(line))
            except:
                continue
        
        print(f"✅ Found {len(records)} records for {variable}")
        return records
    
    def download_grib_slice(self, variable, record):
        """Download a single GRIB2 slice using byte range."""
        step = record['step']
        offset = record['_offset']
        length = record['_length']
        end = offset + length - 1
        
        var_dir = self.main_dir / f"{variable}_data"
        var_dir.mkdir(parents=True, exist_ok=True)
        
        out_path = var_dir / f"{variable}_{step}h.grib2"
        
        # Skip if exists
        if out_path.exists() and out_path.stat().st_size > 0:
            return True
        
        url = f"{self.base_url}/{self.date}000000-{step}h-oper-fc.grib2"
        
        try:
            headers = {'Range': f'bytes={offset}-{end}'}
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code in [200, 206] and len(response.content) > 0:
                with open(out_path, 'wb') as f:
                    f.write(response.content)
                return True
            else:
                return False
        except Exception as e:
            print(f"Error downloading {variable} {step}h: {e}")
            return False
    
    def process_variable(self, variable):
        """Process a single variable: download, merge, crop, fix attributes."""
        print(f"\n{'=' * 60}")
        print(f"🔍 Processing variable: {variable}")
        print(f"{'=' * 60}")
        
        # Step 1: Extract metadata and download GRIB2 files
        print(f"⬇️  Downloading GRIB2 slices...")
        records = self.extract_variable_metadata(variable)
        
        if not records:
            print(f"❌ No records found for {variable}")
            return False
        
        # Download in parallel
        success_count = 0
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(self.download_grib_slice, variable, rec) for rec in records]
            for i, future in enumerate(as_completed(futures)):
                if future.result():
                    success_count += 1
                    print(f"   {i+1}/{len(records)} downloaded", end='\r')
        
        print(f"\n✅ Downloaded {success_count}/{len(records)} slices")
        
        var_dir = self.main_dir / f"{variable}_data"
        grib_files = sorted(var_dir.glob("*.grib2"), 
                           key=lambda x: int(x.stem.split('_')[1].replace('h', '')))
        
        if not grib_files:
            print(f"❌ No GRIB2 files found for {variable}")
            return False
        
        # Step 2: Convert GRIB2 to NetCDF with crop
        print(f"🔄 Converting GRIB2 to NetCDF (merged + cropped)...")
        out_file = self.out_dir / f"tmp_{self.date}_{variable}.nc"
        
        try:
            # Open all GRIB files
            datasets = []
            step_values = []
            
            for grib_file in grib_files:
                try:
                    ds = xr.open_dataset(str(grib_file), engine='cfgrib')
                    datasets.append(ds)
                    
                    # Extract step/lead time from filename
                    step = int(grib_file.stem.split('_')[1].replace('h', ''))
                    step_values.append(step)
                except Exception as e:
                    print(f"   ⚠️  Failed to read {grib_file.name}: {e}")
                    continue
            
            if not datasets:
                print(f"❌ Could not read any GRIB files for {variable}")
                return False
            
            # Concatenate along time dimension
            print(f"   Concatenating {len(datasets)} timesteps...")
            ds_merged = xr.concat(datasets, dim='time')
            
            # Create proper time coordinates based on step values
            # Reference time is the forecast initialization time
            if 'time' in ds_merged.coords:
                ref_time = ds_merged.coords['time'].values[0]
                # Create valid times: ref_time + step (in hours)
                valid_times = pd.to_datetime(
                    [pd.Timestamp(ref_time) + pd.Timedelta(hours=int(s)) for s in step_values]
                )
                ds_merged = ds_merged.assign_coords(time=valid_times)
            
            # Close individual datasets to free memory
            for ds in datasets:
                ds.close()
            
            # Crop region
            print(f"   Cropping region ({self.lon_min}-{self.lon_max}°E, {self.lat_min}-{self.lat_max}°N)...")
            ds_cropped = ds_merged.sel(
                longitude=slice(self.lon_min, self.lon_max),
                latitude=slice(self.lat_max, self.lat_min)  # Note: descending latitude
            )
            
            # Clean up unnecessary variables and attributes
            print(f"   Cleaning variables and attributes...")
            ds_cropped = self._clean_dataset(ds_cropped, variable)
            
            # Convert to NetCDF
            print(f"   Writing NetCDF...")
            ds_cropped.to_netcdf(out_file, format='netcdf4', engine='netcdf4', 
                                unlimited_dims=['time'])
            
            ds_merged.close()
            ds_cropped.close()
            
            file_size = out_file.stat().st_size / (1024**2)  # MB
            print(f"✅ Created: {out_file.name} ({file_size:.1f} MB)")
            
            return True
            
        except Exception as e:
            print(f"❌ Error processing {variable}: {e}")
            import traceback
            traceback.print_exc()
            if out_file.exists():
                out_file.unlink()
            return False
    
    def _clean_dataset(self, ds, var_name):
        """Clean up dataset: remove unnecessary variables and set proper attributes."""
        
        standard_names = {
            '2t': '2m_air_temperature',
            '2d': '2m_dew_point_temperature',
            'sp': 'surface_air_pressure',
            'tp': 'total_accumulated_precipitation',
            'ssr': 'surface_net_downward_shortwave_flux',
            'str': 'surface_net_downward_longwave_flux',
            'tcwv': 'total_column_vertically-integrated_water_vapour',
            'tcw': 'total_column_water',
            'msl': 'mean_sea_level_pressure',
        }
        
        # Find main data variable
        data_vars = [v for v in ds.data_vars if v not in ['step', 'surface', 'number']]
        if not data_vars:
            return ds
        
        main_var = data_vars[0]
        
        # Remove unnecessary data variables
        vars_to_drop = [v for v in ds.data_vars 
                       if v not in [main_var]]
        ds = ds.drop_vars(vars_to_drop, errors='ignore')
        
        # Keep only necessary coordinates
        coords_to_keep = ['time', 'latitude', 'longitude']
        coords_to_drop = [c for c in ds.coords if c not in coords_to_keep]
        ds = ds.drop_vars(coords_to_drop, errors='ignore')
        
        # Load metadata for this variable
        metadata = self.var_metadata.get(var_name, {})
        
        # Convert tp from m to mm FIRST (before setting attributes)
        if var_name == 'tp' and metadata.get('conversion_factor', 1.0) != 1.0:
            ds[main_var] = ds[main_var] * metadata['conversion_factor']
        
        # NOW set attributes AFTER data conversion
        if metadata:
            if 'long_name' in metadata:
                ds[main_var].attrs['long_name'] = metadata['long_name']
            if 'units' in metadata:
                ds[main_var].attrs['units'] = metadata['units']
        
        if var_name in standard_names:
            ds[main_var].attrs['standard_name'] = standard_names[var_name]
        
        # Clean coordinate attributes - keep only essential info
        if 'latitude' in ds.coords:
            ds.coords['latitude'].attrs = {
                'units': 'degrees_north',
                'standard_name': 'latitude'
            }
        
        if 'longitude' in ds.coords:
            ds.coords['longitude'].attrs = {
                'units': 'degrees_east',
                'standard_name': 'longitude'
            }
        
        # Clean time attributes
        if 'time' in ds.coords:
            ds.coords['time'].attrs = {
                'standard_name': 'time'
            }
        
        return ds
    
    
    def merge_variables(self):
        """Merge all individual variable files into a single NetCDF file."""
        print(f"\n{'=' * 60}")
        print(f"STEP 5: Merging variables into single file")
        print(f"{'=' * 60}")
        
        # Collect all individual variable files
        var_files = [self.out_dir / f"tmp_{self.date}_{var}.nc" 
                    for var in self.variables]
        var_files = [f for f in var_files if f.exists()]
        
        if not var_files:
            print("❌ No variable files found to merge")
            return False
        
        try:
            # Open all variable files
            datasets = []
            for var_file in var_files:
                try:
                    ds = xr.open_dataset(var_file)
                    datasets.append(ds)
                    print(f"   📖 Loaded: {var_file.name}")
                except Exception as e:
                    print(f"   ⚠️  Failed to load {var_file.name}: {e}")
                    continue
            
            if not datasets:
                print("❌ Could not load any variable files")
                return False
            
            # Merge along data_vars dimension (not time, since all share same time)
            print(f"   Merging {len(datasets)} datasets...")
            ds_merged = xr.merge(datasets)
            
            # Close individual datasets
            for ds in datasets:
                ds.close()
            
            # Write merged file
            merged_file = self.out_dir / f"tmp_{self.date}.nc"
            print(f"   Writing: {merged_file.name}...")
            ds_merged.to_netcdf(merged_file, format='netcdf4', engine='netcdf4',
                               unlimited_dims=['time'])
            ds_merged.close()
            
            file_size = merged_file.stat().st_size / (1024**2)
            print(f"✅ Merged file created: {merged_file.name} ({file_size:.1f} MB)")
            
            # Delete individual variable files after successful merge
            print(f"\n🗑️  Removing individual variable files...")
            for var_file in var_files:
                try:
                    var_file.unlink()
                    print(f"   Deleted: {var_file.name}")
                except Exception as e:
                    print(f"   ⚠️  Failed to delete {var_file.name}: {e}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error merging variables: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def cleanup_grib_files(self):
        """Clean up temporary GRIB2 files."""
        print(f"\n🧹 Cleaning GRIB2 temporary files...")
        
        for var in self.variables:
            var_dir = self.main_dir / f"{var}_data"
            if var_dir.exists():
                shutil.rmtree(var_dir)
                print(f"   Removed: {var_dir.name}")
        
        # Remove index files
        if self.index_dir.exists():
            shutil.rmtree(self.index_dir)
        
        print(f"✅ Cleanup complete")
    
    def run(self):
        """Run the full download and processing pipeline."""
        try:
            # Step 1: Download indexes
            if not self.download_index_files():
                print("❌ Failed to download index files")
                return False
            
            # Step 2-4: Process each variable
            print(f"\n{'=' * 60}")
            print(f"Processing {len(self.variables)} variables...")
            print(f"{'=' * 60}")
            
            success_vars = 0
            for variable in self.variables:
                if self.process_variable(variable):
                    success_vars += 1
            
            if success_vars == 0:
                print("\n❌ Failed to process any variables")
                return False
            
            # Step 5: Merge all variable files into single file
            if not self.merge_variables():
                print("⚠️  Failed to merge variables, but individual files are available")
            
            # Step 6: Summary
            print(f"\n{'=' * 60}")
            print(f"✅ SUMMARY")
            print(f"{'=' * 60}")
            print(f"Successfully processed: {success_vars}/{len(self.variables)} variables\n")
            print(f"📂 Output location: {self.out_dir.absolute()}\n")
            
            # Show merged file
            merged_file = self.out_dir / f"tmp_{self.date}.nc"
            if merged_file.exists():
                size = merged_file.stat().st_size / (1024**2)
                print(f"   ✅ {merged_file.name} ({size:.1f} MB)")
            else:
                print("   ⚠️  Merged file not created")
            
            # Cleanup only after successful processing
            self.cleanup_grib_files()
            
            return True
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user. Keeping temporary files for recovery.")
            return False
        except Exception as e:
            print(f"\n❌ Fatal error: {e}")
            print("⚠️  Temporary files kept for debugging")
            return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/ecmwf_hres_downloader.py <YYYYMMDD> [VAR1 VAR2 ...]")
        print("Example: python scripts/ecmwf_hres_downloader.py 20250615")
        print("         python scripts/ecmwf_hres_downloader.py 20250615 tp 2t msl")
        sys.exit(1)
    
    date = sys.argv[1]
    
    # Default variables if none specified
    if len(sys.argv) < 3:
        variables = ['tp', '2t', '2d', 'ssr', 'str', 'sp']
    else:
        variables = sys.argv[2:]
    
    downloader = ECMWFDownloader(date, variables)
    success = downloader.run()
    sys.exit(0 if success else 1)
