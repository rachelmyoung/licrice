# licrice-standalone

A standalone package for the LICRICE tropical cyclone wind field model on historical IBTrACS tracks. Given a raw IBTrACS NetCDF (or Zarr) file plus a geographic region, it processes the tracks, estimates missing storm radii using RF, runs the wind fields model, and writes the output.


**Outputs:**
- `maxs` — maximum sustained wind speed (m/s) experienced at each grid cell during a storm
- `pddi` — power dissipation density index (m³/s²), the integral of wind speed cubed over time


## Setup

Requires Python 3.9 or greater

```bash
git clone https://github.com/<your-org>/licrice-standalone.git 
cd licrice-standalone
pip install -e .
```

Dependencies (`numpy`, `xarray`, `pandas`, `dask`, `zarr`, `scipy`, `scikit-learn`, `joblib`, `beautifulsoup4`, `tqdm`, `netCDF4`, `bottleneck` ) are declared in `pyproject.toml` and installed automatically.


## Data

Download IBTrACS (v04r00 or v04r01) from NOAA:

```bash
python - <<'EOF'
from licrice.io.ibtracs import download
download(
    url="https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/netcdf/",
    outdir="data/raw/",
)
EOF
```

Or manually from [NOAA IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive).

## Usage

### Run on IBTrACS tracks

```bash
# Standard run (recommended)
python run_licrice.py \
    --input data/raw/IBTrACS.ALL.v04r01.nc \
    --domain all \
    --outdir data/output/
    
# Run selected domains
python run_licrice.py \
    --input data/raw/IBTrACS.ALL.v04r01.nc \
    --domain north_atlantic_southwest western_pacific_south \
    --outdir data/output/

# Advanced options: If you already have preprocessed tracks
python run_licrice.py \
    --input data/raw/IBTrACS.ALL.v04r01.nc \
    --domain north_atlantic_southwest \
    --outdir data/output/ \
    --no-overwrite-preproc

# Advanced options: Use an already-preprocessed Zarr directly
python run_licrice.py \
    --input data/raw/ibtracs_preprocessed.zarr \
    --domain north_atlantic_southwest \
    --outdir data/output/
```

Each domain produces: `data/output/hazard_wind_licrice_hist_<domain>.zarr`

The first run processes the NetCDF (cleans tracks, trains RF radius models, saves to `ibtracs_preprocessed.zarr`). Only needs to be run once. Later runs with `--no-overwrite-preproc` skip straight to LICRICE.


### CLI flags

| Flag | Description |
|---|---|
| `--input` | Path to raw IBTrACS `.nc` or preprocessed `.zarr` |
| `--domain` | One or more domain names, or `all` |
| `--outdir` | Output directory |
| `--preproc-zarr` | Custom path for the preprocessed zarr (default: `<outdir>/ibtracs_preprocessed.zarr`) |
| `--no-overwrite-preproc` | Skip preprocessing if the zarr already exists |
| `--no-overwrite-output` | Skip LICRICE run if the output zarr already exists |
| `--params` | Path to a custom LICRICE params JSON (default: `params/licrice/v1.1.json`) |
| `--storm-chunksize` | Storms per processing chunk (default 25) |
| `--list-domains` | Print all built-in domain names and bounds, then exit |

### Available domains (from coastal-core-main)

| Domain | Description |
|---|---|
| `south_atlantic` | Southern coast of Brazil and Uruguay |
| `east_pacific_north` | Western Canada and Southern Alaska |
| `east_pacific_southwest` | Hawaii |
| `north_america_northeast` | Eastern Canada and Northeastern US |
| `north_atlantic_southwest` | Gulf of Mexico, Caribbean Sea |
| `north_america_south` | Mexico and Southwestern US |
| `north_america_west` | Western United States |
| `north_atlantic_east` | West Africa, Western Europe, Iceland |
| `western_pacific_south` | South China Sea, Philippine Sea |
| `western_pacific` | East China Sea, Sea of Japan |
| `western_pacific_north` | Sea of Okhotsk, Bering Sea |
| `south_pacific_central` | Western Polynesia |
| `south_pacific_northwest` | Northern Australia, Papua New Guinea |
| `south_pacific_southwest` | Southern Australia, New Zealand |
| `south_pacific_east` | Eastern Polynesia |
| `south_indian` | Southern Africa, South Indian Ocean islands |
| `north_indian` | Bay of Bengal, Arabian Sea |
| `conus` | Atlantic and Gulf coasts of the US |

## Output
Each domain produces one Zarr file at `<outdir>/hazard_wind_licrice_hist_<domain>.zarr` with dimensions `(storm, y_ix, x_ix)`.

Coordinates
- `lat (y_ix)` — latitude of grid cell centers
- `lon (x_ix)` — longitude of grid cell centers
- `storm (storm)` — IBTrACS storm ID
- `x_ix (x_ix)` — x-direction grid index
- `y_ix (y_ix)` — y-direction grid index

Data variables
- `maxs (storm, y_ix, x_ix)` — peak wind speed per grid cell per storm (m/s, float32)
- `pddi (storm, y_ix, x_ix)` — power dissipation density index per grid cell per storm (m³/s², float32)
- `start_date (storm)` — storm start datetime

Grid cell size is 0.1° (configurable via the params JSON).

## Code Structure

```
licrice-standalone/
├── run_licrice.py          # CLI entry point
├── run_example.py          # Synthetic storm demo
├── params/
│   └── licrice/v1.1.json   # Model parameters (vortex func, grid resolution, etc.)
├── licrice/
│   ├── io/
│   │   └── ibtracs.py      # IBTrACS ingestion and preprocessing
│   ├── tracks/
│   │   ├── radius.py       # RF-based RMW and ROCI estimation
│   │   ├── velocity.py     # Translational and circular velocity calculations
│   │   └── utils.py        # Track filtering and cleaning utilities
│   ├── licrice/
│   │   ├── run.py          # LICRICE execution loop
│   │   ├── preprocess.py   # Track → pixel step conversion, Zarr setup
│   │   ├── vortex_funcs.py # Parametric vortex models (modified Rankine, Holland 1980)
│   │   ├── dist_funcs.py   # Distance/angle grid construction
│   │   └── utils.py        # Wind field accumulation helpers
│   ├── spatial.py          # Great circle distances, grid index conversions
│   └── utilities.py        # smooth_fill, unit conversions, Zarr workflow checks
```

## Notes

This package is extracted from coastal-core-main and strips out the cloud I/O, GeoClaw surge model, damage estimation, and `Settings` configuration object. It is intended to run the IBTRACS to LICRICE pipeline locally without complicated dependencies.
