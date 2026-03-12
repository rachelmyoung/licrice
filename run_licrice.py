#!/usr/bin/env python
"""Command-line script to run LICRICE on IBTrACS

Usage ex.
# Run two domains, preprocessing the raw IBTrACS netCDF into a zarr
python run_licrice.py \\
    --input /path/to/IBTrACS.ALL.v04r01.nc \\
    --domain south_atlantic western_pacific_south \\
    --outdir /path/to/output/

# Run all domains
python run_licrice.py \\
    --input /path/to/IBTrACS.ALL.v04r01.nc \\
    --domain all \\
    --outdir /path/to/output/

# Skip preprocessing if we already have the zarr
python run_licrice.py \\
    --input /path/to/IBTrACS.ALL.v04r01.nc \\
    --domain south_atlantic \\
    --outdir /path/to/output/ \\
    --no-overwrite-preproc

# Use an already preprocessed zarr
python run_licrice.py \\
    --input /path/to/ibtracs_preprocessed.zarr \\
    --domain south_atlantic \\
    --outdir /path/to/output/
"""

import argparse
import json
import pathlib
import sys
import subprocess


# Domain definitions from pyTC.settings.Settings.GLOBAL_BBOXES and Settings.CONUS_BBOXES
DOMAINS = {
    "south_atlantic": {
        "long_name": "Western coasts of the South Atlantic",
        "description": "Southern Coast of Brazil and Uruguay",
        "xlim": [-60, -38],
        "ylim": [-35, -15],
    },
    "east_pacific_north": {
        "long_name": "Northern coasts of the Eastern Pacific",
        "description": "Western Canada and Southern Alaska",
        "xlim": [-180, -125],
        "ylim": [50, 64],
    },
    "east_pacific_southwest": {
        "long_name": "Southwestern islands of the Eastern Pacific",
        "description": "Hawaii and friends",
        "xlim": [-180, -154],
        "ylim": [0, 30],
    },
    "north_america_northeast": {
        "long_name": "Northeastern North America",
        "description": "Eastern Canada and the Northeastern United States",
        "xlim": [-90, -52],
        "ylim": [36, 64],
    },
    "north_atlantic_southwest": {
        "long_name": "Southwestern North Atlantic Ocean",
        "description": "Gulf of Mexico, Carribean Sea",
        "xlim": [-90, -52],
        "ylim": [5, 36],
    },
    "north_america_south": {
        "long_name": "Southern North America",
        "description": "Mexico and Southwestern United States",
        "xlim": [-123, -90],
        "ylim": [8, 36],
    },
    "north_america_west": {
        "long_name": "Western North America",
        "description": "Western United States",
        "xlim": [-125, -90],
        "ylim": [36, 50],
    },
    "north_atlantic_east": {
        "long_name": "Northeastern Atlantic Ocean",
        "description": "West Africa, Western Europe, Iceland",
        "xlim": [-33, 2],
        "ylim": [6, 68],
    },
    "western_pacific_south": {
        "long_name": "Southwestern Pacific Ocean",
        "description": "Gulf of Thailand, South China Sea, Phillipine Sea",
        "xlim": [98, 174],
        "ylim": [0, 25],
    },
    "western_pacific": {
        "long_name": "Western Pacific Ocean",
        "description": (
            "East China Sea, Yellow Sea, Sea of Japan, Southern Sea of Okhotsk"
        ),
        "xlim": [98, 155],
        "ylim": [25, 50],
    },
    "western_pacific_north": {
        "long_name": "Northwestern Pacific Ocean",
        "description": "Sea of Okhotsk, Bering Sea",
        "xlim": [98, 180],
        "ylim": [50, 65],
    },
    "south_pacific_central": {
        "long_name": "Central South Pacific Ocean",
        "description": "Western Polynesian Island Nations (minus New Zealand)",
        "xlim": [-180, -170],
        "ylim": [-51, 0],
    },
    "south_pacific_northwest": {
        "long_name": "Northwestern South Pacific Ocean",
        "description": "Southern Indonesia, Papua New Guinea, Northern Australia",
        "xlim": [95, 180],
        "ylim": [-25, 0],
    },
    "south_pacific_southwest": {
        "long_name": "Southwestern South Pacific Ocean",
        "description": "Southern Australia and New Zealand",
        "xlim": [112, 180],
        "ylim": [-55, -25],
    },
    "south_pacific_east": {
        "long_name": "Eastern South Pacific Ocean",
        "description": "Eastern Polynesian Islands",
        "xlim": [-170, -107],
        "ylim": [-30, 0],
    },
    "south_indian": {
        "long_name": "South Indian Ocean",
        "description": "Southern Africa plus Islands in the South Indian Ocean",
        "xlim": [13, 78],
        "ylim": [-51, 0],
    },
    "north_indian": {
        "long_name": "North Indian Ocean",
        "description": "Bay of Bengal, Arabian Sea, Laccadive Sea",
        "xlim": [40, 98],
        "ylim": [0, 35],
    },
    "conus": {
        "long_name": "Atlantic and Gulf Coasts",
        "description": "Atlantic and Gulf Coasts",
        "xlim": [-125, -66],
        "ylim": [24, 50],
    },
}


# Output metadata matches coastal-core run-licrice-hist.ipynb

HISTORY = """NEEED TO CHANGE THIS"""

NOTES = "NEED TO CHANGE THIS"
AUTHOR = "NEED TO CHANGE THIS"
CONTACT = "NEED TO CHANGE THIS"


def load_params(params_path=None):
    if params_path is None:
        params_path = pathlib.Path(__file__).parent / "params" / "licrice" / "v1.1.json"
    with open(params_path) as f:
        return json.load(f)

def run_aggregation(outdir, domains, args):

    if not args.aggregate:
        return

    print("\nRunning aggregation pipeline...")

    schemes = args.agg_schemes
    if "all" in schemes:
        schemes = ["spatial", "population", "asset"]

    for scheme in schemes:

        cmd = [
            sys.executable,
            args.agg_script,
            "--zarr-dir",
            str(outdir),
            "--scheme",
            scheme,
            "--admin-file",
            args.admin_file,
            "--outroot",
            str(pathlib.Path(outdir) / "aggregated"),
        ]

        if scheme == "asset":
            if args.litpop_dir is None:
                raise ValueError("Asset weighting requires --litpop-dir")
            cmd += ["--litpop-dir", args.litpop_dir]

        if scheme == "population":
            if args.landscan_path is None:
                raise ValueError("Population weighting requires --landscan-path")
            cmd += ["--landscan-path", args.landscan_path]

        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    print("Aggregation complete.")

def main():
    parser = argparse.ArgumentParser(
        description="Run LICRICE wind field model on IBTrACS tracks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i", required="--list-domains" not in sys.argv,
        help=(
            "Path to input track file. Either a raw IBTrACS netCDF (.nc) or a "
            "pre-processed zarr directory (.zarr). If .nc is given, the file is "
            "preprocessed automatically before running LICRICE."
        ),
    )
    parser.add_argument(
        "--domain", "-d", nargs="+", required="--list-domains" not in sys.argv,
        metavar="DOMAIN",
        help=(
            "One or more domain names to process, or 'all' to run every built-in "
            f"domain. Built-in domains: {', '.join(DOMAINS)}."
        ),
    )
    parser.add_argument(
        "--outdir", "-o", required="--list-domains" not in sys.argv,
        help="Output directory. One zarr file per domain is written here.",
    )
    parser.add_argument(
        "--preproc-zarr", default=None,
        metavar="PATH",
        help=(
            "Where to save/load the preprocessed IBTrACS zarr when --input is a "
            ".nc file. Defaults to <outdir>/ibtracs_preprocessed.zarr."
        ),
    )
    parser.add_argument(
        "--no-overwrite-preproc", action="store_true",
        help="Skip preprocessing if the preprocessed zarr already exists.",
    )
    parser.add_argument(
        "--no-overwrite-output", action="store_true",
        help="Skip LICRICE run if the output zarr already exists.",
    )
    parser.add_argument(
        "--params", default=None,
        metavar="PATH",
        help="Path to LICRICE params JSON. Defaults to params/licrice/v1.1.json.",
    )
    parser.add_argument(
        "--storm-chunksize", type=int, default=25,
        help="Number of storms per processing chunk (default 25).",
    )
    parser.add_argument(
        "--list-domains", action="store_true",
        help="Print all built-in domain names and bounds, then exit.",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Run administrative aggregation after LICRICE finishes."
    )
    parser.add_argument(
        "--agg-schemes",
        nargs="+",
        default=["spatial"],
        choices=["spatial", "population", "asset", "all"],
        help="Aggregation weighting schemes to compute."
    )
    parser.add_argument(
        "--litpop-dir",
        default=None,
        help="Path to LitPop asset files (required for asset weighting)."
    )
    parser.add_argument(
        "--landscan-path",
        default=None,
        help="Path to static LandScan raster (required for population weighting)."
    )
    parser.add_argument(
        "--admin-file",
        default="data/raw/admin/gadm_410.gpkg",
        help="Path to GADM geopackage for administrative aggregation."
    )
    parser.add_argument(
        "--agg-script",
        default="licrice/aggregation/aggregate_storm_admin.py",
        help="Path to aggregation script."
    )

    args = parser.parse_args()

    if args.list_domains:
        print("Built-in domains:")
        for name, d in DOMAINS.items():
            print(f"  {name:30s}  xlim={d['xlim']}  ylim={d['ylim']}")
        sys.exit(0)

    
    # Resolve domains
    if len(args.domain) == 1 and args.domain[0].lower() == "all":
        selected_domains = list(DOMAINS.keys())
    else:
        selected_domains = args.domain
        unknown = [d for d in selected_domains if d not in DOMAINS]
        if unknown:
            parser.error(
                f"Unknown domain(s): {unknown}. "
                f"Use --list-domains to see available domains."
            )

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    params = load_params(args.params)

   
    # Resolve track input path (preprocess .nc to zarr if needed)
    input_path = pathlib.Path(args.input)

    if input_path.suffix == ".nc":
        preproc_zarr = (
            pathlib.Path(args.preproc_zarr)
            if args.preproc_zarr
            else outdir / "ibtracs_preprocessed.zarr"
        )
        overwrite_preproc = not args.no_overwrite_preproc

        from licrice.io.ibtracs import preprocess_ibtracs
        preprocess_ibtracs(  # NEW
            nc_path=input_path,  # NEW
            zarr_outpath=preproc_zarr,  # NEW
            overwrite=overwrite_preproc,  # NEW
        )  # NEW
        track_zarr = preproc_zarr
    elif input_path.suffix == ".zarr" or input_path.is_dir():
        track_zarr = input_path
    else:
        parser.error(f"--input must be a .nc file or a .zarr directory, got: {input_path}")

    
    # Find valid tracks for each domain, then run LICRICE
    import pandas as pd
    from licrice.licrice.preprocess import find_valid_tracks
    from licrice.licrice.run import run_licrice_on_trackset

    print(f"\nScanning tracks for valid storms across {len(selected_domains)} domain(s)...")
    bboxes = {d: DOMAINS[d] for d in selected_domains}
    valid_by_domain = find_valid_tracks(str(track_zarr), params, bboxes)

    if not valid_by_domain:
        print("No storms found intersecting any of the requested domains. Exiting.")
        sys.exit(0)

    for domain in selected_domains:
        if domain not in valid_by_domain:
            print(f"\n[{domain}] No storms intersect this domain — skipping.")
            continue

        info = valid_by_domain[domain]
        n_storms = len(info["valid_tracks"])
        print(f"\n[{domain}] {n_storms} storm(s) found.")

        outpath = outdir / f"hazard_wind_licrice_hist_{domain}.zarr"
        tmppath = outdir / f"_tmp_{domain}.zarr"
        checkfile = outdir / f"_check_{domain}.txt"

        if args.no_overwrite_output and outpath.exists():
            print(f"  Output already exists at {outpath} — skipping.")
            continue

        xlim = DOMAINS[domain]["xlim"]
        ylim = DOMAINS[domain]["ylim"]
        attr_dict = {
            "author": AUTHOR,
            "contact": CONTACT,
            "updated": pd.Timestamp.now(tz="US/Pacific").strftime("%c"),
            "method": f"`licrice` for the {domain} domain (xlim:{xlim}, ylim:{ylim})",
            "history": HISTORY,
            "notes": NOTES,
            "licrice_domain": domain,
            "basin_long": DOMAINS[domain]["long_name"],
            "basin_desc": DOMAINS[domain]["description"],
        }

        print(f"  Running LICRICE → {outpath}")
        result = run_licrice_on_trackset(
            ds_path=track_zarr,
            valid_storms=info["valid_tracks"],
            start_dates=info["start_dates"],
            params=params,
            xlim=xlim,
            ylim=ylim,
            outpath=outpath,
            tmppath=tmppath,
            checkfile_path=checkfile,
            attr_dict=attr_dict,
            storm_chunksize=args.storm_chunksize,
            trackset_type="ibtracs",
            overwrite=not args.no_overwrite_output,
        )

        if result == 0:
            print(f"  No storms produced non-zero wind in this domain.")
        else:
            print(f"  Done. {result} storm(s) written to {outpath}")

    print("\nAll domains complete.")

    # Runs the aggregation step
    run_aggregation(outdir, selected_domains, args)
    
    print("\nAggregation complete.")

if __name__ == "__main__":
    main()
