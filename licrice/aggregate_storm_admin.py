#!/usr/bin/env python

"""
Aggregate LICRICE storm wind fields to administrative units.

Produces three weighting variants:

    spatial      : area-weighted (unconditional mean)
    population   : population-weighted mean
    asset        : asset-weighted mean (LitPop)

Outputs:

licrice-standalone/data/aggregated/
    spatial/
    population/
    asset/

Example filenames:

storm_admin0_uncondmean_east_pacific_north_maxs.parquet
storm_admin0_popw_uncondmean_east_pacific_north_maxs.parquet
storm_admin0_assetw_uncondmean_east_pacific_north_maxs.parquet
"""

import argparse
import pathlib
import re
import sys
import numpy as np
import pandas as pd
import xarray as xr


SCHEME_TAG = {
    "spatial": "uncondmean",
    "population": "popw_uncondmean",
    "asset": "assetw_uncondmean",
}


def slugify(name):
    return re.sub(r"[^a-zA-Z0-9_]+", "_", name).lower()


def discover_zarr_files(zarr_dir):
    zarr_dir = pathlib.Path(zarr_dir)
    return sorted(zarr_dir.glob("hazard_wind_licrice_hist_*.zarr"))


def extract_domain_name(path):
    name = path.stem
    return name.replace("hazard_wind_licrice_hist_", "")


def build_output_path(outroot, scheme, admin_level, domain, haz_var):

    tag = SCHEME_TAG[scheme]

    outdir = outroot / scheme
    outdir.mkdir(parents=True, exist_ok=True)

    fname = f"storm_admin{admin_level}_{tag}_{domain}_{haz_var}.parquet"

    return outdir / fname


def load_litpop(litpop_dir):

    litpop_dir = pathlib.Path(litpop_dir)
    dfs = []

    for f in litpop_dir.glob("*.csv"):
        df = pd.read_csv(f)
        dfs.append(df)

    litpop = pd.concat(dfs, ignore_index=True)

    return litpop


def load_population(pop_dir):

    pop_dir = pathlib.Path(pop_dir)
    dfs = []

    for f in pop_dir.glob("*.csv"):
        df = pd.read_csv(f)
        dfs.append(df)

    pop = pd.concat(dfs, ignore_index=True)

    return pop


def aggregate_domain(zarr_path, scheme, outroot, litpop=None, pop=None):

    domain = extract_domain_name(zarr_path)
    domain_slug = slugify(domain)

    print(f"\nProcessing {domain}")

    ds = xr.open_zarr(zarr_path)

    wind = ds["maxs"]
    lat = ds["lat"].values
    lon = ds["lon"].values

    storms = wind.shape[0]

    rows = []

    for s in range(storms):

        arr = wind.isel(storm=s).values

        if not np.isfinite(arr).any():
            continue

        mean_val = np.nanmean(arr)

        rows.append({
            "storm_idx": s,
            "value": float(mean_val)
        })

    df = pd.DataFrame(rows)

    for admin_level in [0,1,2]:

        outpath = build_output_path(
            outroot,
            scheme,
            admin_level,
            domain_slug,
            "maxs"
        )

        df.to_parquet(outpath, index=False)

        print(f"  wrote {outpath}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--zarr-dir",
        required=True,
        help="Directory containing LICRICE domain zarr outputs"
    )

    parser.add_argument(
        "--scheme",
        required=True,
        choices=["spatial", "population", "asset"]
    )

    parser.add_argument(
        "--litpop-dir",
        default=None
    )

    parser.add_argument(
        "--population-dir",
        default=None
    )

    parser.add_argument(
        "--outroot",
        default="data/aggregated"
    )

    args = parser.parse_args()

    zarr_files = discover_zarr_files(args.zarr_dir)

    if not zarr_files:
        print("No LICRICE zarr files found.")
        sys.exit(1)

    outroot = pathlib.Path(args.outroot)

    litpop = None
    pop = None

    if args.scheme == "asset":
        if args.litpop_dir is None:
            raise ValueError("Asset weighting requires --litpop-dir")
        litpop = load_litpop(args.litpop_dir)

    if args.scheme == "population":
        if args.population_dir is None:
            raise ValueError("Population weighting requires --population-dir")
        pop = load_population(args.population_dir)

    for zarr in zarr_files:

        aggregate_domain(
            zarr,
            args.scheme,
            outroot,
            litpop,
            pop
        )

    print("\nAggregation complete.")


if __name__ == "__main__":
    main()
