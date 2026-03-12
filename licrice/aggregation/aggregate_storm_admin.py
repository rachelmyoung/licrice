#!/usr/bin/env python
"""
Aggregate LICRICE storm wind fields to administrative units.

Currently supported:
    spatial      : area-weighted (unconditional mean)
    asset        : asset-weighted mean (LitPop)
    population   : population-weighted mean (static 2015 LandScan population)

Expected raw inputs:
    data/raw/admin/gadm_410.gpkg
    data/raw/LitPop_v1_2
    data/raw/landscan/...
    data/output/hazard/hazard_wind_licrice_hist_<domain>.zarr

Outputs:
    data/output/aggregated/spatial/
    data/output/aggregated/asset/
    data/output/aggregated/population/

Example filenames:
    storm_admin0_area_east_pacific_north_maxs.parquet
    storm_admin0_popw_east_pacific_north_maxs.parquet
    storm_admin0_assetw_east_pacific_north_maxs.parquet
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr
from pyproj import Transformer
from scipy import sparse

import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

PROJECTED_CRS = "EPSG:6933"
DEFAULT_ADMIN_FILE = Path("data/raw/admin/gadm_410.gpkg")
DEFAULT_OUTROOT = Path("data/output/aggregated")

SCHEME_TAG = {
    "spatial": "area",
    "asset": "assetw",
    "population": "popw",
}

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def safe_slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", s).strip("_").lower()


def discover_zarr_files(zarr_dir: str | Path) -> list[Path]:
    zarr_dir = Path(zarr_dir)
    return sorted(zarr_dir.glob("hazard_wind_licrice_hist_*.zarr"))


def extract_domain_name(path: str | Path) -> str:
    path = Path(path)
    name = path.stem
    return name.replace("hazard_wind_licrice_hist_", "")


def build_output_path(
    outroot: str | Path,
    scheme: str,
    admin_level: int,
    basin_slug: str,
    haz_var: str,
) -> Path:

    outroot = Path(outroot)

    if scheme == "asset":
        outdir = outroot / "asset"
    elif scheme == "population":
        outdir = outroot / "population"
    else:
        outdir = outroot / "spatial"

    outdir.mkdir(parents=True, exist_ok=True)

    tag = SCHEME_TAG[scheme]
    fname = f"storm_admin{admin_level}_{tag}_{basin_slug}_{haz_var}.parquet"

    return outdir / fname


def require_admin_file(admin_file: str | Path) -> Path:
    admin_file = Path(admin_file)
    if not admin_file.exists():
        raise FileNotFoundError(
            f"Admin file not found: {admin_file}\n"
            f"Expected something like data/raw/admin/gadm_410.gpkg"
        )
    return admin_file




# ---------------------------------------------------------------------
# GRID / OVERLAP HELPERS
# ---------------------------------------------------------------------

def _grid_edges_1d(vals: np.ndarray) -> np.ndarray:
    """
    Build 1D cell edges from 1D centers (monotone, equally spaced).
    Returns edges of length n+1.
    Works for ascending or descending.
    """
    v = np.asarray(vals, dtype="float64")
    if v.size < 2:
        raise ValueError("Need at least 2 grid points to infer spacing.")
    dv = float(v[1] - v[0])
    edges = np.concatenate([v - dv / 2.0, [v[-1] + dv / 2.0]])
    return edges


def _window_from_bounds_1d(
    centers: np.ndarray,
    edges: np.ndarray,
    vmin: float,
    vmax: float,
) -> tuple[int, int]:
    """
    Given monotone grid centers + edges, and [vmin,vmax] bounds,
    return (i0, i1) slice indices on centers that cover the bbox.
    i1 is exclusive. Handles ascending or descending centers.
    """
    c = np.asarray(centers, dtype="float64")
    asc = c[0] < c[-1]

    if asc:
        i0 = int(np.searchsorted(edges, vmin, side="right") - 1)
        i1 = int(np.searchsorted(edges, vmax, side="left"))
    else:
        c_rev = c[::-1]
        edges_rev = edges[::-1]
        j0 = int(np.searchsorted(edges_rev, vmin, side="right") - 1)
        j1 = int(np.searchsorted(edges_rev, vmax, side="left"))
        n = c.size
        i0 = n - j1
        i1 = n - j0

    n = c.size
    i0 = max(0, min(i0, n))
    i1 = max(0, min(i1, n))
    if i1 < i0:
        i0, i1 = i1, i0
    return i0, i1
    
def load_gadm_admin(
    admin_file: str | Path,
    level: int,
    projected_crs: str = PROJECTED_CRS,
    base_layer: str = "gadm_410",
) -> gpd.GeoDataFrame:
    admin_file = require_admin_file(admin_file)
    layers = fiona.listlayers(str(admin_file))
    if base_layer not in layers:
        raise ValueError(
            f"Layer {base_layer} not found in {admin_file}. Available layers: {layers}"
        )

    gdf = gpd.read_file(admin_file, layer=base_layer)

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs(projected_crs)

    if level == 0:
        keep_cols = [c for c in ["GID_0", "COUNTRY", "NAME_0", "geometry"] if c in gdf.columns]
        if "GID_0" not in gdf.columns:
            raise ValueError("GID_0 not found in GADM file.")
        out = gdf[keep_cols].drop_duplicates(subset=["GID_0"]).copy()

    elif level == 1:
        keep_cols = [c for c in ["GID_1", "NAME_1", "GID_0", "COUNTRY", "NAME_0", "geometry"] if c in gdf.columns]
        if "GID_1" not in gdf.columns:
            raise ValueError("GID_1 not found in GADM file.")
        out = gdf[gdf["GID_1"].notna()][keep_cols].drop_duplicates(subset=["GID_1"]).copy()

    elif level == 2:
        keep_cols = [c for c in ["GID_2", "NAME_2", "GID_1", "GID_0", "COUNTRY", "NAME_0", "geometry"] if c in gdf.columns]
        if "GID_2" not in gdf.columns:
            raise ValueError("GID_2 not found in GADM file.")
        out = gdf[gdf["GID_2"].notna()][keep_cols].drop_duplicates(subset=["GID_2"]).copy()

    else:
        raise ValueError(f"Unsupported level: {level}")

    return out.reset_index(drop=True)


def load_all_admin_layers(
    admin_file: str | Path,
    projected_crs: str = PROJECTED_CRS,
) -> dict[str, gpd.GeoDataFrame]:
    return {
        "admin0": load_gadm_admin(admin_file, 0, projected_crs=projected_crs),
        "admin1": load_gadm_admin(admin_file, 1, projected_crs=projected_crs),
        "admin2": load_gadm_admin(admin_file, 2, projected_crs=projected_crs),
    }

def admin_id_field(level_name: str) -> str:
    return {
        "admin0": "GID_0",
        "admin1": "GID_1",
        "admin2": "GID_2",
    }[level_name]


def build_uncond_area_share_matrix_lazy(
    poly_gdf: gpd.GeoDataFrame,
    id_field: str,
    lat_array: np.ndarray,
    lon_array: np.ndarray,
    projected_crs: str = PROJECTED_CRS,
    lonlat_crs: str = "EPSG:4326",
    fix_invalid: bool = True,
    verbose_every: int | None = None,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """
    Build unconditional area-share weights without constructing all cell polygons.

    For each polygon:
      1) use polygon bounds in lon/lat to pick a (lat, lon) index window
      2) construct only those cell boxes in lon/lat
      3) project those boxes to projected_crs
      4) intersect with polygon (in projected_crs) and compute weights

    Returns:
      W_share: (n_polys x n_cells) CSR with weights = area(intersection)/area(poly)
      poly_ids: polygon ids in row order
    """
    # Prepare polygons in projected CRS
    g = poly_gdf[[id_field, "geometry"]].copy()
    if getattr(g, "crs", None) is None:
        g = g.set_crs(projected_crs)
    elif str(g.crs) != str(projected_crs):
        g = g.to_crs(projected_crs)

    g = g[g.geometry.notnull()].copy()
    g = g[~g.geometry.is_empty].copy()

    if fix_invalid:
        try:
            g["geometry"] = g.geometry.make_valid()
        except Exception:
            g["geometry"] = g.geometry.buffer(0)

    g["poly_area"] = g.geometry.area
    g = g[g["poly_area"] > 0].copy()
    g = g.reset_index(drop=True)

    n_polys = len(g)

    # Also keep lon/lat bounds for windowing
    g_ll = g.to_crs(lonlat_crs)
    minx_ll = g_ll.geometry.bounds.minx.to_numpy(dtype="float64")
    miny_ll = g_ll.geometry.bounds.miny.to_numpy(dtype="float64")
    maxx_ll = g_ll.geometry.bounds.maxx.to_numpy(dtype="float64")
    maxy_ll = g_ll.geometry.bounds.maxy.to_numpy(dtype="float64")

    # Grid bookkeeping
    lat = np.asarray(lat_array, dtype="float64")
    lon = np.asarray(lon_array, dtype="float64")
    nlat = lat.size
    nlon = lon.size
    n_cells = nlat * nlon

    lat_edges = _grid_edges_1d(lat)
    lon_edges = _grid_edges_1d(lon)

    transformer = Transformer.from_crs(lonlat_crs, projected_crs, always_xy=True)

    rows = []
    cols = []
    data = []

    poly_geom = g.geometry.to_numpy()
    poly_area = g["poly_area"].to_numpy(dtype="float64")

    for p in range(n_polys):
        if verbose_every and (p + 1) % verbose_every == 0:
            print(f"[weights] polygon {p+1:,}/{n_polys:,}")

        lon_min, lon_max = float(minx_ll[p]), float(maxx_ll[p])
        lat_min, lat_max = float(miny_ll[p]), float(maxy_ll[p])

        # Note: this does not handle dateline-straddling polygons specially.
        i0, i1 = _window_from_bounds_1d(lat, lat_edges, lat_min, lat_max)
        j0, j1 = _window_from_bounds_1d(lon, lon_edges, lon_min, lon_max)

        if (i1 - i0) <= 0 or (j1 - j0) <= 0:
            continue

        # Build only cell boxes in local window
        y0 = lat_edges[i0:i1]
        y1 = lat_edges[i0 + 1 : i1 + 1]
        x0 = lon_edges[j0:j1]
        x1 = lon_edges[j0 + 1 : j1 + 1]

        X0, Y0 = np.meshgrid(x0, y0)
        X1, Y1 = np.meshgrid(x1, y1)

        cell_boxes_ll = shapely.box(X0.ravel(), Y0.ravel(), X1.ravel(), Y1.ravel())

        # Project boxes for area intersections
        cell_boxes_prj = shapely.transform(
            cell_boxes_ll,
            transformer.transform,
            interleaved=False,
        )

        inter = shapely.intersection(poly_geom[p], cell_boxes_prj)
        inter_area = shapely.area(inter)

        m = inter_area > 0
        if not np.any(m):
            continue

        w = (inter_area[m] / poly_area[p]).astype(np.float32)

        ii = np.repeat(np.arange(i0, i1, dtype=np.int32), (j1 - j0))
        jj = np.tile(np.arange(j0, j1, dtype=np.int32), (i1 - i0))
        cell_id = (ii * nlon + jj).astype(np.int32)
        cell_id = cell_id[m]

        rows.append(np.full(cell_id.size, p, dtype=np.int32))
        cols.append(cell_id)
        data.append(w)

    if len(data) == 0:
        W_share = sparse.csr_matrix((n_polys, n_cells), dtype=np.float32)
    else:
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        data = np.concatenate(data)
        W_share = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(n_polys, n_cells),
            dtype=np.float32,
        )

    poly_ids = g[id_field].astype(str).to_numpy()
    return W_share, poly_ids


# ---------------------------------------------------------------------
# PLACEHOLDERS FOR FUTURE WEIGHTING SCHEMES
# ---------------------------------------------------------------------

def load_litpop_assets_to_wind_grid(
    litpop_dir: str | Path,
    lat_array: np.ndarray,
    lon_array: np.ndarray,
    pattern: str = "*.csv",
    recursive: bool = True,
    chunksize: int = 500_000,
) -> np.ndarray:
    """
    Aggregate high-resolution LitPop assets into the coarser wind grid
    using wind-cell edges.

    Safely handles both ascending and descending wind-grid coordinates.

    Returns:
        asset_grid with shape (nlat, nlon)
    """
    litpop_dir = Path(litpop_dir)
    files = list(litpop_dir.rglob(pattern)) if recursive else list(litpop_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No LitPop CSVs found in {litpop_dir}")

    lat = np.asarray(lat_array, dtype=np.float64)
    lon = np.asarray(lon_array, dtype=np.float64)

    nlat, nlon = len(lat), len(lon)
    grid = np.zeros((nlat, nlon), dtype=np.float32)

    # Build ascending centers/edges for search
    lat_desc = lat[0] > lat[-1]
    lon_desc = lon[0] > lon[-1]

    lat_search = lat[::-1] if lat_desc else lat
    lon_search = lon[::-1] if lon_desc else lon

    lat_edges = _grid_edges_1d(lat_search)
    lon_edges = _grid_edges_1d(lon_search)

    for f in files:
        for chunk in pd.read_csv(
            f,
            usecols=["value", "latitude", "longitude"],
            chunksize=chunksize,
        ):
            v = chunk["value"].to_numpy(dtype=np.float32, copy=False)
            la = chunk["latitude"].to_numpy(dtype=np.float64, copy=False)
            lo = chunk["longitude"].to_numpy(dtype=np.float64, copy=False)

            m = np.isfinite(v) & np.isfinite(la) & np.isfinite(lo) & (v != 0)
            if not np.any(m):
                continue

            v = v[m]
            la = la[m]
            lo = lo[m]

            i = np.searchsorted(lat_edges, la, side="right") - 1
            j = np.searchsorted(lon_edges, lo, side="right") - 1

            valid = (i >= 0) & (i < nlat) & (j >= 0) & (j < nlon)
            if not np.any(valid):
                continue

            i = i[valid]
            j = j[valid]
            v = v[valid]

            # Map back to original array orientation
            if lat_desc:
                i = (nlat - 1) - i
            if lon_desc:
                j = (nlon - 1) - j

            np.add.at(grid, (i, j), v)

    return grid

def build_asset_weight_matrix(
    W_share: sparse.csr_matrix,
    asset_grid: np.ndarray,
) -> sparse.csr_matrix:
    """
    Convert area-share matrix to row-normalized asset-weight matrix.

    W_asset[p, c] ∝ W_share[p, c] * asset[c]
    """
    asset_flat = np.asarray(asset_grid, dtype=np.float64).ravel()
    W = W_share.multiply(asset_flat)

    row_sums = np.asarray(W.sum(axis=1)).ravel()
    nonzero = row_sums > 0

    if np.any(nonzero):
        inv = np.zeros_like(row_sums, dtype=np.float64)
        inv[nonzero] = 1.0 / row_sums[nonzero]
        W = sparse.diags(inv) @ W

    return W.tocsr()

def discover_landscan_files(landscan_dir: str | Path) -> dict[int, Path]:
    """
    Discover annual LandScan rasters in a directory.

    Returns:
        dict mapping year -> raster path
    """
    landscan_dir = Path(landscan_dir)
    if not landscan_dir.exists():
        raise FileNotFoundError(f"LandScan directory not found: {landscan_dir}")

    candidates = list(landscan_dir.glob("*.tif")) + list(landscan_dir.glob("*.tiff"))
    if not candidates:
        raise FileNotFoundError(f"No .tif/.tiff files found in {landscan_dir}")

    out = {}
    for p in sorted(candidates):
        m = re.search(r"(19|20)\d{2}", p.name)
        if m:
            year = int(m.group(0))
            out[year] = p

    if not out:
        raise ValueError(
            f"Could not parse years from LandScan filenames in {landscan_dir}. "
            "Expected filenames containing a 4-digit year."
        )
    return out


def _grid_transform_from_centers(lat_array: np.ndarray, lon_array: np.ndarray):
    """
    Build rasterio affine transform for the LICRICE grid using cell-center coords.
    Assumes regular spacing.
    """
    lat = np.asarray(lat_array, dtype="float64")
    lon = np.asarray(lon_array, dtype="float64")

    if lat.size < 2 or lon.size < 2:
        raise ValueError("Need at least 2 lat/lon points to infer grid spacing.")

    dlat = float(lat[1] - lat[0])
    dlon = float(lon[1] - lon[0])

    west = lon.min() - abs(dlon) / 2.0
    east = lon.max() + abs(dlon) / 2.0
    south = lat.min() - abs(dlat) / 2.0
    north = lat.max() + abs(dlat) / 2.0

    return rasterio.transform.from_bounds(
        west=west,
        south=south,
        east=east,
        north=north,
        width=lon.size,
        height=lat.size,
    )


def load_landscan_to_wind_grid(
    landscan_path: str | Path,
    lat_array: np.ndarray,
    lon_array: np.ndarray,
    dst_crs: str = "EPSG:4326",
) -> np.ndarray:
    """
    Load one annual LandScan raster and resample/reproject it onto the LICRICE wind grid.

    Returns:
        pop_grid with shape (nlat, nlon)
    """
    landscan_path = Path(landscan_path)
    if not landscan_path.exists():
        raise FileNotFoundError(f"LandScan raster not found: {landscan_path}")

    dst_shape = (len(lat_array), len(lon_array))
    dst = np.zeros(dst_shape, dtype=np.float32)
    dst_transform = _grid_transform_from_centers(lat_array, lon_array)

    with rasterio.open(landscan_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            src_nodata=src.nodata,
            dst_nodata=0.0,
            resampling=Resampling.average,
        )

    dst = np.where(np.isfinite(dst), dst, 0.0).astype(np.float32, copy=False)
    dst[dst < 0] = 0.0
    return dst


def build_population_weight_matrix(
    W_share: sparse.csr_matrix,
    pop_grid: np.ndarray,
) -> sparse.csr_matrix:
    """
    Convert area-share matrix to row-normalized population-weight matrix.

    W_pop[p,c] ∝ W_share[p,c] * pop[c]
    """
    pop_flat = np.asarray(pop_grid, dtype=np.float64).ravel()
    W = W_share.multiply(pop_flat)

    row_sums = np.asarray(W.sum(axis=1)).ravel()
    nonzero = row_sums > 0

    if np.any(nonzero):
        inv = np.zeros_like(row_sums, dtype=np.float64)
        inv[nonzero] = 1.0 / row_sums[nonzero]
        W = sparse.diags(inv) @ W

    return W.tocsr()



# ---------------------------------------------------------------------
# CORE AGGREGATION
# ---------------------------------------------------------------------

def aggregate_one_basin(
    zarr_path: str | Path,
    scheme: str,
    outroot: str | Path,
    admin_gdfs: dict[str, gpd.GeoDataFrame],
    litpop_dir: str | Path | None = None,
    landscan_path: str | Path | None = None,
    haz_var: str = "maxs",
) -> None:
    zarr_path = Path(zarr_path)
    basin = extract_domain_name(zarr_path)
    basin_slug = safe_slug(basin)

    print(f"\nProcessing basin: {basin} [{scheme}]")

    ds = xr.open_zarr(zarr_path, consolidated=True)
    if haz_var not in ds:
        print(f"  [skip] {zarr_path.name}: '{haz_var}' not present.")
        return

    wind_da = ds[haz_var]
    lat_array = ds["lat"].values
    lon_array = ds["lon"].values

    asset_grid = None
    pop_grid = None

    if scheme == "asset":
        if litpop_dir is None:
            raise ValueError("Asset weighting requires --litpop-dir")
        asset_grid = load_litpop_assets_to_wind_grid(litpop_dir, lat_array, lon_array)

    elif scheme == "population":
        if landscan_path is None:
            raise ValueError("Population weighting requires --landscan-path")
        pop_grid = load_landscan_to_wind_grid(landscan_path, lat_array, lon_array)

    start_dates = pd.to_datetime(ds["start_date"].values) if "start_date" in ds else None
    storms = wind_da.sizes["storm"]

    for level_name, poly_gdf in admin_gdfs.items():
        id_field = admin_id_field(level_name)
        admin_level_num = int(level_name.replace("admin", ""))

        print(f"  building overlap weights for {level_name}")
        W_share, poly_ids = build_uncond_area_share_matrix_lazy(
            poly_gdf=poly_gdf,
            id_field=id_field,
            lat_array=lat_array,
            lon_array=lon_array,
            projected_crs=PROJECTED_CRS,
        )
        
        if scheme == "spatial":
            W = W_share
        elif scheme == "asset":
            W = build_asset_weight_matrix(W_share, asset_grid)
        elif scheme == "population":
            W = build_population_weight_matrix(W_share, pop_grid)
        else:
            raise ValueError(f"Unknown scheme: {scheme}")

        rows = []
        for s in range(storms):
            arr = wind_da.isel(storm=s).transpose("y_ix", "x_ix").values
            vflat = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32, copy=False).ravel()
            vals = np.asarray(W @ vflat).ravel()

            hit = np.where(vals > 0)[0]
            if hit.size == 0:
                continue

            for r in hit:
                row = {
                    "storm_idx": int(s),
                    "admin_id": str(poly_ids[r]),
                    "value": float(vals[r]),
                    "basin": basin,
                    "admin_level": level_name,
                    "weighting": scheme,
                    "haz_var": haz_var,
                }
                if start_dates is not None:
                    row["storm_date"] = pd.Timestamp(start_dates[s])
                    row["year"] = int(pd.Timestamp(start_dates[s]).year)
                    row["month"] = int(pd.Timestamp(start_dates[s]).month)
                rows.append(row)

        outpath = build_output_path(
            outroot=outroot,
            scheme=scheme,
            admin_level=admin_level_num,
            basin_slug=basin_slug,
            haz_var=haz_var,
        )
        pd.DataFrame(rows).to_parquet(outpath, index=False)
        print(f"  wrote {outpath}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--zarr-dir",
        required=True,
        help="Directory containing LICRICE hazard zarr outputs",
    )
    parser.add_argument(
        "--scheme",
        required=True,
        choices=["spatial", "asset", "population", "all"],
        help="Aggregation scheme to run, or 'all' to run all weighting schemes.",
    )
    parser.add_argument(
        "--admin-file",
        default=str(DEFAULT_ADMIN_FILE),
        help="Path to GADM geopackage (default: data/raw/admin/gadm_410.gpkg)",
    )
    parser.add_argument(
        "--litpop-dir",
        default=None,
        help="Path to LitPop directory; required for asset weighting",
    )
    parser.add_argument(
        "--outroot",
        default=str(DEFAULT_OUTROOT),
        help="Root output directory (default: data/output/aggregated)",
    )
    parser.add_argument(
        "--haz-var",
        default="maxs",
        help="Hazard variable in LICRICE zarr (default: maxs)",
    )
    parser.add_argument(
        "--landscan-path",
        default=None,
        help="Path to static LandScan raster (e.g. 2015); required for population weighting",
    )

    args = parser.parse_args()

    if args.scheme == "all":
        schemes = ["spatial", "asset", "population"]
    else:
        schemes = [args.scheme]

    if "asset" in schemes and args.litpop_dir is None:
        raise ValueError("Asset weighting requires --litpop-dir")

    if "population" in schemes and args.landscan_path is None:
        raise ValueError("Population weighting requires --landscan-path")

    zarr_files = discover_zarr_files(args.zarr_dir)
    if not zarr_files:
        print(f"No LICRICE zarr files found in {args.zarr_dir}")
        sys.exit(1)

    admin_gdfs = load_all_admin_layers(args.admin_file)

    for scheme in schemes:
        print(f"\n=== Running scheme: {scheme} ===")
        for zarr_path in zarr_files:
            aggregate_one_basin(
                zarr_path=zarr_path,
                scheme=scheme,
                outroot=args.outroot,
                admin_gdfs=admin_gdfs,
                litpop_dir=args.litpop_dir,
                landscan_path=args.landscan_path,
                haz_var=args.haz_var,
            )

    print("\nAggregation complete.")


if __name__ == "__main__":
    main()
