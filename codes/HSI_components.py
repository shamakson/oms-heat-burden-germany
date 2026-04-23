#!/usr/bin/env python3
"""
Figure 1: HSI components for Germany-mean TG (2004–2024)
HSI components (Germany-mean TG):
  • Mean anomaly (JJA vs 1991–2020)
  • Frequency = # JJA days > P95 (P95 from 1991–2020)
  • Magnitude = mean exceedance depth on exceedance days

Default study years: 2001–2024.

Usage
  python figure1_hsi_germany_panels.py \
      --eobs tg_ens_mean_0.1deg_reg_v31.0e.nc \
      --save figures/fig1_HSI.png

Deps: xarray netCDF4 geopandas shapely pandas numpy matplotlib
"""

import argparse
import warnings
from typing import Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import shapely.geometry as sgeom
import shapely.ops as sops
import xarray as xr

# ----------------- Config -----------------
BASELINE_YEARS = (1991, 2020)           # for P95 + mean
STUDY_YEARS    = (2004, 2024)           # inclusive
HOT_LABELS     = {2018, 2019, 2022}
NUTS_LEVEL     = 1
NUTS_YEAR      = 2021
NUTS_RES       = "01M"
FIGSIZE        = (14.5, 7.6)
DPI            = 300
# ------------------------------------------

def read_de_nuts_from_web(level:int=1, year:int=2021, resolution:str="01M") -> gpd.GeoDataFrame:
    base = "https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson"
    url  = f"{base}/NUTS_RG_{resolution}_{year}_4326_LEVL_{level}.geojson"
    gdf  = gpd.read_file(url)
    if gdf.crs is None:
        gdf.set_crs(4326, inplace=True)
    gdf = gdf.to_crs(4326)
    gdf = gdf[gdf["CNTR_CODE"]=="DE"].copy()
    name_col = "NAME_LATN" if "NAME_LATN" in gdf.columns else ("NAME_ENGL" if "NAME_ENGL" in gdf.columns else None)
    if name_col is None:
        raise ValueError("Name column not found in GISCO NUTS file.")
    gdf = gdf.rename(columns={name_col:"land"})
    return gdf[["land","geometry"]].reset_index(drop=True)

def standardize_lon1d(ds: xr.Dataset) -> xr.Dataset:
    if "lon" in ds.coords and float(ds.lon.max()) > 180:
        ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180).sortby("lon")
    return ds

def get_lonlat_arrays(ds: xr.Dataset, var: str):
    # Case A: 2D 'longitude'/'latitude' on (y,x)
    if "longitude" in ds.variables and "latitude" in ds.variables and ds["longitude"].ndim == 2:
        return ds["longitude"].values, ds["latitude"].values, ("y","x")
    # Case B: 1D 'lon'/'lat'
    if "lon" in ds.coords and "lat" in ds.coords and ds["lon"].ndim == 1:
        lon1d, lat1d = ds["lon"].values, ds["lat"].values
        lon2d, lat2d = np.meshgrid(lon1d, lat1d)
        return lon2d, lat2d, ("lat","lon")
    # Case C: 1D 'longitude'/'latitude'
    if "longitude" in ds.coords and "latitude" in ds.coords and ds["longitude"].ndim == 1:
        lon1d, lat1d = ds["longitude"].values, ds["latitude"].values
        lon2d, lat2d = np.meshgrid(lon1d, lat1d)
        return lon2d, lat2d, ("latitude","longitude")
    raise ValueError("Could not locate lon/lat coordinates in dataset.")

def _union_geoms(geoms):
    try:
        return sops.union_all(list(geoms))   # Shapely 2
    except AttributeError:
        return sops.unary_union(list(geoms)) # Shapely 1.x

def germany_mask_union_for_da(da: xr.DataArray, lon2d: np.ndarray, lat2d: np.ndarray, de: gpd.GeoDataFrame) -> xr.DataArray:
    union = _union_geoms(de.geometry.values)
    flat_lon = lon2d.ravel(); flat_lat = lat2d.ravel()
    pts = (sgeom.Point(float(flat_lon[k]), float(flat_lat[k])) for k in range(flat_lon.size))
    inside = np.fromiter((union.covers(pt) for pt in pts), dtype=bool, count=flat_lon.size).reshape(lon2d.shape)
    ydim, xdim = da.dims[-2], da.dims[-1]
    return xr.DataArray(inside, coords={ydim: da.coords[ydim], xdim: da.coords[xdim]}, dims=(ydim, xdim))

def germany_mean_daily(ds: xr.Dataset, var: str, mask: xr.DataArray, lat2d: np.ndarray) -> pd.Series:
    V = ds[var].where(mask)
    w = np.cos(np.deg2rad(lat2d))
    w_da = xr.DataArray(w, coords={V.dims[-2]: V.coords[V.dims[-2]], V.dims[-1]: V.coords[V.dims[-1]]}, dims=V.dims[-2:])
    w_da = w_da.where(mask)
    w_da = w_da / w_da.sum(dim=(V.dims[-2], V.dims[-1]), skipna=True)
    gm = (V * w_da).sum(dim=(V.dims[-2], V.dims[-1]), skipna=True)
    s = gm.to_series(); s.index = pd.to_datetime(s.index); s.name = var
    return s

def jja_mask(idx: pd.DatetimeIndex) -> np.ndarray:
    return idx.month.isin([6,7,8])

def compute_hsi_components(g_daily: pd.Series, baseline: Tuple[int,int], study_years: Tuple[int,int]):
    """Return per-year components + z-scores + HSI + ranks and stability."""
    J = g_daily[jja_mask(g_daily.index)].copy()
    df = J.to_frame("tg"); df["year"] = df.index.year

    base = df[(df.index.year>=baseline[0]) & (df.index.year<=baseline[1])]["tg"]
    if base.empty:
        warnings.warn(f"No data in baseline {baseline[0]}–{baseline[1]}; using available years.")
        base = df["tg"]
    p95 = base.quantile(0.95); base_mean = base.mean()

    y0, y1 = study_years
    df = df[(df["year"]>=y0) & (df["year"]<=y1)].copy()

    # (a) mean anomaly per year
    jja_mean = df.groupby("year")["tg"].mean().rename("mean_tg")
    anom = (jja_mean - base_mean).rename("anom_mean")

    # (b) exceedances vs P95
    df["exceed"] = (df["tg"] - p95).clip(lower=0.0)
    n_days = df.groupby("year")["exceed"].apply(lambda s: int((s > 0).sum())).rename("n_days_gt_p95")
    mean_exc = df.groupby("year")["exceed"].apply(lambda s: float(s[s>0].mean() if (s>0).any() else 0.0)).rename("mean_exceed")

    comp = pd.concat([anom, n_days, mean_exc], axis=1).reset_index().sort_values("year")

    # z-scores
    z = lambda x: (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0)>0 else 1.0)
    comp["z_anom"]   = z(comp["anom_mean"])
    comp["z_ndays"]  = z(comp["n_days_gt_p95"])
    comp["z_exceed"] = z(comp["mean_exceed"])
    comp["HSI"]      = comp[["z_anom","z_ndays","z_exceed"]].sum(axis=1)

    # ranks (lower number = hotter)
    comp["HSI_rank"] = comp["HSI"].rank(ascending=False, method="min").astype(int)

    # leave-one-out HSI and ranks
    comp["HSI_no_mean"]   = comp[["z_ndays","z_exceed"]].sum(axis=1)
    comp["HSI_no_ndays"]  = comp[["z_anom","z_exceed"]].sum(axis=1)
    comp["HSI_no_exceed"] = comp[["z_anom","z_ndays"]].sum(axis=1)
    for c in ["HSI_no_mean","HSI_no_ndays","HSI_no_exceed"]:
        comp[c+"_rank"] = comp[c].rank(ascending=False, method="min").astype(int)

    # stability metric
    comp["max_rank_shift"] = comp[["HSI_no_mean_rank","HSI_no_ndays_rank","HSI_no_exceed_rank"]].sub(
        comp["HSI_rank"], axis=0).abs().max(axis=1)

    meta = {"p95": float(p95), "base_mean": float(base_mean)}
    return comp, meta

def plot_panels(comp: pd.DataFrame, save_path: str, hot_labels=set(), use_scores_heatmap: bool=True):
    # --- layout: (a)(b) on top, (c) full-width bottom ---
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=FIGSIZE, constrained_layout=True)
    gs = GridSpec(2, 2, height_ratios=[1.0, 1.15], figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])   # (a)
    ax2 = fig.add_subplot(gs[0, 1])   # (b)
    ax3 = fig.add_subplot(gs[1, :])   # (c) spans both columns

    years = comp["year"].values

    # (a) JJA mean anomaly
    # Base time series
    ax1.plot(
        comp["year"], comp["anom_mean"],
        linestyle="-", linewidth=1.8, marker="o", markersize=3,
        color="0.3", zorder=1
    )

    # Highlight HOT_LABEL years with red stars
    for y in sorted(hot_labels):
        if y in set(comp["year"]):
            v = comp.loc[comp["year"] == y, "anom_mean"].values[0]
            ax1.plot(
                [y], [v],
                marker="*", markersize=11,
                color="red", markeredgecolor="blue", markeredgewidth=0.7,
                zorder=3
            )
            ax1.annotate(
                str(y), (y, v),
                xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=8, fontweight="bold", zorder=4
            )

    # Horizontal reference line
    ax1.axhline(0, linewidth=1, color="0.5", zorder=0)

    # Clean integer x-axis ticks
    xticks = np.arange(comp["year"].min(), comp["year"].max() + 1, 2)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels([str(int(x)) for x in xticks])
    # Control y-axis range for anomaly
    ax1.set_ylim(-1.0, 2.2)

    # Labels & styling
    ax1.set_xlim(comp["year"].min() - 0.5, comp["year"].max() + 0.5)
    ax1.set_title("(a) JJA mean anomaly (°C)", fontweight="bold")
    ax1.set_xlabel("Year", fontweight="bold")
    ax1.set_ylabel("Anomaly (°C)", fontweight="bold")
    ax1.grid(alpha=0.25, linewidth=0.8)

    # (b) Stacked z-scores (HSI components)
    width = 0.65
    ax2.bar(comp["year"], comp["z_anom"], width=width, label="Mean anomaly (z)")
    ax2.bar(comp["year"], comp["z_ndays"], bottom=comp["z_anom"], width=width, label="# days > P95 (z)")
    ax2.bar(comp["year"], comp["z_exceed"], bottom=comp["z_anom"]+comp["z_ndays"], width=width, label="Mean exceed > P95 (z)")
    ax2.set_title("(b) HSI components", fontweight="bold")
    ax2.set_xlabel("Year", fontweight="bold"); ax2.set_ylabel("Z-score sum", fontweight="bold")
    ax2.legend(frameon=False, fontsize=8)

    # (c) Heatmap — scores (warm→cool) by default
    rank_df = comp[["year",
                    "HSI","HSI_no_mean","HSI_no_ndays","HSI_no_exceed",
                    "HSI_rank","HSI_no_mean_rank","HSI_no_ndays_rank","HSI_no_exceed_rank"]].sort_values("year")

    if use_scores_heatmap:
        # matrix of HSI and leave-one-out HSI (scores, not ranks)
        M = np.vstack([
            rank_df["HSI"].values,
            rank_df["HSI_no_mean"].values,
            rank_df["HSI_no_ndays"].values,
            rank_df["HSI_no_exceed"].values
        ])
        # center colormap at 0 so blue=cool, red=warm
        norm = mpl.colors.TwoSlopeNorm(vcenter=0.0, vmin=np.nanmin(M), vmax=np.nanmax(M))
        im = ax3.imshow(M, aspect="auto", cmap="coolwarm", norm=norm)
        ax3.set_title("(c) HSI score stability", fontweight="bold")
        # annotate with rounded scores
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax3.text(j, i, f"{M[i,j]:.1f}", ha="center", va="center", fontsize=7, color="black")
        cbar = fig.colorbar(im, ax=ax3, orientation="vertical", pad=0.015, shrink=0.9)
        cbar.set_label("HSI (z-sum)", fontweight="bold")
    else:
        # ranks view (1 = hottest) with reversed palette
        M = np.vstack([
            rank_df["HSI_rank"].values,
            rank_df["HSI_no_mean_rank"].values,
            rank_df["HSI_no_ndays_rank"].values,
            rank_df["HSI_no_exceed_rank"].values
        ])
        im = ax3.imshow(M, aspect="auto", cmap="viridis_r")
        ax3.set_title("(c) HSI ranks (1 = hottest)", fontweight="bold")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax3.text(j, i, f"{int(M[i,j])}", ha="center", va="center", fontsize=7, color="white")
        cbar = fig.colorbar(im, ax=ax3, orientation="vertical", pad=0.015, shrink=0.9)
        cbar.set_label("Rank", fontweight="bold")

    ax3.set_yticks([0,1,2,3]); ax3.set_yticklabels(["All","Mean Anomaly","# of days > P95","Mean exceed > P95"], fontweight="bold")
    ax3.set_xticks(np.arange(len(rank_df["year"]))); ax3.set_xticklabels(rank_df["year"].tolist(), rotation=45)
    ax3.tick_params(axis='x', labelsize=8)
    ax3.set_xlabel("Year", fontweight="bold")
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    return fig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eobs", required=True, help="Path to E-OBS TG NetCDF (e.g., *_reg_v31.0e.nc)")
    ap.add_argument("--var", default="tg", help="Variable name (default: tg)")
    ap.add_argument("--save", default="figures/fig1_HSI_DE.png", help="Output image path")
    ap.add_argument("--years", default=f"{STUDY_YEARS[0]}-{STUDY_YEARS[1]}", help="Study years, e.g., 2001-2024")
    ap.add_argument("--heatmap", choices=["scores","ranks"], default="scores", help="Show HSI scores (coolwarm) or ranks (viridis_r)")
    args = ap.parse_args()

    # Germany mask
    de = read_de_nuts_from_web(level=NUTS_LEVEL, year=NUTS_YEAR, resolution=NUTS_RES)

    # Load E-OBS (keep var + coords)
    ds = xr.open_dataset(args.eobs)
    if args.var not in ds:
        raise KeyError(f"Variable '{args.var}' not found. Available: {list(ds.data_vars)}")
    keep = [args.var]
    for extra in ("longitude","latitude","lon","lat","time"):
        if extra in ds.variables or extra in ds.coords:
            keep.append(extra)
    ds = standardize_lon1d(ds[keep])

    # Time subset
    y0, y1 = [int(x) for x in args.years.split("-")]
    t0 = f"{min(BASELINE_YEARS[0], y0)}-01-01"
    t1 = f"{max(BASELINE_YEARS[1], y1)}-12-31"
    ds = ds.sel(time=slice(t0, t1))

    # Coords + mask
    lon2d, lat2d, _ = get_lonlat_arrays(ds, args.var)
    mask = germany_mask_union_for_da(ds[args.var], lon2d, lat2d, de)

    # Germany-mean daily TG
    g_daily = germany_mean_daily(ds, args.var, mask, lat2d)

    # Components + table
    comp, meta = compute_hsi_components(g_daily, BASELINE_YEARS, (y0, y1))

    # Plot (scores heatmap by default)
    use_scores = (args.heatmap == "scores")
    _ = plot_panels(comp, args.save, hot_labels=HOT_LABELS, use_scores_heatmap=use_scores)

    # Save CSV of table
    out_csv = args.save.rsplit(".",1)[0] + "_table.csv"
    comp.to_csv(out_csv, index=False)

    # Objective selection (print to stdout)
    top3 = (comp.sort_values("HSI_rank")
                .query("max_rank_shift <= 5")
                .head(3)["year"].tolist())
    anchor = (comp.query("year < 2010")
                 .sort_values("HSI_rank")
                 .head(1)["year"].tolist())
    print(f"Saved: {args.save}")
    print(f"Saved: {out_csv}")
    print(f"Meta — baseline_mean={meta['base_mean']:.3f} °C, P95={meta['p95']:.3f} °C")
    print("Selected (top-3 with stability filter ≤5):", top3)
    print("Historical anchor (pre-2010 top):", anchor)

if __name__ == "__main__":
    main()
