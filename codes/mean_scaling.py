#!/usr/bin/env python3
"""
Outputs:
  (a) β regression: JJA(DE) anomaly vs GMST with fit + stats
  (b) KDEs of daily TG for selected summers under PI, Observed, +1.5 °C, +2.0 °C

Usage
  python figure2_ab.py \
    --eobs tg_ens_mean_0.1deg_reg_v31.0e.nc \
    --gmst_csv HadCRUT.5.0.2.0.analysis.summary_series.global.monthly.csv \
    --summers 2018,2019,2022 \
    --save figures/fig2_ab.png
"""

import argparse
import warnings
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import shapely.geometry as sgeom
import shapely.ops as sops
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, linregress

# ---------- Config ----------
BASELINE_YEARS = (1991, 2020)
STUDY_YEARS    = (2004, 2024)
NUTS_LEVEL     = 1
NUTS_YEAR      = 2021
NUTS_RES       = "01M"
FIGSIZE        = (12.5, 5.8)
DPI            = 300
# ----------------------------

# ---------- GIS / grid helpers ----------
def read_de_nuts_from_web(level:int=1, year:int=2021, resolution:str="01M") -> gpd.GeoDataFrame:
    base = "https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson"
    url  = f"{base}/NUTS_RG_{resolution}_{year}_4326_LEVL_{level}.geojson"
    gdf  = gpd.read_file(url)
    if gdf.crs is None: gdf.set_crs(4326, inplace=True)
    gdf = gdf.to_crs(4326)
    gdf = gdf[gdf["CNTR_CODE"]=="DE"].copy()
    name_col = "NAME_LATN" if "NAME_LATN" in gdf.columns else ("NAME_ENGL" if "NAME_ENGL" in gdf.columns else None)
    if name_col is None: raise ValueError("Name column not found in GISCO NUTS file.")
    gdf = gdf.rename(columns={name_col:"land"})
    return gdf[["land","geometry"]].reset_index(drop=True)

def standardize_lon1d(ds: xr.Dataset) -> xr.Dataset:
    if "lon" in ds.coords and float(ds.lon.max()) > 180:
        ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180).sortby("lon")
    return ds

def get_lonlat_arrays(ds: xr.Dataset, var: str):
    if "longitude" in ds.variables and "latitude" in ds.variables and ds["longitude"].ndim == 2:
        return ds["longitude"].values, ds["latitude"].values, ("y","x")
    if "lon" in ds.coords and "lat" in ds.coords and ds["lon"].ndim == 1:
        lon1d, lat1d = ds["lon"].values, ds["lat"].values
        lon2d, lat2d = np.meshgrid(lon1d, lat1d)
        return lon2d, lat2d, ("lat","lon")
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

# ---------- Climate + stats ----------
def load_gmst_annual_from_hadcrut(csv_path: str) -> pd.Series:
    df = pd.read_csv(csv_path)
    if "Time" not in df or "Anomaly (deg C)" not in df:
        raise ValueError("HadCRUT CSV must contain 'Time' and 'Anomaly (deg C)'.")
    df["year"] = df["Time"].astype(str).str.split("-").str[0].astype(int)
    gmst_ann = df.groupby("year")["Anomaly (deg C)"].mean()
    gmst_ann.name = "GMST"
    return gmst_ann

def beta_regression(jja_anom_by_year: pd.Series, gmst_ann: pd.Series, years: Tuple[int,int]):
    y0,y1 = years
    s1 = jja_anom_by_year[(jja_anom_by_year.index>=y0)&(jja_anom_by_year.index<=y1)]
    s2 = gmst_ann[(gmst_ann.index>=y0)&(gmst_ann.index<=y1)]
    df = pd.concat([s1,s2], axis=1, join="inner").dropna()
    df.columns = ["JJA_DE","GMST"]
    if len(df)<5: raise RuntimeError("Not enough overlapping years for regression.")
    lr = linregress(df["GMST"].values, df["JJA_DE"].values)
    beta, se = lr.slope, lr.stderr
    ci = (beta - 1.96*se, beta + 1.96*se)
    return {"beta":beta,"beta_se":se,"beta_ci":ci,"intercept":lr.intercept,
            "r2":lr.rvalue**2,"n":len(df),"df_plot":df}

def build_states(Jy: pd.Series, gmst: float, beta: float):
    """Return dict of arrays for Obs, PI, 1.5°C, 2.0°C using given beta."""
    out = {"Obs": Jy.values.copy()}
    out["PI"] = Jy.values - beta * gmst
    out["1.5°C"] = Jy.values + beta * (1.5 - gmst)
    out["2.0°C"] = Jy.values + beta * (2.0 - gmst)
    return out

# ---------- Figure builder ----------
def build_figure2_ab(eobs_path: str, var: str, gmst_csv: str, summers: List[int], save_path: str):
    # Data
    ds = xr.open_dataset(eobs_path)
    if var not in ds: raise KeyError(f"{var} not found. Available: {list(ds.data_vars)}")
    keep=[var]
    for extra in ("longitude","latitude","lon","lat","time"):
        if extra in ds.variables or extra in ds.coords: keep.append(extra)
    ds = standardize_lon1d(ds[keep])
    ds = ds.sel(time=slice(f"{min(BASELINE_YEARS[0],STUDY_YEARS[0])}-01-01",
                           f"{max(BASELINE_YEARS[1],STUDY_YEARS[1])}-12-31"))

    # Germany mask
    nuts = gpd.read_file(f"https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_{NUTS_RES}_{NUTS_YEAR}_4326_LEVL_{NUTS_LEVEL}.geojson")
    nuts = nuts[nuts["CNTR_CODE"]=="DE"].to_crs(4326)
    union = _union_geoms(nuts.geometry.values)
    lon2d, lat2d, _ = get_lonlat_arrays(ds, var)
    flat_lon = lon2d.ravel(); flat_lat = lat2d.ravel()
    inside = np.fromiter((union.covers(sgeom.Point(float(flat_lon[k]), float(flat_lat[k])))
                          for k in range(flat_lon.size)),
                         dtype=bool, count=flat_lon.size).reshape(lon2d.shape)
    mask = xr.DataArray(inside, coords={ds[var].dims[-2]: ds[var].coords[ds[var].dims[-2]],
                                        ds[var].dims[-1]: ds[var].coords[ds[var].dims[-1]]},
                        dims=ds[var].dims[-2:])

    # Germany-mean daily TG
    g_daily = germany_mean_daily(ds, var, mask, lat2d)
    J = g_daily[jja_mask(g_daily.index)]

    # Baseline stats
    base = J[(J.index.year>=BASELINE_YEARS[0])&(J.index.year<=BASELINE_YEARS[1])]
    if base.empty:
        warnings.warn("Baseline window empty; using all JJA.")
        base = J
    base_mean = base.mean()

    # Annual JJA anomalies and GMST
    jja_mean_by_year = J.groupby(J.index.year).mean()
    jja_anom_by_year = jja_mean_by_year - base_mean
    gmst_ann = load_gmst_annual_from_hadcrut(gmst_csv)
    reg = beta_regression(jja_anom_by_year, gmst_ann, STUDY_YEARS)

    # ===== layout: ONLY (a) and (b) =====
    fig = plt.figure(figsize=FIGSIZE, constrained_layout=True)
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.0, 1.2])
    ax1 = fig.add_subplot(gs[0,0])                      # (a)
    sub = gs[0,1].subgridspec(3,1, hspace=0.10)         # (b)
    axb = [fig.add_subplot(sub[i,0]) for i in range(3)]

    # ===== (a) β regression =====
    dfp = reg["df_plot"]
    ax1.scatter(dfp["GMST"], dfp["JJA_DE"], s=28, label="Years (JJA mean Temperature)")
    xline = np.linspace(dfp["GMST"].min()-0.1, dfp["GMST"].max()+0.1, 100)
    yline = reg["intercept"] + reg["beta"] * xline
    ax1.plot(xline, yline, linewidth=2.0, label="Linear fit")
    ax1.annotate(
        f"β = {reg['beta']:.2f} °C/°C ({reg['beta_ci'][0]:.2f}–{reg['beta_ci'][1]:.2f}),  R² = {reg['r2']:.2f}, n = {reg['n']}",
        xy=(0.02,0.98), xycoords="axes fraction", va="top", ha="left",
        fontsize=9, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7")
    )
    #ax1.set_title("(a) Mean-state scaling β", fontweight="bold")
    ax1.set_title("(a)", fontweight="bold", loc="left")
    ax1.set_xlabel("Annual global mean temperature (GMST) anomaly (°C)", fontweight="bold")
    ax1.set_ylabel("JJA temperature anomaly (°C)", fontweight="bold")
    ax1.legend(frameon=False, fontsize=8)

    # ===== (b) KDEs under PI/Obs/+GWL  =====
    summers_shown = summers[:3]
    xs_all=[]; kde_max=0.0
    for i,y in enumerate(summers_shown):
        Jy = J[J.index.year==y]
        if Jy.empty: raise RuntimeError(f"No JJA data for {y}")
        gmst_y = float(gmst_ann.get(y, np.nan))
        if np.isnan(gmst_y): raise RuntimeError(f"GMST missing for {y}")
        states = build_states(Jy, gmst_y, reg["beta"])

        xs = np.linspace(min(base.min(), Jy.min())-0.5, max(base.max(), Jy.max())+0.5, 400)
        xs_all.append(xs)
        # KDEs (fallback-safe)
        def kde_safe(vals, xs):
            if np.std(vals) < 1e-6:
                # nearly constant — approximate with narrow gaussian around mean
                m = float(np.mean(vals)); s = 0.1
                return np.exp(-(xs-m)**2/(2*s*s)) / (np.sqrt(2*np.pi)*s)
            return gaussian_kde(vals)(xs)

        k_base = kde_safe(base.values, xs)
        k_obs  = kde_safe(states["Obs"], xs)
        k_pi   = kde_safe(states["PI"], xs)
        k_15   = kde_safe(states["1.5°C"], xs)
        k_20   = kde_safe(states["2.0°C"], xs)
        kde_max = max(kde_max, k_base.max(), k_obs.max(), k_pi.max(), k_15.max(), k_20.max())

        for data, color, lab in [(k_pi, "#3B82F6", "PI"),
                                 (k_obs,"#111111","Observed"),
                                 (k_15,"#F59E0B","+1.5 °C"),
                                 (k_20,"#DC2626","+2.0 °C")]:
            axb[i].fill_between(xs, 0, data, alpha=0.25, color=color, label=lab if i==1 else None)
            axb[i].plot(xs, data, color=color, linewidth=1.8)
        axb[i].plot(xs, k_base, color="#2563EB", linestyle="--", linewidth=1.4, alpha=0.9,
                    label="Baseline" if i==1 else None)
        axb[i].set_ylabel("Density", fontweight="bold")
        axb[i].set_title(f"{y}", loc="right")
        if i<2: axb[i].set_xticklabels([])

    x_min = min(min(x) for x in xs_all); x_max = max(max(x) for x in xs_all)
    for axi in axb:
        axi.set_xlim(x_min, x_max)
        axi.set_ylim(0, kde_max*1.05)
    axb[-1].set_xlabel("Daily JJA temperature (°C)", fontweight="bold")
    axb[1].legend(frameon=False, fontsize=8, ncol=1, loc="upper left")
    axb[0].set_title("(b)", fontweight="bold", loc="left", pad=2)

    # save
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    print(f"Saved: {save_path}")
    print(f"β = {reg['beta']:.3f} (95% CI {reg['beta_ci'][0]:.3f}–{reg['beta_ci'][1]:.3f}), SE={reg['beta_se']:.3f}, R²={reg['r2']:.2f}, n={reg['n']}")
    print(f"Baseline JJA mean (1991–2020) = {base_mean:.2f} °C")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eobs", required=True, help="Path to E-OBS TG NetCDF")
    ap.add_argument("--var", default="tg", help="Variable name (default: tg)")
    ap.add_argument("--gmst_csv", required=True, help="HadCRUT5 monthly CSV path")
    ap.add_argument("--summers", default="2018,2019,2022", help="Comma-separated years (up to 3)")
    ap.add_argument("--save", default="figures/fig2_ab.png", help="Output image path")
    args = ap.parse_args()
    summers = [int(x.strip()) for x in args.summers.split(",") if x.strip()]
    if len(summers)==0: raise ValueError("Provide at least one summer via --summers")
    build_figure2_ab(args.eobs, args.var, args.gmst_csv, summers, args.save)

if __name__ == "__main__":
    main()
