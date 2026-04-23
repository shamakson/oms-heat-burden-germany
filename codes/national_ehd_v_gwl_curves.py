#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
- Reads daily TG from NetCDF (E-OBS style; gridded or already national-mean)
- Optional mask to Germany (supply polygon or auto-download GISCO NUTS0 DE)
- Area-weight by cos(lat) -> national daily TG series
- Build fixed JJA P95 (1991–2020), compute JJA seasonal mean anomalies vs baseline
- Fit β with OLS + HAC (Newey–West)
- Weather-preserving mean shift across a GWL grid
- Recompute national EHD and propagate β-uncertainty via Monte Carlo
- Plot national EHD vs GWL with ribbons + markers (PI / Obs / +1.5 / +2.0)

Usage:
  python oms_ehd_gwl_cli.py \
    --nc EOBS_tg_daily_1990_2024.nc --var tg \
    --gmst-csv HadCRUT.5.0.2.0.analysis.summary_series.global.monthly.csv \
    --auto-de-mask \
    --summers 2018,2019,2022 \
    --out-prefix national_ehd_vs_gwl
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# Optional: only needed if masking
try:
    import geopandas as gpd
    from shapely.geometry import Point
except Exception:
    gpd = None

import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac  # (not used directly, kept for reference)


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="OMS national EHD vs GWL from daily TG NetCDF (β-uncertainty ribbons).")
    p.add_argument("--nc", required=True, help="Path to daily TG NetCDF (e.g., E-OBS).")
    p.add_argument("--var", default="tg", help="Variable name in NetCDF (default: tg).")
    p.add_argument("--gmst-csv", required=True, help="HadCRUT monthly CSV or simple year,gmst CSV.")
    p.add_argument("--out-prefix", default="national_ehd_vs_gwl", help="Output filename prefix.")
    p.add_argument("--baseline", default="1991-2020", help="Baseline years for anomalies/P95, e.g., 1991-2020.")
    p.add_argument("--summers", default="2018,2019,2022", help="Comma list of summers to plot.")
    p.add_argument("--gwl-grid", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0",
                   help="Comma list of GWL points (°C above PI).")
    p.add_argument("--hac-lag", type=int, default=3, help="Newey–West lag for HAC SE (default: 3).")
    p.add_argument("--beta-samples", type=int, default=2000, help="Monte Carlo samples for β (default: 2000).")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--assume-areamean", action="store_true",
                   help="Treat input as already national mean (skip spatial averaging).")
    p.add_argument("--mask-geo", default=None, help="Polygon file (GeoJSON/GeoPackage/Shapefile) to mask (e.g., Germany).")
    p.add_argument("--mask-layer", default=None, help="Layer name for multi-layer files (optional).")
    p.add_argument("--auto-de-mask", action="store_true", help="Auto-download GISCO NUTS0 polygon for Germany.")
    return p.parse_args()


# ---------------------------
# GMST loader (mirrors Fig. 3 style, plus simple CSV fallback)
# ---------------------------
def load_gmst_annual(csv_path: str) -> pd.Series:
    """
    (a) Simple CSV with columns: year, gmst  → returns Series index=year, name='gmst'
    (b) HadCRUT monthly summary CSV: find 'Time' & 'Anomaly (deg C)', average to annual
    """
    df = pd.read_csv(csv_path)

    # Case (a): simple year,gmst
    year_cols = [c for c in df.columns if c.strip().lower() == "year"]
    gmst_like = [c for c in df.columns if c.strip().lower() in ("gmst", "anomaly", "anom")]
    if year_cols and len(year_cols) == 1 and gmst_like:
        ycol = year_cols[0]
        gcol = None
        for c in gmst_like:
            if pd.api.types.is_numeric_dtype(df[c]): gcol = c; break
        if gcol is None:
            raise ValueError("Found 'year' but no numeric GMST-like column.")
        gmst = (df[[ycol, gcol]].dropna().astype({ycol: int})
                .set_index(ycol)[gcol].sort_index())
        gmst.name = "gmst"
        return gmst

    # Case (b): HadCRUT monthly → annual mean
    time_col = "Time" if "Time" in df.columns else df.columns[0]
    anom_col = [c for c in df.columns if ("Anomaly" in c and "deg" in c)]
    if not anom_col:
        # last resort: pick any numeric col not time
        anom_col = [c for c in df.columns if c != time_col and pd.api.types.is_numeric_dtype(df[c])]
    if not anom_col:
        raise ValueError(f"Cannot infer anomaly column from columns={list(df.columns)}")
    anom_col = anom_col[0]

    years = pd.to_datetime(df[time_col].astype(str), errors="coerce", format="%Y-%m").dt.year
    if years.isna().all():
        years = pd.to_datetime(df[time_col].astype(str), errors="coerce").dt.year
    ok = ~years.isna()
    if not ok.any():
        raise ValueError("Could not parse years from GMST CSV time column.")

    df = df.loc[ok, [anom_col]].copy()
    df["year"] = years.loc[ok].astype(int)
    gmst_ann = df.groupby("year")[anom_col].mean().sort_index()
    gmst_ann.name = "gmst"
    return gmst_ann

# ---------------------------
# Geo helpers (optional)
# ---------------------------
def load_mask_from_file(path, layer=None):
    if gpd is None:
        raise RuntimeError("geopandas is required for masking; install geopandas/shapely/pyproj.")
    gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    if gdf.crs is None: gdf = gdf.set_crs(4326)
    else: gdf = gdf.to_crs(4326)
    return gdf.unary_union

def auto_germany_polygon():
    if gpd is None:
        raise RuntimeError("geopandas is required for auto mask.")
    base = "https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson"
    url = f"{base}/NUTS_RG_01M_2021_4326_LEVL_0.geojson"
    gdf = gpd.read_file(url)
    return gdf[gdf["CNTR_CODE"] == "DE"].to_crs(4326).unary_union


# ---------------------------
# Data helpers (match Fig. 3 handling)
# ---------------------------
def jja_mask(idx: pd.DatetimeIndex) -> np.ndarray:
    return idx.month.isin([6, 7, 8])

def spatial_mean_da(da: xr.DataArray, mask_geom=None) -> xr.DataArray:
    """Area-weighted mean over (lat, lon), optional polygon mask; weights ~ cos(lat)."""
    lat_name = "lat" if "lat" in da.coords else ("latitude" if "latitude" in da.coords else None)
    lon_name = "lon" if "lon" in da.coords else ("longitude" if "longitude" in da.coords else None)
    if not lat_name or not lon_name:
        raise ValueError("Could not find lat/lon in dataset.")

    if (mask_geom is not None) and (gpd is not None):
        lats = da[lat_name].values
        lons = da[lon_name].values
        lon2d, lat2d = np.meshgrid(lons, lats)
        pts = gpd.GeoSeries([Point(float(x), float(y)) for x, y in zip(lon2d.ravel(), lat2d.ravel())], crs="EPSG:4326")
        inside = pts.within(mask_geom).to_numpy().reshape(lat2d.shape)
        da = da.where(xr.DataArray(inside, dims=(lat_name, lon_name)), drop=True)

    lats = da[lat_name].values
    w = np.cos(np.deg2rad(lats))
    w = w / np.nanmean(w)
    w_da = xr.DataArray(w, coords={lat_name: da.coords[lat_name]}, dims=(lat_name,))
    w_da = w_da.broadcast_like(da.isel({lat_name: slice(None), lon_name: slice(None)}))

    num = (da * w_da).sum(dim=(lat_name, lon_name), skipna=True)
    den = (w_da.where(~np.isnan(da))).sum(dim=(lat_name, lon_name), skipna=True)
    return num / den

def compute_fixed_p95_jja(ts: pd.Series, baseline_years) -> float:
    """Fixed national JJA P95 (mirrors Fig. 3: use J=JJA; baseline slice within J)."""
    ts = ts.sort_index()
    J = ts[jja_mask(ts.index)]
    base = J[(J.index.year >= baseline_years[0]) & (J.index.year <= baseline_years[1])]
    if base.empty:
        raise ValueError("Baseline window has no JJA data.")
    return float(base.quantile(0.95))

def jja_seasonal_mean_anomalies(ts: pd.Series, baseline_years):
    """
    JJA seasonal mean anomalies vs baseline JJA mean.
    Mirrors Fig. 3 flow: J=JJA; baseline as J within years; anomalies = yearly JJA mean minus baseline JJA mean.
    """
    ts = ts.sort_index()
    J = ts[jja_mask(ts.index)].dropna()

    # Yearly JJA means (safe grouping: group on J, not on original ts)
    y_index = J.index.year
    jja_yearly = J.groupby(y_index).mean()

    # Baseline climatological JJA mean (mean of yearly means in baseline)
    base_years_mask = (jja_yearly.index >= baseline_years[0]) & (jja_yearly.index <= baseline_years[1])
    base_slice = jja_yearly.loc[base_years_mask]
    if base_slice.empty:
        raise ValueError("Baseline window has no yearly JJA means.")
    clim = float(base_slice.mean())

    return (jja_yearly - clim).astype(float).squeeze()  # Series indexed by year


# ---------------------------
# Regression & translation
# ---------------------------
def fit_beta_hac(A_series: pd.Series, G_series: pd.Series, hac_lag: int):
    """
    OLS of A on G with HAC (Newey–West) SE and 95% CI.
    Uses statsmodels' robustcov_results to avoid se_cov misuse.
    """
    df = pd.DataFrame({"A": A_series, "G": G_series}).dropna()
    X = sm.add_constant(df["G"].values)
    y = df["A"].values

    ols = sm.OLS(y, X).fit()
    rob = ols.get_robustcov_results(cov_type="HAC", maxlags=hac_lag)

    beta_hat = float(rob.params[1])
    se_beta  = float(rob.bse[1])
    ci_lo, ci_hi = [float(v) for v in rob.conf_int(alpha=0.05)[1]]

    return {
        "alpha_hat": float(rob.params[0]),
        "beta_hat": beta_hat,
        "se_beta": se_beta,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "R2": float(ols.rsquared),  # R^2 from OLS fit (robust R^2 isn’t defined)
        "n": int(df.shape[0]),
    }

def translate_series(ts_jja_year: pd.Series, beta: float, gmst_of_year: float, target_gwl: float) -> pd.Series:
    """Weather-preserving shift: add beta * (target_gwl - gmst_of_year) to each daily value."""
    delta = beta * (target_gwl - gmst_of_year)
    return ts_jja_year + delta


# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()
    summers = [int(s.strip()) for s in args.summers.split(",") if s.strip()]
    gwl_grid = np.array([float(x) for x in args.gwl_grid.split(",")])
    b0, b1 = [int(v) for v in args.baseline.split("-")]
    baseline_years = (b0, b1)

    # GMST (annual)
    gmst = load_gmst_annual(args.gmst_csv)               # Series index=year

    # NetCDF → daily national TG (°C)
    ds = xr.open_dataset(args.nc)
    if args.var not in ds.variables:
        raise ValueError(f"Variable {args.var} not in dataset. Available: {list(ds.data_vars)}")
    if "time" in ds.coords and not np.issubdtype(ds["time"].dtype, np.datetime64):
        ds = xr.decode_cf(ds)
    da = ds[args.var]

    if args.assume_areamean:
        da_ts = da
    else:
        mask_geom = None
        if args.auto_de_mask:
            mask_geom = auto_germany_polygon()
        elif args.mask_geo:
            mask_geom = load_mask_from_file(args.mask_geo, args.mask_layer)
        da_ts = spatial_mean_da(da, mask_geom=mask_geom)

    time = pd.to_datetime(da_ts["time"].values)
    ts = pd.Series(da_ts.values.astype(float), index=time).sort_index()
    ts = ts[~ts.index.duplicated(keep="first")]  # guard

    # Fixed threshold & seasonal means (Fig. 3-style handling)
    p95 = compute_fixed_p95_jja(ts, baseline_years)
    A_y = jja_seasonal_mean_anomalies(ts, baseline_years)
    G_y = gmst.reindex(A_y.index)  # align by year

    # Fit β (HAC)
    fit = fit_beta_hac(A_y, G_y, args.hac_lag)
    beta_hat, se_beta = fit["beta_hat"], fit["se_beta"]
    print(f"β̂ = {beta_hat:.2f} °C/°C (95% HAC CI: {fit['ci_lo']:.2f}, {fit['ci_hi']:.2f}); R²={fit['R2']:.2f}; n={fit['n']}")

    # Monte Carlo β samples
    rng = np.random.default_rng(args.seed)
    beta_samples = rng.normal(loc=beta_hat, scale=se_beta, size=args.beta_samples)

    # Build curves per summer
    records = []
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for y in summers:
        # JJA daily for that summer
        J = ts[(ts.index.year == y) & jja_mask(ts.index)]
        if J.empty or (y not in gmst.index):
            print(f"Warning: missing data for {y}")
            continue
        gmst_y = float(gmst.loc[y])

        # Central curve
        y_c = []
        for g in gwl_grid:
            shifted = translate_series(J, beta_hat, gmst_y, g)
            y_c.append(float(np.maximum(shifted.values - p95, 0.0).sum()))
        y_c = np.array(y_c)

        # Ribbon via β MC
        y_samps = np.zeros((args.beta_samples, len(gwl_grid)))
        for i, b in enumerate(beta_samples):
            for j, g in enumerate(gwl_grid):
                shifted = translate_series(J, b, gmst_y, g)
                y_samps[i, j] = float(np.maximum(shifted.values - p95, 0.0).sum())
        y_lo = np.percentile(y_samps, 2.5, axis=0)
        y_hi = np.percentile(y_samps, 97.5, axis=0)

        # Plot (thicker lines + more opaque ribbons so they read under print)
        ax.plot(gwl_grid, y_c, linewidth=2.2, label=str(y), zorder=3)
        ax.fill_between(gwl_grid, y_lo, y_hi, alpha=0.18, zorder=2)

        # Markers at PI, observed, +1.5, +2.0 (larger, white face, black edge)
        for xm in [0.0, gmst_y, 1.5, 2.0]:
            ym = float(np.maximum(translate_series(J, beta_hat, gmst_y, xm).values - p95, 0.0).sum())
            ax.plot([xm], [ym], marker="o", markersize=6,
                    markerfacecolor="white", markeredgecolor="black", lw=0.0, zorder=4)

        # Save numeric outputs
        for g, yc, yl, yh in zip(gwl_grid, y_c, y_lo, y_hi):
            records.append({"year": y, "gwl": g, "ehd_central": yc, "ehd_lo": yl, "ehd_hi": yh})
        # Observed point explicitly
        records.append({"year": y, "gwl": gmst_y,
                        "ehd_central": float(np.maximum(J.values - p95, 0.0).sum()),
                        "ehd_lo": np.nan, "ehd_hi": np.nan})

    # Policy level guides at 1.5°C and 2.0°C
    for xv in [1.5, 2.0]:
        ax.axvline(xv, lw=0.8, linestyle="--", color="0.7", zorder=1)

    # Style & save
    ax.set_xlabel("Global-warming level (°C above pre-industrial)")
    ax.set_ylabel("National EHD (°C·days)")
    ax.set_xlim(gwl_grid.min(), gwl_grid.max())
    ax.margins(y=0.12)
    ax.grid(True, linewidth=0.5, alpha=0.3)
    leg = ax.legend(title="Summers", ncol=min(3, len(summers)),
                    loc="upper center", bbox_to_anchor=(0.5, 1.02),
                    frameon=True, framealpha=1.0)
    leg.get_frame().set_linewidth(0.8)
    ax.text(0.03, 0.95, r"Shaded bands: 95% from $\beta$ (HAC)",
            transform=ax.transAxes, ha="left", va="top", fontsize=9)

    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}.pdf", bbox_inches="tight")
    fig.savefig(f"{args.out_prefix}.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {args.out_prefix}.pdf/.png")

    pd.DataFrame.from_records(records).to_csv(f"{args.out_prefix}_curves.csv", index=False)
    print(f"Saved: {args.out_prefix}_curves.csv")


if __name__ == "__main__":
    main()
