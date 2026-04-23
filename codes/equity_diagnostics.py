#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Equity of anthropogenic heat burden across states (embedded populations)

(a) Paired dots: ΔEHD (EHD_Obs − EHD_PI) by Land for all summers, ordered by the first summer.
(b) Per-capita ΔEHD (°C·days per 100k) using embedded Destatis populations (2018, 2019, 2022).
(c) Lorenz curves of ΔEHD across Länder with Gini coefficients.
(d) Lorenz curves of ΔEHD across Länder with Gini coefficients per 100k inhabitants.

Usage
  python figure5_equity_embedded_pop.py \
    --eobs tg_ens_mean_0.1deg_reg_v31.0e.nc \
    --gmst_csv HadCRUT.5.0.2.0.analysis.summary_series.global.monthly.csv \
    --beta 2.63 \
    --summers 2018,2019,2022 \
    --save figures/fig5_equity.png
"""
import argparse
import warnings
from typing import List, Dict
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import shapely.geometry as sgeom
from shapely.prepared import prep
import matplotlib.pyplot as plt

# ----------------- Embedded populations (absolute persons) -----------------
# Order of states must match GISCO NUTS-1 names below (see note)
STATES = [
    "Baden-Württemberg","Bayern","Berlin","Brandenburg","Bremen","Hamburg","Hessen",
    "Mecklenburg-Vorpommern","Niedersachsen","Nordrhein-Westfalen","Rheinland-Pfalz",
    "Saarland","Sachsen","Sachsen-Anhalt","Schleswig-Holstein","Thüringen"
]

POP = {
    2018: {
        "Baden-Württemberg":11046479, "Bayern":13036963, "Berlin":3629161, "Brandenburg":2507979,
        "Bremen":682009, "Hamburg":1835882, "Hessen":6254536, "Mecklenburg-Vorpommern":1610397,
        "Niedersachsen":7972612, "Nordrhein-Westfalen":17922393, "Rheinland-Pfalz":4079262,
        "Saarland":992348, "Sachsen":4079623, "Sachsen-Anhalt":2215701,
        "Schleswig-Holstein":2893267, "Thüringen":2147175
    },
    2019: {
        "Baden-Württemberg":11084964, "Bayern":13100729, "Berlin":3657159, "Brandenburg":2516905,
        "Bremen":682094, "Hamburg":1844216, "Hessen":6276945, "Mecklenburg-Vorpommern":1608907,
        "Niedersachsen":7988028, "Nordrhein-Westfalen":17939936, "Rheinland-Pfalz":4089374,
        "Saarland":988698, "Sachsen":4074954, "Sachsen-Anhalt":2201552,
        "Schleswig-Holstein":2900243, "Thüringen":2138262
    },
    2022: {
        "Baden-Württemberg":11083914, "Bayern":13003714, "Berlin":3592264, "Brandenburg":2526558,
        "Bremen":691994, "Hamburg":1812573, "Hessen":6186950, "Mecklenburg-Vorpommern":1566726,
        "Niedersachsen":7922782, "Nordrhein-Westfalen":17855019, "Rheinland-Pfalz":4080449,
        "Saarland":1006591, "Sachsen":4026994, "Sachsen-Anhalt":2141421,
        "Schleswig-Holstein":2922542, "Thüringen":2108809
    }
}
# ---------------------------------------------------------------------------

# ---------- Config ----------
BASELINE_YEARS = (1991, 2020)
STUDY_YEARS    = (2001, 2024)
NUTS_LEVEL     = 1
NUTS_YEAR      = 2021
NUTS_RES       = "01M"

FIGSIZE        = (14.8, 9.2)
DPI            = 220
YEAR_COLORS    = {2018:"#1f77b4", 2019:"#ff7f0e", 2022:"#d62728"}
MARKERS        = {2018:"o", 2019:"s", 2022:"^"}

# ---------- Utilities ----------
def read_de_nuts_from_web(level:int=1, year:int=2021, resolution:str="01M") -> gpd.GeoDataFrame:
    base = "https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson"
    url  = f"{base}/NUTS_RG_{resolution}_{year}_4326_LEVL_{level}.geojson"
    gdf  = gpd.read_file(url)
    if gdf.crs is None: gdf.set_crs(4326, inplace=True)
    gdf = gdf[gdf["CNTR_CODE"]=="DE"].to_crs(4326)
    name_col = "NAME_LATN" if "NAME_LATN" in gdf.columns else ("NAME_ENGL" if "NAME_ENGL" in gdf.columns else None)
    if name_col is None: 
        # fallback: try common name columns
        for cand in ["NAME", "NUTS_NAME"]:
            if cand in gdf.columns:
                name_col = cand
                break
    if name_col is None:
        raise ValueError("Name column not found in GISCO NUTS file.")
    gdf = gdf.rename(columns={name_col:"land"})
    # Note: GISCO names may slightly differ from POP keys (e.g. "Mecklenburg-Vorpommern").
    # Ensure we return the geometry and name as-is; matching to POP relies on identical names.
    return gdf[["land","geometry"]].reset_index(drop=True)

def standardize_lon1d(ds: xr.Dataset) -> xr.Dataset:
    if "lon" in ds.coords and float(ds.lon.max()) > 180:
        ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180).sortby("lon")
    return ds

def get_lonlat_arrays(ds: xr.Dataset, var: str):
    # Handles several common variable/coordinate layouts
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
    raise ValueError("Could not locate lon/lat in dataset.")

def points_in_poly_mask(lon2d: np.ndarray, lat2d: np.ndarray, poly) -> np.ndarray:
    P = prep(poly)
    flat_lon = lon2d.ravel(); flat_lat = lat2d.ravel()
    inside = np.fromiter((P.covers(sgeom.Point(float(flat_lon[k]), float(flat_lat[k])))
                          for k in range(flat_lon.size)),
                         dtype=bool, count=flat_lon.size)
    return inside.reshape(lon2d.shape)

def area_weighted_mean_daily(da: xr.DataArray, mask2d: np.ndarray, lat2d: np.ndarray) -> pd.Series:
    # da expected dims: (time, lat, lon) or (time, y, x)
    V = da.where(mask2d)
    w = np.cos(np.deg2rad(lat2d))
    # build weight DataArray aligned with V's spatial dims
    spatial_dims = (V.dims[-2], V.dims[-1])
    w_da = xr.DataArray(w, coords={spatial_dims[0]: V.coords[spatial_dims[0]], spatial_dims[1]: V.coords[spatial_dims[1]]}, dims=spatial_dims)
    w_da = w_da.where(mask2d)
    w_da = w_da / w_da.sum(dim=spatial_dims, skipna=True)
    series = (V * w_da).sum(dim=spatial_dims, skipna=True).to_series()
    series.index = pd.to_datetime(series.index)
    return series

def jja_mask(idx: pd.DatetimeIndex) -> np.ndarray:
    return idx.month.isin([6,7,8])

def load_gmst_annual_from_hadcrut(csv_path: str) -> pd.Series:
    df = pd.read_csv(csv_path)
    time_col = "Time" if "Time" in df.columns else df.columns[0]
    # find an anomaly column containing 'Anomaly' and 'deg' (heuristic used in original script)
    anom_col = next((c for c in df.columns if "Anomaly" in c and "deg" in c), None)
    if anom_col is None:
        # fallback: try common names
        for cand in ["Anomaly (deg C)", "Anomaly (degC)", "anomaly", "anomaly_degC"]:
            if cand in df.columns:
                anom_col = cand
                break
    if anom_col is None:
        raise ValueError("HadCRUT CSV missing 'Anomaly (deg C)' column.")
    # Try to parse a monthly time column (YYYY-MM)
    years = pd.to_datetime(df[time_col].astype(str), errors="coerce", format="%Y-%m").dt.year
    if years.isna().all():
        # try yearly or other formats
        years = pd.to_datetime(df[time_col].astype(str), errors="coerce").dt.year
    df = df.loc[~years.isna()].copy()
    df["year"] = years.loc[~years.isna()].astype(int)
    gmst_ann = df.groupby("year")[anom_col].mean()
    gmst_ann.name = "GMST"
    return gmst_ann

def make_PI(J: pd.Series, gmst_y: float, beta: float) -> np.ndarray:
    return (J.values - beta * gmst_y)

def ehd(arr: np.ndarray, p95: float) -> float:
    exc = (arr - p95).clip(min=0.0)
    return float(exc.sum())

def compute_byland_dEHD(eobs_path: str, var: str, gmst_csv: str, beta: float,
                        summers: List[int]) -> pd.DataFrame:
    ds = xr.open_dataset(eobs_path)
    if var not in ds: 
        raise KeyError(f"{var} not found. Available: {list(ds.data_vars)}")
    keep=[var]
    for extra in ("longitude","latitude","lon","lat","time"):
        if extra in ds.variables or extra in ds.coords: keep.append(extra)
    ds = standardize_lon1d(ds[keep])
    ds = ds.sel(time=slice(f"{min(BASELINE_YEARS[0], STUDY_YEARS[0])}-01-01",
                           f"{max(BASELINE_YEARS[1], STUDY_YEARS[1])}-12-31"))
    lon2d, lat2d, _ = get_lonlat_arrays(ds, var)
    nuts = read_de_nuts_from_web(NUTS_LEVEL, NUTS_YEAR, NUTS_RES)
    gmst_ann = load_gmst_annual_from_hadcrut(gmst_csv)

    rows = []
    for _, row in nuts.iterrows():
        land = row["land"]
        try:
            mask2d = points_in_poly_mask(lon2d, lat2d, row["geometry"])
        except Exception:
            # if geometry invalid or other issue, skip
            warnings.warn(f"Skipping {land}: geometry -> mask error.")
            continue
        s_daily = area_weighted_mean_daily(ds[var], mask2d, lat2d)
        if s_daily.empty:
            warnings.warn(f"No data inside mask for {land}; skipping.")
            continue
        J = s_daily[jja_mask(s_daily.index)]
        base = J[(J.index.year>=BASELINE_YEARS[0])&(J.index.year<=BASELINE_YEARS[1])]
        if base.empty:
            warnings.warn(f"{land}: baseline empty; using all JJA for P95.")
            base = J
        if base.empty:
            # still empty: skip
            warnings.warn(f"{land}: no JJA data available for baseline or overall; skipping.")
            continue
        p95 = float(base.quantile(0.95))
        for y in summers:
            Jy = J[J.index.year==y]
            if Jy.empty: 
                warnings.warn(f"{land} {y}: no JJA days; skipping year.")
                continue
            gmst_y = float(gmst_ann.get(y, np.nan))
            if np.isnan(gmst_y):
                warnings.warn(f"GMST missing for year {y}; skipping.")
                continue
            PI = make_PI(Jy, gmst_y, beta)
            dEHD = ehd(Jy.values, p95) - ehd(PI, p95)
            rows.append({"land": land, "year": y, "dEHD": dEHD})
    if len(rows) == 0:
        # return empty dataframe with columns to avoid later KeyErrors
        return pd.DataFrame(columns=["land","year","dEHD"])
    return pd.DataFrame(rows)

def gini_from_values(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0: return np.nan
    # translate if negatives present to make all non-negative (Gini defined for non-negative incomes)
    if np.any(x < 0): x = x - x.min()
    s = x.sum()
    if s == 0: return 0.0
    x = np.sort(x)
    n = x.size
    # classical discrete Gini
    g = 1 - 2 * np.sum((n + 1 - np.arange(1, n+1)) * x) / (n * s)
    return float(g)

# ---------- Main figure builder ----------
def build_figure5(eobs_path: str, var: str, gmst_csv: str, beta: float,
                  summers: List[int], save_path: str):
    # sanity: embedded populations cover summers?
    for y in summers:
        if y not in POP:
            raise ValueError(f"No embedded population for year {y}. Available: {sorted(POP.keys())}")

    # ΔEHD by Land
    df = compute_byland_dEHD(eobs_path, var, gmst_csv, beta, summers)
    if df.empty:
        raise RuntimeError("No ΔEHD results produced. Check input data, masks, and GMST CSV.")

    # Order by first summer; fall back to STATES if mismatch
    first = summers[0]
    if first in df["year"].values:
        order = df[df["year"]==first].sort_values("dEHD", ascending=False)["land"].tolist()
    else:
        # fallback: use unique lands present
        order = df["land"].unique().tolist()
    # Ensure all POP states appear in ordering (append missing)
    for s in STATES:
        if s not in order:
            order.append(s)
    lands = order

    # Wide ΔEHD table (ensure all lands present)
    wide = df.pivot_table(index="land", columns="year", values="dEHD", aggfunc="mean")
    # reindex with lands but keep any additional names existing in wide
    all_lands_index = [l for l in lands if l in wide.index] + [l for l in wide.index if l not in lands]
    wide = wide.reindex(all_lands_index)

    # Per-capita table (per 100k) — only for lands that match POP keys
    cap = wide.copy()
    for y in summers:
        pop_vec = []
        for ln in wide.index:
            # if pop missing, try to match common variants; otherwise NaN
            try:
                pop_val = POP[y][ln]
            except KeyError:
                # attempt a simple normalization: replace unicode hyphen variants and spaces
                ln_alt = ln.replace("–","-").replace("—","-").strip()
                pop_val = POP[y].get(ln_alt, np.nan)
            pop_vec.append(pop_val)
        pop_vec = np.array(pop_vec, dtype=float)
        # avoid division by zero or NaN: where pop_vec is nan -> result nan
        with np.errstate(invalid='ignore', divide='ignore'):
            cap[y] = cap[y] / (pop_vec/1e5)

    # Lorenz + Gini for absolute ΔEHD
    lorenz_data_abs = {}
    gini_abs = {}
    for y in summers:
        if y not in wide.columns:
            v = np.zeros(len(wide.index))
        else:
            v = wide[y].fillna(0).values
        v_sorted = np.sort(v)
        cum = np.cumsum(v_sorted)
        # handle all-zero case
        if cum.size == 0 or cum[-1] == 0:
            y_l = np.concatenate([[0], np.zeros_like(cum)])
        else:
            y_l = np.concatenate([[0], cum / cum[-1]])
        x_l = np.linspace(0, 1, len(v_sorted)+1)
        lorenz_data_abs[y] = (x_l, y_l)
        gini_abs[y] = gini_from_values(v)

    # Lorenz + Gini for per-capita ΔEHD
    lorenz_data_cap = {}
    gini_cap = {}
    for y in summers:
        if y not in cap.columns:
            v = np.zeros(len(cap.index))
        else:
            v = cap[y].fillna(0).values
        v_sorted = np.sort(v)
        cum = np.cumsum(v_sorted)
        if cum.size == 0 or cum[-1] == 0:
            y_l = np.concatenate([[0], np.zeros_like(cum)])
        else:
            y_l = np.concatenate([[0], cum / cum[-1]])
        x_l = np.linspace(0, 1, len(v_sorted)+1)
        lorenz_data_cap[y] = (x_l, y_l)
        gini_cap[y] = gini_from_values(v)

    # ---------- Plot ----------
    fig = plt.figure(figsize=FIGSIZE, constrained_layout=True)
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[1.2, 1.0])

    # (a) ΔEHD absolute (paired dots)
    axA = fig.add_subplot(gs[0, 0])
    x = np.arange(len(wide.index))
    for i, land in enumerate(wide.index):
        ys = [wide.loc[land, y] if y in wide.columns else np.nan for y in summers]
        axA.plot([i]*len(summers), ys, color="0.85", linewidth=1.0, zorder=1)
    for y in summers:
        vals = wide[y].values if y in wide.columns else np.full(len(wide.index), np.nan)
        axA.scatter(x, vals, s=36,
                    label=str(y), color=YEAR_COLORS.get(y, None),
                    marker=MARKERS.get(y, "o"), zorder=2)
    axA.set_xlim(-0.5, len(wide.index)-0.5)
    axA.set_xticks(x); axA.set_xticklabels(wide.index, rotation=45, ha="right")
    axA.set_ylabel("ΔEHD (°C·days)", fontweight="bold")
    #axA.set_title("(a) Anthropogenic burden by German states", fontweight="bold")
    axA.set_title("(a)", fontweight="bold")
    axA.legend(frameon=False, ncol=min(3,len(summers)))
    axA.grid(axis="y", alpha=0.25, linewidth=0.8)

    # (b) Per-capita grouped bars
    axB = fig.add_subplot(gs[0, 1])
    width = 0.22 if len(summers)==3 else 0.28
    for k, y in enumerate(summers):
        vals = cap[y].fillna(0).values if y in cap.columns else np.zeros(len(cap.index))
        axB.bar(x + (k-1)*width, vals, width=width,
                label=str(y), color=YEAR_COLORS.get(y, None), edgecolor="none")
    axB.set_xlim(-0.5, len(wide.index)-0.5)
    axB.set_xticks(x); axB.set_xticklabels(wide.index, rotation=45, ha="right")
    axB.set_ylabel("ΔEHD per 100k (°C·days)", fontweight="bold")
    #axB.set_title("(b) Per-capita burden by German states", fontweight="bold")
    axB.set_title("(b)", fontweight="bold")
    axB.legend(frameon=False, ncol=min(3,len(summers)))
    axB.grid(axis="y", alpha=0.25, linewidth=0.8)

    # (c) Absolute Lorenz
    axC = fig.add_subplot(gs[1, 0])
    for y in summers:
        lx, ly = lorenz_data_abs[y]
        axC.plot(lx, ly, linewidth=2.0, color=YEAR_COLORS.get(y, None),
                 label=f"{y}  (Gini={gini_abs[y]:.2f})")
    axC.plot([0,1],[0,1], color="0.6", linestyle="--", linewidth=1.2, label="Equality")
    axC.set_xlim(0,1); axC.set_ylim(0,1)
    axC.set_xlabel("Cumulative share of German states", fontweight="bold")
    axC.set_ylabel("Cumulative share of national ΔEHD", fontweight="bold")
    #axC.set_title("(c) Concentration of burden — absolute ΔEHD", fontweight="bold")
    axC.set_title("(c)", fontweight="bold")
    axC.legend(frameon=False, ncol=2, loc="upper left")
    #axC.legend(frameon=False, ncol=min(4,len(summers)+1), loc="lower right")

    # (d) Per-capita Lorenz
    axD = fig.add_subplot(gs[1, 1])
    for y in summers:
        lx, ly = lorenz_data_cap[y]
        axD.plot(lx, ly, linewidth=2.0, color=YEAR_COLORS.get(y, None),
                 label=f"{y}  (Gini={gini_cap[y]:.2f})")
    axD.plot([0,1],[0,1], color="0.6", linestyle="--", linewidth=1.2, label="Equality")
    axD.set_xlim(0,1); axD.set_ylim(0,1)
    axD.set_xlabel("Cumulative share of German states", fontweight="bold")
    axD.set_ylabel("Cumulative share of per-capita ΔEHD", fontweight="bold")
    #axD.set_title("(d) Concentration of burden — per-capita ΔEHD", fontweight="bold")
    axD.set_title("(d)", fontweight="bold")
    axD.legend(frameon=False, ncol=2, loc="upper left")
    #axD.legend(frameon=False, ncol=min(4,len(summers)+1), loc="lower right")

    # Save figure
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    print("Saved:", save_path)

    # Save tables (wide absolute and per100k)
    base = save_path.rsplit(".",1)[0]
    wide.reset_index().to_csv(base + "_dEHD_byLand.csv", index=False)
    cap.reset_index().to_csv(base + "_dEHD_per100k_byLand.csv", index=False)
    print("Tables saved:", base + "_dEHD_byLand.csv", "and", base + "_dEHD_per100k_byLand.csv")

# -------- CLI --------
def parse_summers(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eobs", required=True, help="Path to E-OBS netCDF (daily TG).")
    ap.add_argument("--var", default="tg", help="Variable name in netCDF (default: 'tg').")
    ap.add_argument("--gmst_csv", required=True, help="Path to HadCRUT GMST CSV.")
    ap.add_argument("--beta", type=float, required=True, help="Beta sensitivity (°C local per °C global).")
    ap.add_argument("--summers", default="2018,2019,2022", help="Comma-separated summers to analyze.")
    ap.add_argument("--save", default="figures/fig5_equity.png", help="Output figure path.")
    args = ap.parse_args()

    summers = parse_summers(args.summers)
    # quick guard: ensure populations exist for requested years and all states (best-effort)
    for y in summers:
        missing_states = [s for s in STATES if s not in POP.get(y, {})]
        if missing_states:
            # don't fail hard if NUTS-labelling mismatch expected; warn instead
            warnings.warn(f"Population missing for year {y}: {missing_states}. Panel (b)/(d) may contain NaNs.")
    build_figure5(args.eobs, args.var, args.gmst_csv, args.beta, summers, args.save)

if __name__ == "__main__":
    main()
