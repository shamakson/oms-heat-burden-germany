# figure_ehd_maps_top_bars_onecol_legend.py
import io
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec

CSV = """land,year,dEHD
Nordrhein-Westfalen,2018,16.265361805296077
Nordrhein-Westfalen,2019,14.377701292754256
Nordrhein-Westfalen,2022,9.265126865489972
Hessen,2018,19.8555680443876
Hessen,2019,14.503512452457066
Hessen,2022,10.658994197704555
Baden-Württemberg,2018,9.477344321247191
Baden-Württemberg,2019,14.90711693491243
Baden-Württemberg,2022,10.46197154927669
Sachsen-Anhalt,2018,18.674044072303385
Sachsen-Anhalt,2019,12.750934131195578
Sachsen-Anhalt,2022,9.101410208697548
Thüringen,2018,16.68637032631813
Thüringen,2019,13.179609153314772
Thüringen,2022,11.7738171247447
Sachsen,2018,16.77473017127138
Sachsen,2019,10.393756747160221
Sachsen,2022,13.921094920643803
Rheinland-Pfalz,2018,14.996324692442002
Rheinland-Pfalz,2019,15.570549687079872
Rheinland-Pfalz,2022,10.408158013313457
Brandenburg,2018,16.653105569039166
Brandenburg,2019,13.679251498437228
Brandenburg,2022,12.030654791404345
Niedersachsen,2018,16.300054674679586
Niedersachsen,2019,12.66443590334086
Niedersachsen,2022,7.12207239348411
Bayern,2018,10.219504256234604
Bayern,2019,12.5644343116833
Bayern,2022,9.45206485164331
Mecklenburg-Vorpommern,2018,18.416939104593435
Mecklenburg-Vorpommern,2019,11.341516100953047
Mecklenburg-Vorpommern,2022,10.63544515376557
Schleswig-Holstein,2018,16.114966418927523
Schleswig-Holstein,2019,8.158885860008969
Schleswig-Holstein,2022,5.432334787157558
Saarland,2018,15.23316362498593
Saarland,2019,17.5013722672589
Saarland,2022,10.463979457490296
Berlin,2018,14.715740706233312
Berlin,2019,12.041288959239097
Berlin,2022,10.498882469921814
Bremen,2018,16.002119004119145
Bremen,2019,12.959378270856625
Bremen,2022,8.823290808843936
Hamburg,2018,17.506366920952814
Hamburg,2019,9.516873041344887
Hamburg,2022,7.195957180334144
"""

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

ABBR = {
    "Baden-Württemberg":"BW","Bayern":"BY","Berlin":"BE","Brandenburg":"BB","Bremen":"HB","Hamburg":"HH",
    "Hessen":"HE","Mecklenburg-Vorpommern":"MV","Niedersachsen":"NI","Nordrhein-Westfalen":"NW",
    "Rheinland-Pfalz":"RP","Saarland":"SL","Sachsen":"SN","Sachsen-Anhalt":"ST","Schleswig-Holstein":"SH","Thüringen":"TH",
}

# --- Data prep
df = pd.read_csv(io.StringIO(CSV))
df["abbr"] = df["land"].map(ABBR)
wide = df.pivot_table(index=["land","abbr"], columns="year", values="dEHD").reset_index()
wide_sorted = wide.sort_values(by=2018, ascending=False).reset_index(drop=True)

nuts = read_de_nuts_from_web(level=1, year=2021, resolution="01M")
gdf = nuts.merge(wide, on="land", how="left")
gdf["abbr"] = gdf["land"].map(ABBR)
gdf_eq = gdf.to_crs(3035)
gdf["area_km2"] = gdf_eq.geometry.area/1e6
gdf["label_pt"] = gdf.representative_point()

# --- Figure (maps top + one-column legend; bars bottom)
fig = plt.figure(figsize=(12, 9))
gs  = GridSpec(2, 3, height_ratios=[1.25, 1.0], width_ratios=[1.0, 1.0, 0.55], hspace=0.16, wspace=0.06)
ax_maps = fig.add_subplot(gs[0,0:2])
ax_mleg = fig.add_subplot(gs[0,2])      # single-column legend
ax_bars = fig.add_subplot(gs[1,0:3])

years = [2018, 2019, 2022]
vmin, vmax = float(df["dEHD"].min()), float(df["dEHD"].max())
cmap = "YlOrRd"

# (a) maps: three insets
insets = [ax_maps.inset_axes([0.02+i*0.32, 0.10, 0.30, 0.80]) for i in range(3)]
halo = [pe.withStroke(linewidth=2.0, foreground="white")]
FS_NORMAL, FS_SMALL = 7.5, 6.0
small_thresh = gdf["area_km2"].quantile(0.18)

for ax, yr in zip(insets, years):
    gdf.plot(column=yr, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, edgecolor="white", linewidth=0.6)
    ax.axis("off")
    xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
    ax.text(xmin + 0.012*(xmax-xmin), ymin + 0.02*(ymax-ymin), f"{yr}", ha="left", va="bottom",
            fontsize=9, weight="bold", color="black")
    for _, row in gdf.iterrows():
        if pd.isna(row[yr]): continue
        fs = FS_SMALL if row["area_km2"] <= small_thresh else FS_NORMAL
        x, y = row["label_pt"].x, row["label_pt"].y

        # --- special handling for tiny city-states: Berlin, Bremen, Hamburg (offset SOUTH with arrow)
        if row["land"] == "Berlin":
            dy = -0.35
            ax.annotate("BE", xy=(x, y), xytext=(x, y+dy), ha="center", va="top",
                        fontsize=FS_SMALL, weight="bold", path_effects=halo,
                        arrowprops=dict(arrowstyle="-", lw=0.7, color="black", shrinkA=0, shrinkB=0))
        elif row["land"] == "Bremen":
            dy = 0.35
            ax.annotate("HB", xy=(x, y), xytext=(x, y+dy), ha="center", va="top",
                        fontsize=FS_SMALL, weight="bold", path_effects=halo,
                        arrowprops=dict(arrowstyle="-", lw=0.7, color="black", shrinkA=0, shrinkB=0))
        elif row["land"] == "Hamburg":
            dy = -0.25
            ax.annotate("HH", xy=(x, y), xytext=(x, y+dy), ha="center", va="top",
                        fontsize=FS_SMALL, weight="bold", path_effects=halo,
                        arrowprops=dict(arrowstyle="-", lw=0.7, color="black", shrinkA=0, shrinkB=0))
        else:
            ax.text(x, y, row["abbr"], ha="center", va="center",
                    fontsize=fs, weight="bold", color="black", path_effects=halo)

sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
cbar = fig.colorbar(sm, ax=ax_maps, orientation="horizontal", fraction=0.06, pad=0.02)
cbar.set_label("EHD (°C·days)", fontweight="bold")
ax_maps.text(0.0, 1.02, "(a)", transform=ax_maps.transAxes, ha="left", va="bottom", fontsize=12, weight="bold")
ax_maps.axis("off")

# Map legend (abbr → full), SINGLE column, styled
ax_mleg.axis("off")
pairs = sorted([(ABBR[name], name) for name in ABBR], key=lambda x: x[0])
legend_text = "\n".join([f"{a} — {n}" for a, n in pairs])
ax_mleg.add_patch(plt.Rectangle((0.07, 0.04), 0.92, 0.92, transform=ax_mleg.transAxes,
                                fill=True, facecolor="white", edgecolor="0.7", linewidth=0.9))
ax_mleg.text(0.08, 0.94, "States", transform=ax_mleg.transAxes,
             ha="left", va="top", fontsize=10, weight="bold")
ax_mleg.text(0.08, 0.90, legend_text, transform=ax_mleg.transAxes,
             ha="left", va="top", fontsize=9, linespacing=2.0)

# (b) bars: legend inside panel, horizontal
x = range(len(wide_sorted)); wbar = 0.26
bar_colors = {2018:"#fdae61", 2019:"#f46d43", 2022:"#d73027"}
for i, yr in enumerate(years):
    ax_bars.bar([xi + (i-1)*wbar for xi in x], wide_sorted[yr].values, width=wbar,
                label=str(yr), color=bar_colors[yr])
ax_bars.set_xticks(list(x))
ax_bars.set_xticklabels(wide_sorted["abbr"])
ax_bars.set_xlabel("States", fontweight="bold")
ax_bars.set_ylabel("EHD (°C·days)", fontweight="bold")
ax_bars.margins(x=0.02)
leg = ax_bars.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=3, frameon=True, framealpha=1.0, borderpad=0.8)
leg.get_frame().set_linewidth(0.8)
ax_bars.text(-0.06, 1.02, "(b)", transform=ax_bars.transAxes, ha="left", va="bottom", fontsize=12, weight="bold")

plt.subplots_adjust(bottom=0.08, top=0.965, left=0.06, right=0.98, hspace=0.18, wspace=0.06)
fig.savefig("figure_ehd_maps_bars.pdf", bbox_inches="tight")
fig.savefig("figure_ehd_maps_bars.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Wrote: figure_ehd_maps_bars.(pdf|png)")
