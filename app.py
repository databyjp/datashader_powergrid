# ========== (c) JP Hwang 2/5/21  ==========

import logging
import pandas as pd
import numpy as np
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from colorcet import bgy, fire, blues, CET_L18, dimgray, kgy, CET_L9
import datashader as ds
import datashader.transfer_functions as tf
import json
import datetime
from utils import en2ll, scheduler_url
import math
import coiled
from distributed import Client
from spatialpandas.io import read_parquet, read_parquet_dask
from retrying import retry

# Init logger
logger = logging.getLogger(__name__)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
sh.setFormatter(formatter)
root_logger.addHandler(sh)

# Set app parameters for use in dropdowns
mapbox_styles = ["open-street-map", "carto-positron", "carto-darkmatter", "stamen-terrain", "stamen-toner", "stamen-watercolor"]
def_mapbox_style = "carto-darkmatter"
pp_sectors = ['Commercial Non-CHP', 'Electric Utility', 'IPP Non-CHP', 'Industrial CHP', 'Industrial Non-CHP']
pp_sectord_dict = [{"label": s, "value": s} for s in pp_sectors]


def mask_df_ll(df_in, lons, lats):
    # Mask dataframe based on lon/lat corners
    lon0, lon1 = (min(lons), max(lons))
    lat0, lat1 = (min(lats), max(lats))
    tmp_df = (df_in.query(f"LAT > {lat0}").query(f"LAT < {lat1}")
              .query(f"LON > {lon0}").query(f"LON < {lon1}"))
    return tmp_df


def filter_df(df_in, zoom_in, lon_in, lat_in, volt_range=None, cap_range=None, pp_sectors=None):
    # Mask DF with centre coordinate & zoom level
    m_per_px = 156543.03  # Conversion from "zoom level" to pixels by metres
    lon_offset = m_per_px / (2 ** zoom_in) * np.cos(np.radians(lat_in)) / 111111 * 300
    lat_offset = m_per_px / (2 ** zoom_in) * np.cos(np.radians(lat_in)) / 111111 * 150
    relayout_corners_ll = [
        [lon_in - lon_offset, lat_in + lat_offset],
        [lon_in + lon_offset, lat_in + lat_offset],
        [lon_in + lon_offset, lat_in - lat_offset],
        [lon_in - lon_offset, lat_in - lat_offset],
    ]
    lons, lats = zip(*relayout_corners_ll)
    tmp_df = mask_df_ll(df_in, lons, lats)

    # Filter DF by variable ranges
    if volt_range is not None:
        tmp_df = tmp_df[(tmp_df["VOLTAGE"] >= volt_range[0]) & (tmp_df["VOLTAGE"] <= volt_range[1])]

    if cap_range is not None:
        tmp_df = tmp_df[(tmp_df["Total_MW"] >= cap_range[0]) & (tmp_df["Total_MW"] <= cap_range[1])]

    if pp_sectors is not None:
        tmp_df = tmp_df[tmp_df["Sector_Name"].isin(pp_sectors)]

    return tmp_df


def get_cnr_coords(agg, coord_params):
    # Get corners of aggregated image, which need to be passed to mapbox
    coords_lon, coords_lat = agg.coords[coord_params[0]].values, agg.coords[coord_params[1]].values  # agg is an xarray object, see http://xarray.pydata.org/en/stable/ for more details
    coords_ll = [
        [coords_lon[0], coords_lat[0]],
        [coords_lon[-1], coords_lat[0]],
        [coords_lon[-1], coords_lat[-1]],
        [coords_lon[0], coords_lat[-1]],
    ]
    curr_coords_ll_out = en2ll(coords_ll)
    return curr_coords_ll_out


def get_mapbox_layer(df_in, agg_type="points", agg_param=None, x_col="x", y_col="y", geom=None, cmap=fire, res="fine",
                     opacity=1.0, rasterize=False, meshgrid_cols=None):
    if agg_param is None:
        agg_param = ds.any()

    if geom is not None:
        x_agg = None
        y_agg = None
    else:
        x_agg = x_col
        y_agg = y_col

    if res == "fine":
        width = 700
        height = 500
    elif res == "medium":
        width = 350
        height = 250
    elif res == "coarse":
        width = 200
        height = 140
    else:
        logger.warning("Resolution input unrecognised, using default width & height")
        width = 700
        height = 500

    logger.info("Starting datashader")
    cvs = ds.Canvas(plot_width=width, plot_height=height)
    if agg_type == "line":
        agg = cvs.line(df_in, agg=agg_param, geometry=geom, x=x_agg, y=y_agg)
    else:
        if not agg_type == "points":
            logger.warning("Inappropriate aggregation data type, defaulting to points")
        if rasterize is True:
            if meshgrid_cols is None:
                meshgrid_cols = [int(df_in[x_agg].nunique() * 0.8), int(df_in[y_agg].nunique() * 0.8)]  # Avoid gaps in the mesh
            cvs = ds.Canvas(plot_width=meshgrid_cols[0], plot_height=meshgrid_cols[1])
            agg = cvs.points(df_in, agg=agg_param, geometry=geom, x=x_agg, y=y_agg)
            agg = cvs.raster(agg, interpolate='linear')
        else:
            agg = cvs.points(df_in, agg=agg_param, geometry=geom, x=x_agg, y=y_agg)
            # if spread is True:  # TODO - check how this affects sums/data
            #     agg = tf.spread(agg, px=1)  # Spread for visibility

    img_out = tf.shade(agg, cmap=cmap)[::-1].to_pil()
    curr_coords_ll_out = get_cnr_coords(agg, [x_col, y_col])
    logger.info("Finished getting datashader layer")

    return {"sourcetype": "image", "opacity": opacity, "source": img_out, "coordinates": curr_coords_ll_out}, float(agg.min().values), float(agg.max().values)


def build_legend(scale_min=0.0, scale_max=1.0, colorscale_n=7, cmap=bgy, legend_title="Legend", dec_pl=0):
    colorscale_int = int((len(cmap) - 1) / (colorscale_n - 1))
    legend_headers = list()
    legend_colors = list()
    colwidth = int(100 / (colorscale_n))
    for i in range(colorscale_n):
        tmp_col = cmap[i * colorscale_int]  # Color
        tmp_num = round(scale_min + (scale_max - scale_min) / (colorscale_n - 1) * i, dec_pl)  # Number
        legend_headers.append(
            html.Th(
                f" ",
                style={
                    "background-color": tmp_col,
                    "color": "black",
                    "fontSize": 11,
                    "height": "0.9em",
                    "width": str(colwidth) + "%"
                },
            ),
        )  # Build the color boxes
        legend_colors.append(html.Td(tmp_num, style={"fontSize": 11}))  # Build the text legend

    legend_body = html.Table([
        html.Tr(legend_headers),
        html.Tr(legend_colors),
    ], style={"width": "90%"})
    legend = html.Table([
        html.Tr([html.Td(html.Strong(f"{legend_title}:", style={"fontSize": 13}))]),
        html.Tr([html.Td(legend_body)])
    ], style={"width": "90%"})
    return legend


def get_lon_lat_zoom(relayout_data, prev_center, prev_zoom):
    # If there is a zoom level or relayout_data["mapbox.center"] - Update map based on the info. Otherwise - use default
    if relayout_data:  # Center point loc will not always be in relayout data
        relayout_lon = relayout_data.get("mapbox.center", {}).get("lon", prev_center[0])
        relayout_lat = relayout_data.get("mapbox.center", {}).get("lat", prev_center[1])
        relayout_zoom = relayout_data.get("mapbox.zoom", float(prev_zoom))
    else:
        relayout_lon = prev_center[0]
        relayout_lat = prev_center[1]
        relayout_zoom = float(prev_zoom)
    return relayout_lon, relayout_lat, relayout_zoom


def build_base_map(mapbox_style=def_mapbox_style):
    # Build the underlying map that the Datashader overlay will be on top of
    fig = px.scatter_mapbox(dummy_df, lat="x", lon="y")
    fig["layout"]["mapbox"].update(default_position)
    fig.update_layout(mapbox_style=mapbox_style, showlegend=False, margin=dict(l=5, r=5, t=5, b=5))
    return fig


def build_demand_fig():
    tmp_df = demand_df[demand_df["Series"].isin(["US", "CA", "TX", "NY", "WI", "Solar", "Wind"])]
    tmp_df.sort_values("MWh", inplace=True)
    fig = px.bar(tmp_df, x="MWh", y="Series", color="Category", orientation="h",
                 color_discrete_sequence=px.colors.qualitative.D3,
                 height=200, template="plotly_white", labels={"MWh": "Energy (MWh)", "Series": "Group"})
    fig.update_layout(margin=dict(l=5, r=5, t=5, b=5))
    return fig


def build_grid_hist(df_in):
    tmp_grid_bins = pd.cut(df_in["VOLTAGE"], voltage_bins)
    df_in["grid_bin"] = tmp_grid_bins
    filt_grid_histdata = (df_in.groupby("grid_bin").sum()["SHAPE_Leng"] / df_in["SHAPE_Leng"].sum() * 100).reset_index()
    filt_grid_histdata["bin_txt"] = filt_grid_histdata["grid_bin"].apply(lambda x: str(x.left + 1) + " to " + str(x.right) + "V")
    filt_grid_histdata["data"] = "Shown"

    tot_grid_histdata = pd.concat([grid_histdata, filt_grid_histdata])
    fig = px.bar(tot_grid_histdata, x="bin_txt", y="SHAPE_Leng", color="data", barmode="group",
                 color_discrete_map={"Shown": "SeaGreen", "Overall": "MediumAquamarine"},
                 height=250, template="plotly_white", labels={"SHAPE_Leng": "% of data"})
    fig.update_layout(margin=dict(l=5, r=5, t=5, b=5), xaxis_title=None)
    return fig


def built_pot_hist_df(pdf_in, ref_df, bins):
    tmp_df_bins = pd.cut(pdf_in["z"], bins=bins)
    pdf_in["bins"] = tmp_df_bins
    tmp_df_binned = pdf_in.groupby("bins")["z"].count() / len(pdf_in) * 100
    tmp_df_binned = tmp_df_binned.reset_index()
    tmp_df_binned["data"] = "Shown"

    hist_df = pd.concat([ref_df, tmp_df_binned])
    hist_df["bins_txt"] = hist_df["bins"].apply(lambda x: str(x.left) + " to " + str(x.right))
    return hist_df


def get_summ_para():
    ref_perc = 10
    solar_req = (ref_demand["US"] * ref_perc / 100) / ref_potential["solar"]
    wind_req = (ref_demand["US"] * ref_perc / 100) / ref_potential["wind"]
    return [
        html.P([
            html.Span("The solar/wind capacity in the U.S. has grown, but is still limited (see right).")
        ]),
        html.P([
            html.Span('But every additional '),
            html.Span(dbc.Badge(f"{int(solar_req):,} sq. km", color="warning")),
            html.Span(" of solar or "),
            html.Span(dbc.Badge(f"{int(wind_req):,} sq. km", color="primary")),
            html.Span(" of wind power could supply another "),
            html.Span(dbc.Badge(f"{ref_perc} % of the U.S.'s", color="info")),  # Reference: https://www.eia.gov/energyexplained/use-of-energy/electricity-use-in-homes.php
            html.Span(" electricity demands."),
        ]),
        html.P([
            "These are relatively small areas. Take a look below to find any number of suitable areas with resource availability and grid installations."
        ])
    ]


def get_select_est_para(select_area, select_power, avg_var, var_type="wind"):
    if np.isnan(select_power):
        return html.Div([
            html.P([
                html.Span("The selected area is outside the view window, try moving back or selecting a new area."),
            ]),
        ])
    else:
        if var_type == "wind":
            avg_resource_sent = html.Span([
                f" with an average wind speed of ",
                dbc.Badge(f"{round(avg_var, 1)}m/s", color="primary"),
                f" (viable min: 5m/s)."
            ])
            highlight_col = "primary"
        else:
            avg_resource_sent = html.Span([
                f" with an average solar radiation of ",
                # avg_var * 1000 => MWh/km2/day
                # avg_var * 1000 / 24 = MWh/km2/hr = MW / km2
                dbc.Badge(f"{round(avg_var, 1)} kWh/m2/day", color="warning"),
                "."
            ])
            highlight_col = "warning"

        return html.Div([
            html.P([
                html.Span("The selected area is approximately "),
                html.Span(dbc.Badge(f"{int(select_area):,}" + " sq. km", color=highlight_col)),
                avg_resource_sent
            ]),
            html.P([
                html.Span("This could provide "),
                html.Span(dbc.Badge(str(round(select_power / 10 ** 6, 2)) + " million MWh", color="danger")),
                html.Span(" of power annually, or "),
                html.Span(dbc.Badge(str(round(select_power / ref_demand["US"] * 100, 1)) + " %", color="info")),  # Reference: https://www.eia.gov/energyexplained/use-of-energy/electricity-use-in-homes.php
                html.Span(" of average electricty demand for the United States."),
            ]),
        ])


def get_overlay_trace(lons, lats):
    return go.Scattermapbox(
        mode="markers+lines",
        line=dict(color='firebrick', width=10),
        lon=[min(lons), min(lons), max(lons), max(lons), min(lons)],
        lat=[max(lats), min(lats), min(lats), max(lats), max(lats)],
        marker={'size': 10}
    )


def get_potential_power(para, avg_pot_val):
    if para == "wind":
        return ref_potential["wind"] * ((avg_pot_val / ref_wind_speed))  # MW / hectare - Although wind power follows cubic root of velocity; using avg as a stand-in for how often rated power is reached
    else:
        return avg_pot_val * 1000 * 365 * solar_cap_factor * pv_eff  # MW / hectare


def get_ref_ul():
    # Build links to resources to display at the bottom of the page
    res_df = pd.read_csv("data/resource_refs.csv")
    return html.Ul([
        html.Li([
            row["Resource"] + " ", html.A(row["Link"], href=row["Link"])
        ]) for i, row in res_df.iterrows()
    ])


def build_pp_hist(df_in, ref_df, catname):
    comb_df = pd.concat([ref_df, df_in])
    comb_df = comb_df.assign(shown=comb_df.duplicated("OBJECTID"))
    comb_df = comb_df.assign(cat=catname + ", not shown")
    comb_df.loc[comb_df.duplicated("OBJECTID"), "cat"] = catname + ", shown"
    return comb_df


def get_hist_count_fig(in_df, hist_colormap, hist_labels):
    hist_count_fig = px.histogram(
        in_df, x="Total_MW", color="cat", log_y=True, barmode="group", nbins=10,
        color_discrete_map=hist_colormap, template="plotly_white", height=300,
        labels=hist_labels
    )
    hist_count_fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=0.99, xanchor="right", x=0.99),
        margin=dict(l=5, r=5, t=5, b=5)
    )
    return hist_count_fig


def get_hist_cap_fig(in_df, hist_colormap, hist_labels):
    hist_cap_fig = px.histogram(
        in_df, x="Total_MW", y="Total_MW", log_y=True, color="cat", barmode="group", nbins=10, histfunc='avg',
        color_discrete_map=hist_colormap, template="plotly_white", height=300,
        labels=hist_labels
    )
    hist_cap_fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=0.99, xanchor="right", x=0.99),
        margin=dict(l=5, r=5, t=5, b=5)
    )
    return hist_cap_fig


# ====================
# Connect to cluster
# ====================

# Global initialization - To ensure that different clients are generated
client = None


# # ====================
# # IF USING COILED - USE BELOW CODE
# # ====================
# def get_client(client):
#     if client is None or client.status != "running":
#         logger.info("Starting or connecting to Coiled cluster...")
#         cluster = coiled.Cluster(
#             name="grid-app-clust-1",
#             software="grid-app-env",
#             n_workers=1,
#             worker_cpu=2,
#             worker_memory="8 GiB",
#             shutdown_on_close=False,
#             scheduler_options={"idle_timeout": "24 hours"}
#         )
#         try:
#             client = Client(cluster)
#         except:
#             logger.info("Failed, trying to close the client and connect again...")
#             Client(cluster).close()
#             client = Client(cluster)
#         logger.info(f"Coiled cluster is up! ({client.dashboard_link})")
#
#     return client
#
#
# # Read data
# def load_df():
#     logger.info("Loading data from S3 bucket")
#     df = read_parquet_dask(
#         "s3://databyjp/power_data/Transmission_Lines_proc_sm_packed.parq",
#     )
#     df["SHAPE_Leng"] = df["SHAPE_Leng"].astype(np.float32)
#     df["VOLTAGE"] = df["VOLTAGE"].astype(np.float32)
#     df["lon_a"] = df["lon_a"].astype(np.float32)
#     df["lat_a"] = df["lat_a"].astype(np.float32)
#     df["lon_b"] = df["lon_b"].astype(np.float32)
#     df["lat_b"] = df["lat_b"].astype(np.float32)
#     df["LAT"] = df["LAT"].astype(np.float32)
#     df["LON"] = df["LON"].astype(np.float32)
#     df["x_en"] = df["x_en"].astype(np.float32)
#     df["y_en"] = df["y_en"].astype(np.float32)
#     df = df.assign(TYPE=df["TYPE"].astype("category"))
#     df = df.assign(STATUS=df["STATUS"].astype("category"))
#
#     # Filter out rows with unknown voltage
#     df = df[df["VOLTAGE"] > 0]

#     # Load renewable potentials data
#     sp_df = dd.read_csv("s3://databyjp/power_data/nsrdb3_ghi_en_us_proc.csv")  # Load solar potential data
#     sp_df = sp_df.persist()
#
#     wp_df = dd.read_csv("s3://databyjp/power_data/wtk_conus_80m_mean_masked_proc.csv")  # Load wind potential data
#     wp_df = wp_df.persist()

#     logger.info("Data loaded")
#     return df, sp_df, wp_df
#
#
# client = get_client(client)
# df, sp_df, wp_df = load_df()
# df = df.persist()
# sp_df = sp_df.persist()
# wp_df = wp_df.persist()
# # ====================
# # END - COILED
# # ====================

# ====================
# FOR LOCAL COMPUTE - USE BELOW CODE
# ====================
def init_client():
    """
    This function must be called before any of the functions that require a client.
    """
    global client
    logger.info(f"Connecting to cluster at {scheduler_url} ... ")
    client = Client(scheduler_url)
    logger.info("done")


# Read data into a Dask DataFrame
@retry(wait_exponential_multiplier=100, wait_exponential_max=2000, stop_max_delay=6000)
def load_df(client, name):
    df_out = client.get_dataset(name)
    return df_out


init_client()
df = load_df(client, "df")
df = df.build_sindex()  # IMPORTANT: This is required to make sure that dask can read mult-part spatialpandas file
wp_df = load_df(client, "wp_df")
sp_df = load_df(client, "sp_df")
# ====================
# END - LOCAL CLUSTER
# ====================

logger.info("Loading minor datasets")
solar_df = pd.read_csv("data/Power_Plants_Solar_proc.csv", index_col=0)  # Load Solar Powerplant data
wind_df = pd.read_csv("data/Power_Plants_Wind_proc.csv", index_col=0)  # Load Wind Powerplant data
demand_df = pd.read_csv("data/us_states_elec_demand_proc.csv")  # State-by-state demand  https://www.eia.gov/electricity/data/state/sales_annual.xlsx

logger.info("Preprocessing datasets")
sp_df["bins"] = sp_df["z"].map_partitions(pd.cut, 15)  # Or use math.ceil((wp_df["z"].max()+1)) for full max
sp_df_binned = (sp_df.groupby("bins")["z"].count() / len(sp_df) * 100).reset_index().compute()
sp_df_binned["data"] = "Overall"

wp_df = wp_df[wp_df["z"] > 0]
wp_df["bins"] = wp_df["z"].map_partitions(pd.cut, list(range(3, 11)))  # Or use math.ceil((wp_df["z"].max()+1)) for full max
wp_df_binned = (wp_df.groupby("bins")["z"].count() / len(wp_df) * 100).reset_index().compute()
wp_df_binned["data"] = "Overall"

voltage_bins = [0, 100, 161, 287, 450, 1000]
df["grid_bin"] = df["VOLTAGE"].map_partitions(pd.cut, voltage_bins)
grid_histdata = (df.groupby("grid_bin").sum()["SHAPE_Leng"] / df["SHAPE_Leng"].sum() * 100).compute().reset_index()
grid_histdata["bin_txt"] = grid_histdata["grid_bin"].apply(lambda x: str(x.left + 1) + " to " + str(x.right) + "V")
grid_histdata["data"] = "Overall"

ref_potential = {"solar": 120000, "wind": 10000}  # Viable outputs in MWh / hectare / year (see docs for ref)
ref_demand = demand_df[["Series", "MWh"]].set_index("Series").to_dict()["MWh"]
ref_wind_speed = 6
solar_cap_factor = 0.25
pv_eff = 0.2

max_pcap = max(solar_df["Total_MW"].max(), wind_df["Total_MW"].max())

logger.info("Setting dataset extents")
lonmin = min(df["lon_a"].min().compute(), df["lon_b"].min().compute())
lonmax = max(df["lon_a"].max().compute(), df["lon_b"].max().compute())
latmin = min(df["lat_a"].min().compute(), df["lat_b"].min().compute())
latmax = max(df["lat_a"].max().compute(), df["lat_b"].max().compute())

geo_extents_ll = np.array([[lonmin, latmin], [lonmax, latmax]])

# Set initial map view
init_zoom = 6.5
init_lon = -102.07
init_lat = 31.34

default_position = {
    "zoom": init_zoom,
    "pitch": 0,
    "bearing": 0,
    "center": {"lon": init_lon, "lat": init_lat},
}

logger.info("Building a placeholder DF with categories")
dummy_df = pd.DataFrame([{"x": latmin, "y": lonmin}, {"x": latmax, "y": lonmax}])
init_fig = build_base_map()

# ====================================================================
# ========== Init Dash App ==========
# ====================================================================
app = dash.Dash(__name__, external_stylesheets=[
    dbc.themes.BOOTSTRAP,
    {
        'href': 'https://use.fontawesome.com/releases/v5.8.1/css/all.css',
        'rel': 'stylesheet',
        'integrity': 'sha384-50oBUHEmvpQ+1lW4y57PTFmhCaXp0ML5d60M1M7uH2+nqUivzIebhndOJK28anvf',
        'crossorigin': 'anonymous'
    }
])
server = app.server

# ==============================
# App Layout
# ==============================
header_class = "pl-3 pt-3 mt-2 pb-1 border-bottom border-dark"
subheader_class = "ml-3 mt-3 text-dark"

header = html.Div(
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2([
                    "U.S. Power Grid".upper(),
                    html.I(className="fas fa-bolt rounded text-danger py-1 px-1 mx-2"),
                    "Explorer".upper()
                ]),
            ], className="col-12 col-md-8"),
            dbc.Col([
                html.Small(["App built with ", html.A("Plotly Dash".upper(), href="https://plotly.com/")], className="py-2 px-3 bg-light rounded text-dark")
            ], className="col-12 col-md-4 text-right"),
        ], className="py-1 border-bottom pb-1 mb-3"),

        html.P([
            html.Span("Visually explore the current distribution of the U.S. power grid"),
            html.I(className="fas fa-bolt rounded text-danger py-1 px-1 mx-2"),
            html.Span(", current wind"),
            html.I(className="fas fa-wind rounded text-info py-1 px-1 mx-2"),
            html.Span("and solar"),
            html.I(className="fas fa-solar-panel rounded text-warning py-1 px-1 mx-2"),
            html.Span("power plants as well as potential locations for further wind/solar power plants."),
        ], className="lead mb-1"),
    ]), className="bg-dark text-light py-3"
)

body = html.Div([dbc.Container(
    [
        dbc.Row(dbc.Col(html.H3("Summary".upper(), className=header_class))),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([html.H5("Overview", className="card-title mb-0 mt-0")]),
                    dbc.CardBody(get_summ_para(), id="summ-txt")
                ], color="info", outline=True),
                html.Div([
                    html.Span("Select an area below with the 'BOX SELECT' tool to see its renewable energy potential.")
                ], className="rounded bg-primary text-light px-4 py-2 mt-2")
            ], className="col-sm-12 col-md-7"),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Current Renewables vs. Demand", className="card-title mb-0 mt-0")),
                    dbc.CardBody(
                        dcc.Graph(figure=build_demand_fig())
                    )
                ])
            ], className="col-sm-12 col-md-5"),
        ]),
        dbc.Row(
            [
                dbc.Col([
                    html.H3("Explorer".upper(), className=header_class),
                    dbc.Card(
                        [
                            dbc.CardHeader([
                                dbc.Row([
                                    dbc.Col([html.H5("Shading:".upper(), className="card-title mt-1 pt-1")], className="col-sm-4 col-md-3 pl-4 py-0 mt-1"),
                                    dbc.Col([
                                        dcc.Dropdown(
                                            options=[{'label': s, 'value': s} for s in mapbox_styles],
                                            value=def_mapbox_style,
                                            id="mapbox-style",
                                            className="py-0 my-0"
                                        ),
                                    ], className="col-sm-6 py-0 my-1"),
                                ])
                            ], className="py-0"),
                            dbc.CardBody([
                                dcc.Graph(figure=init_fig, id="map-graph", config={"modeBarButtonsToRemove": ["lasso2d", "toImage", "toggleHover", "pan2d", "zoomInGeo", "zoomOutGeo"], "displayModeBar": True}),
                            ], className="p-0 m-0"),
                            html.Div(
                                dbc.Badge("Datashader update time: Unknown", color="secondary", className="ml-3 mt-0 mb-2", id="update-time")
                            ),
                            html.Div([
                                html.P("...")
                            ], className="p-1 m-1", id="map-notes")
                        ], color="primary", outline=True
                    ),
                    html.H3("Renewables: Evaluation".upper(), className=header_class),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader(html.H5("Renewable Availability", className="card-title mb-0 mt-0")),
                                dbc.CardBody([dcc.Graph(figure=px.scatter(height=200), id="histogram-potentials")]),
                            ], outline=True),
                        ], className="col-sm-12 col-md-6"),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader(html.H5("Grid Composition", className="card-title mb-0 mt-0")),
                                dbc.CardBody(dcc.Graph(figure=px.scatter(height=300), id="grid-hist"))
                            ])
                        ], className="col-sm-12 col-md-6"),
                    ]),
                    html.H3("Current renewables".upper(), className=header_class),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card(
                                [
                                    dbc.CardHeader(html.H5("Plant Numbers", className="card-title mb-0 mt-0")),
                                    dbc.CardBody(children=[
                                        dcc.Graph(figure=px.scatter(), id="histogram-fig-count")
                                    ], className="p-0 m-0"),
                                ], outline=True, className="mx-0 px-0"
                            ),
                        ], className="col-sm-12 col-md-6"),
                        dbc.Col([
                            dbc.Card(
                                [
                                    dbc.CardHeader(html.H5("Plant Capacities", className="card-title mb-0 mt-0")),
                                    dbc.CardBody(children=[
                                        dcc.Graph(figure=px.scatter(), id="histogram-fig-cap")
                                    ], className="p-0 m-0"),
                                ], outline=True, className="mx-0 px-0"
                            ),
                        ], className="col-sm-12 col-md-6"),
                    ]),
                ], className="col-sm-12 col-md-7 col-lg-9"),
                dbc.Col([
                    html.H3("Legend".upper(), className=header_class),
                    dbc.Card([
                        dbc.CardBody(
                            build_legend(scale_min=0, scale_max=100),
                            className="p-1 m-1", id="legends-card"
                        ),
                    ]),
                    html.H3("Controls".upper(), className=header_class),
                    html.H5("Focus (Presets)".upper(), className=subheader_class),
                    dbc.Card([
                        dbc.RadioItems(
                            options=[
                                {'label': 'Wind'.upper(), 'value': 'wind'},
                                {'label': 'Solar'.upper(), 'value': 'solar'},
                            ],
                            value='wind',
                            className="big-radio my-2 py-2 pl-3",
                            id="focus-preset-radio",
                            inline=False
                        ),
                    ], color="info", outline=True),
                    html.H5("Layer display".upper(), className=subheader_class),
                    dbc.Button("Show/hide", outline=True, color="primary", size="sm", id="collapse-layers-button"),
                    dbc.Collapse([
                        dbc.Card([
                            dbc.CardBody(
                                [
                                    dcc.Checklist(
                                        options=[
                                            {'label': 'Electrical Grid', 'value': 'grid'},
                                            {'label': 'Solar Power Plants', 'value': 'solar'},
                                            {'label': 'Wind Power Plants', 'value': 'wind'},
                                        ],
                                        value=['grid', 'solar', 'wind'],
                                        id="layer-checklist"
                                    )
                                ]
                            )
                        ], className="mt-2"),
                    ], id="collapse-layers", is_open=False),
                    html.H5("Renewables potential".upper(), className=subheader_class),
                    dbc.Button("Show/hide", outline=True, color="primary", size="sm", id="collapse-renewable-button"),
                    dbc.Collapse([
                        dbc.Card([
                            # dbc.CardHeader("Renewable energy potential"),
                            dbc.CardBody(
                                [
                                    dbc.Label("Resource type"),
                                    dcc.Dropdown(
                                        options=[
                                            {'label': 'Solar', 'value': 'solar'},
                                            {'label': 'Wind', 'value': 'wind'},
                                            {'label': 'None', 'value': 'none'},
                                        ],
                                        value='wind',
                                        id="potential-overlay"
                                    ),
                                    dbc.Label("Colormap"),
                                    dcc.Dropdown(
                                        options=[
                                            {'label': 'Dimgray', 'value': 'dimgray'},
                                            {'label': 'CET_L9', 'value': 'CET_L9'},
                                        ],
                                        value='CET_L9',
                                        id="overlay-colormap",
                                        className="mb-2"
                                    ),
                                ]
                            )
                        ], className="mt-2"),
                    ], id="collapse-renewable", is_open=False),
                    html.H5("Grid".upper(), className=subheader_class),
                    dbc.Button("Show/hide", outline=True, color="primary", size="sm", id="collapse-grid-button"),
                    dbc.Collapse([
                        dbc.Card([
                            # dbc.CardHeader("Grid voltages"),
                            dbc.CardBody(
                                [
                                    dcc.RangeSlider(
                                        id='grid-voltages',
                                        min=math.log(100),
                                        max=math.log(1000),
                                        step=None,
                                        marks={
                                            math.log(100): {'label': '100V'},
                                            math.log(161): {'label': '161V', 'style': {'color': 'gray'}},
                                            math.log(287): '287V',
                                            math.log(450): '450V',
                                            math.log(1000): '1kV',
                                        },
                                        value=[math.log(161), math.log(1000)]
                                    ),
                                ]
                            )
                        ], className="mt-2"),
                    ], id="collapse-grid", is_open=False),
                    html.H5("Power plants".upper(), className=subheader_class),
                    dbc.Button("Show/hide", outline=True, color="primary", size="sm", id="collapse-plants-button"),
                    dbc.Collapse([
                        dbc.Card([
                            dbc.CardHeader("Aggregation Resolution"),
                            dbc.CardBody(
                                [
                                    html.Label("Solar power plants"),
                                    dcc.Dropdown(
                                        options=[{"label": "Fine", "value": "fine"}, {"label": "Medium", "value": "medium"}, {"label": "Coarse", "value": "coarse"}],
                                        value="coarse",
                                        multi=False,
                                        id="solar-res"
                                    ),
                                    html.Label("Wind power plants"),
                                    dcc.Dropdown(
                                        options=[{"label": "Fine", "value": "fine"}, {"label": "Medium", "value": "medium"}, {"label": "Coarse", "value": "coarse"}],
                                        value="coarse",
                                        multi=False,
                                        id="wind-res"
                                    )
                                ]
                            )
                        ], className="mt-2"),
                        dbc.Card([
                            dbc.CardHeader("Aggregatation Type"),
                            dbc.CardBody(
                                [
                                    html.Label("Solar power plants"),
                                    dcc.Dropdown(
                                        options=[{"label": "Total Power", "value": "total"}, {"label": "Average Power", "value": "avg"}, {"label": "Count", "value": "count"}],
                                        value="total",
                                        multi=False,
                                        id="solar-agg"
                                    ),
                                    html.Label("Wind power plants"),
                                    dcc.Dropdown(
                                        options=[{"label": "Total Power", "value": "total"}, {"label": "Average Power", "value": "avg"}, {"label": "Count", "value": "count"}],
                                        value="total",
                                        multi=False,
                                        id="wind-agg"
                                    )
                                ]
                            )
                        ], className="mt-2"),
                        dbc.Card([
                            dbc.CardHeader("Power Plant - Capacities"),
                            dbc.CardBody(
                                [
                                    dcc.RangeSlider(
                                        id='pp-caps',
                                        min=0,
                                        max=max_pcap,
                                        step=10,
                                        value=[0, max_pcap]
                                    ),
                                    html.Div(id="pp-cap-note", className="mt-0 pt-0")
                                ]
                            ),
                        ], className="mt-2"),
                        dbc.Card([
                            dbc.CardHeader("Power Plant - Sectors"),
                            dbc.CardBody(
                                [
                                    dcc.Dropdown(
                                        options=pp_sectord_dict,
                                        value=[i["value"] for i in pp_sectord_dict],
                                        multi=True,
                                        clearable=False,
                                        id="pp-sectors"
                                    ),
                                ]
                            )
                        ], className="mt-2"),
                    ], id="collapse-plants", is_open=False)
                ], className="col-sm-12 col-md-5 col-lg-3"),
            ]
        ),
        # ===== SHOW THESE ROWS TO HELP DEBUGGING =====
        dbc.Row(
            [html.P(id="prev-zoom", children=init_zoom)], style={"display": "none"}
        ),
        dbc.Row(
            [html.P(id="prev-center", children=json.dumps([init_lon, init_lat]))],
            style={"display": "none"},
        ),
        dbc.Row(
            [html.P(id="relayout-text-old"), html.Span("", id="placeholder")],
            style={"display": "none"},
        ),
        # # ===== ONLY IF USING COILED =====
        # html.H5("Dashboard Status", className=subheader_class),
        # dbc.Row([
        #     dbc.Col([
        #         dbc.Card(
        #             [
        #                 dbc.CardHeader("Coiled Cluster"),
        #                 dbc.CardBody(
        #                     children=[
        #                         html.Div([
        #                             "Status: ", dbc.Badge("", color="secondary", className="ml-3 mt-0 mb-2", id="coiled-status"),
        #                             html.Br(),
        #                             "Cluster dashboard: ", html.A("", href="", id="coiled-dashboard-href"),
        #                             html.Br(),
        #                             html.Div([
        #                                 dbc.Button(
        #                                     "Restart Cluster", id="restart-btn", color="danger", className="mt-3"
        #                                 ),
        #                             ], id="restart-div", style={"display": "none"}),
        #                         ])
        #                     ],
        #                     id="cluster-status",
        #                 ),
        #             ], color="secondary", outline=True
        #         ),
        #     ], className="col-sm-6 col-md-4"),
        #     dbc.Col([
        #         html.Div([
        #             dbc.Alert("The cluster is down - try restarting it.",
        #                       id="alert-clusterdown", dismissable=True, is_open=False, color="warning"),
        #             dbc.Alert("Restarting the cluster - please try again in a few minutes",
        #                       id="alert-restart", dismissable=True, is_open=False, color="info"),
        #         ], id="alerts-div")
        #     ], className="col-sm-6 col-md-4"),
        # ]),
        # # ===== END - COILED STATUS =====
        html.H3("References".upper(), className=header_class),
        get_ref_ul(),
    ]
)])

app.layout = html.Div([header, body])


@app.callback(
    [
        Output("map-graph", "figure"),
        Output("prev-center", "children"),
        Output("prev-zoom", "children"),
        Output("pp-cap-note", "children"),
        Output("update-time", "children"),
        Output("relayout-text-old", "children"),
        Output("legends-card", "children"),
        Output("histogram-fig-count", "figure"),
        Output("histogram-fig-cap", "figure"),
        Output("histogram-potentials", "figure"),
        Output("map-notes", "children"),
        Output("grid-hist", "figure")
    ],
    [
        Input("map-graph", "relayoutData"),
        Input("map-graph", "selectedData"),
        Input("mapbox-style", "value"),
        Input("layer-checklist", "value"),
        Input("potential-overlay", "value"),
        Input("overlay-colormap", "value"),
        Input("grid-voltages", "value"),
        Input("pp-caps", "value"),
        Input("pp-sectors", "value"),
        Input("solar-res", "value"),
        Input("wind-res", "value"),
        Input("solar-agg", "value"),
        Input("wind-agg", "value"),
        State("prev-center", "children"),
        State("prev-zoom", "children"),
    ],
)
def update_overlay(relayout_data, selected_data, mapbox_style_in, layers_list, potential_layer, pot_cmap, log_grid_voltages, pp_caps, pp_sectors, solar_res, wind_res, solar_agg, wind_agg, prev_center_json, prev_zoom):
    prev_center = json.loads(prev_center_json)
    relayout_lon, relayout_lat, relayout_zoom = get_lon_lat_zoom(relayout_data, prev_center, prev_zoom)
    new_center = {"lon": relayout_lon, "lat": relayout_lat}

    if relayout_zoom is None:
        logger.info("No relayout info, getting default map...")
        relayout_zoom = init_zoom
        relayout_lon = init_lon
        relayout_lat = init_lat
    else:
        logger.info("Got relayout info, processing...")

    start_time = datetime.datetime.now()
    # ===== UPDATE MAPBOX LAYERS =====
    fig = build_base_map(mapbox_style=mapbox_style_in)
    mapbox_layers = list()
    legends_div = list()

    # UPDATE RENEWABLE POTENTIALS LAYERS
    overlay_opacity = 0.35
    if pot_cmap == "CET_L9":
        pot_cmap = CET_L9
    else:
        if pot_cmap != "dimgray":
            logger.warning("Did not recognise colormap selection")
        pot_cmap = CET_L9

    if potential_layer == "none":
        select_est_para = "Try selecting a resource type under 'Renewables Potential'."
        hist_df = wp_df_binned
        hist_df["bins_txt"] = hist_df["bins"].apply(lambda x: str(x.left) + " to " + str(x.right))
        pot_fig_colmap = {"Shown": "MediumBlue", "Overall": "LightSkyBlue"}
        pot_fig_labels = {"z": "% of data", "bins_txt": "Wind speed (m/s) at 80m"}
    else:
        if potential_layer == "solar":
            tmp_pot_df = filter_df(sp_df, relayout_zoom, relayout_lon, relayout_lat).compute()
            pot_layer, agg_min, agg_max = get_mapbox_layer(tmp_pot_df, agg_type='points', agg_param=ds.mean("z"), x_col="x_en", y_col="y_en",
                                                           cmap=pot_cmap, res="fine", opacity=overlay_opacity, rasterize=True, meshgrid_cols=[int(0.8 * tmp_pot_df["LON"].nunique()), int(0.8 * tmp_pot_df["LAT"].nunique())])
            pot_legend = build_legend(scale_min=agg_min, scale_max=agg_max, legend_title="Solar potential (kWh/sqm/day)", cmap=pot_cmap, dec_pl=1)
            hist_df = built_pot_hist_df(tmp_pot_df, sp_df_binned, bins=sp_df_binned["bins"].unique().categories)
            pot_fig_colmap = {"Shown": "Orange", "Overall": "Khaki"}
            pot_fig_labels = {"z": "% of data", "bins_txt": "kWh/m2/day"}
        else:
            if potential_layer != "wind":
                logger.warning("Something is wrong!")
            tmp_pot_df = filter_df(wp_df, relayout_zoom, relayout_lon, relayout_lat).compute()

            pot_layer, agg_min, agg_max = get_mapbox_layer(tmp_pot_df, agg_type='points', agg_param=ds.mean("z"), x_col="x_en", y_col="y_en",
                                                           cmap=pot_cmap, res="fine", opacity=overlay_opacity, rasterize=True, meshgrid_cols=[int(0.8 * tmp_pot_df["x"].nunique()), int(0.8 * tmp_pot_df["y"].nunique())])
            pot_legend = build_legend(scale_min=agg_min, scale_max=agg_max, legend_title="Wind speed (m/s at 80m)", cmap=pot_cmap, dec_pl=0)
            hist_df = built_pot_hist_df(tmp_pot_df, wp_df_binned, bins=wp_df_binned["bins"].unique().categories)
            pot_fig_colmap = {"Shown": "MediumBlue", "Overall": "LightSkyBlue"}
            pot_fig_labels = {"z": "% of data", "bins_txt": "Wind speed (m/s) at 80m"}
        mapbox_layers.append(pot_layer)
        legends_div.append(pot_legend)

        # Draw overlay on screen
        if (selected_data is not None) and ("range" in selected_data.keys()):
            select_coords = selected_data["range"]["mapbox"]
            lons = [ll[0] for ll in select_coords]
            lats = [ll[1] for ll in select_coords]
        else:  # Note: deg of latitude: ~111km, at 40 deg - deg of longitude: ~85km
            if (selected_data is not None) and ("range" not in selected_data.keys()):
                logger.warning("Something has gone wrong!")
            lons = [init_lon - 0.2, init_lon + 0.2]
            lats = [init_lat - 0.2, init_lat + 0.2]
        fig.add_trace(get_overlay_trace(lons, lats))

        # Get data for the selected area
        sel_pot_df = mask_df_ll(tmp_pot_df, lons, lats)
        select_area = (max(lons) - min(lons)) * 85 * (max(lats) - min(lats)) * 110
        avg_pot_val = sel_pot_df["z"].mean()
        tmp_potential = get_potential_power(potential_layer, avg_pot_val)
        select_power = tmp_potential * select_area
        select_est_para = get_select_est_para(select_area, select_power, avg_pot_val, var_type=potential_layer)

    pot_fig = px.bar(hist_df, x="bins_txt", y="z", color="data", barmode="group", color_discrete_map=pot_fig_colmap,
                     labels=pot_fig_labels, height=250, template="plotly_white")
    pot_fig.update_layout(
        legend=dict(orientation="h", yanchor="top", y=0.99, xanchor="right", x=0.99),
        margin=dict(l=5, r=5, t=5, b=5),
    )

    # Filter DFs by coordinates
    grid_voltages = [int(math.exp(i)) for i in log_grid_voltages]
    grid_voltages = [grid_voltages[0] - 1, grid_voltages[1] + 1]  # Account for rounding errors
    tmp_df = filter_df(df, relayout_zoom, relayout_lon, relayout_lat, volt_range=grid_voltages).compute()
    tmp_solar_df = filter_df(solar_df, relayout_zoom, relayout_lon, relayout_lat, cap_range=pp_caps, pp_sectors=pp_sectors)
    tmp_wind_df = filter_df(wind_df, relayout_zoom, relayout_lon, relayout_lat, cap_range=pp_caps, pp_sectors=pp_sectors)

    if "grid" in layers_list:
        grid_cmap = kgy
        grid_layer, agg_min, agg_max = get_mapbox_layer(tmp_df, agg_type="line", agg_param=ds.mean("VOLTAGE"), geom="geometry", cmap=kgy)
        grid_legend = build_legend(scale_min=agg_min, scale_max=agg_max, legend_title="Grid avg. voltage", cmap=grid_cmap, dec_pl=0)
        mapbox_layers.append(grid_layer)
        legends_div.append(grid_legend)

    if "solar" in layers_list:
        solar_cmap = CET_L18
        if solar_agg == "total":
            solar_agg_func = ds.sum("Total_MW")
            solar_legend_txt = "Total solar capacity (MW)"
        elif solar_agg == "avg":
            solar_agg_func = ds.mean("Total_MW")
            solar_legend_txt = "Avg solar capacity (MW)"
        else:
            if solar_agg != "count":
                logger.info("Unrecognised variable, defaulting to count")
            solar_agg_func = ds.count()
            solar_legend_txt = "Solar power plants"
        solar_layer, agg_min, agg_max = get_mapbox_layer(tmp_solar_df, agg_type="points", agg_param=solar_agg_func, x_col="x_en", y_col="y_en", cmap=solar_cmap, res=solar_res)
        solar_legend = build_legend(scale_min=agg_min, scale_max=agg_max, legend_title=solar_legend_txt, cmap=solar_cmap, dec_pl=0)
        mapbox_layers.append(solar_layer)
        legends_div.append(solar_legend)
    if "wind" in layers_list:
        wind_cmap = blues
        if wind_agg == "total":
            wind_agg_func = ds.sum("Total_MW")
            wind_legend_txt = "Total wind capacity (MW)"
        elif wind_agg == "avg":
            wind_agg_func = ds.mean("Total_MW")
            wind_legend_txt = "Avg wind capacity (MW)"
        else:
            if wind_agg != "count":
                logger.info("Unrecognised variable, defaulting to count")
            wind_agg_func = ds.count()
            wind_legend_txt = "Wind power plants"
        wind_layer, agg_min, agg_max = get_mapbox_layer(tmp_wind_df, agg_type="points", agg_param=wind_agg_func, x_col="x_en", y_col="y_en", cmap=wind_cmap, res=wind_res)
        wind_legend = build_legend(scale_min=agg_min, scale_max=agg_max, legend_title=wind_legend_txt, cmap=wind_cmap, dec_pl=0)
        mapbox_layers.append(wind_layer)
        legends_div.append(wind_legend)

    fig.update_layout(mapbox_layers=mapbox_layers)
    position = {"zoom": relayout_zoom, "center": new_center}
    fig["layout"]["mapbox"].update(position)
    fig.update_traces(hoverinfo="none", hovertemplate=None)

    update_time = datetime.datetime.now() - start_time

    # Build histograms
    comb_wind_df = build_pp_hist(tmp_wind_df, wind_df, "wind power")
    comb_solar_df = build_pp_hist(tmp_solar_df, solar_df, "solar power")
    comb_pwr_df = pd.concat([comb_wind_df, comb_solar_df])
    hist_colormap = {
        "wind power, not shown": "#d0f0ff",
        "wind power, shown": "RoyalBlue",
        "solar power, not shown": "MistyRose",
        "solar power, shown": "Firebrick",
    }
    hist_labels = {
        "cat": "Category",
        "Total_MW": "Plant capacity (MW)",
    }
    hist_count_fig = get_hist_count_fig(comb_pwr_df, hist_colormap, hist_labels)
    hist_cap_fig = get_hist_cap_fig(comb_pwr_df, hist_colormap, hist_labels)

    return (
        fig,
        json.dumps([relayout_lon, relayout_lat]),
        relayout_zoom,
        f"{pp_caps[0]}-{pp_caps[1]} MW",
        f"Datashader update time: {str(round(update_time.total_seconds(), 2))}s",
        json.dumps([relayout_data]),
        legends_div,
        hist_count_fig,
        hist_cap_fig,
        pot_fig,
        select_est_para,
        build_grid_hist(tmp_df)
    )


@app.callback(
    Output("layer-checklist", "value"),
    Output("potential-overlay", "value"),
    [Input("focus-preset-radio", "value")],
)
def preset_buttons(btn_input):
    if btn_input == "wind":
        layer_list = ["grid", "wind"]
        overlay_layer = "wind"
    else:
        layer_list = ["grid", "solar"]
        overlay_layer = "solar"
    return layer_list, overlay_layer


@app.callback(
    Output("collapse-layers", "is_open"),
    [Input("collapse-layers-button", "n_clicks")],
    [State("collapse-layers", "is_open")],
)
def toggle_collapse_layers(n, is_open):
    if n:
        return not is_open
    return is_open


@app.callback(
    Output("collapse-renewable", "is_open"),
    [Input("collapse-renewable-button", "n_clicks")],
    [State("collapse-renewable", "is_open")],
)
def toggle_collapse_renewable(n, is_open):
    if n:
        return not is_open
    return is_open


@app.callback(
    Output("collapse-grid", "is_open"),
    [Input("collapse-grid-button", "n_clicks")],
    [State("collapse-grid", "is_open")],
)
def toggle_collapse_grid(n, is_open):
    if n:
        return not is_open
    return is_open


@app.callback(
    Output("collapse-plants", "is_open"),
    [Input("collapse-plants-button", "n_clicks")],
    [State("collapse-plants", "is_open")],
)
def toggle_collapse_plants(n, is_open):
    if n:
        return not is_open
    return is_open


# # ===== CALLBACKS FOR WHEN USING COILED =====
# @app.callback(
#     [
#         Output("coiled-status", "children"),
#         Output("coiled-status", "color"),
#         Output("coiled-dashboard-href", "href"),
#         Output("coiled-dashboard-href", "children"),
#         Output("restart-div", "style"),
#         Output("alert-clusterdown", "is_open")
#     ],
#     [
#         Input("map-graph", "relayoutData"),
#     ],
# )
# def update_status(relayout_data,
#                   # , n_clicks
#                   ):
#     if client is None or client.status != "running":
#         if client is None:
#             status = "None"
#         else:
#             status = client.status
#         status_color = "warning"
#         coiled_href = "#"
#         restart_block = {"display": ""}
#         alert_clusterdown_open = True
#     else:
#         status = client.status
#         status_color = "success"
#         coiled_href = client.dashboard_link
#         restart_block = {"display": "none"}
#         alert_clusterdown_open = False
#
#     return status, status_color, coiled_href, "link", restart_block, alert_clusterdown_open
#
#
# @app.callback(
#     [Output("placeholder", "children"), Output("alert-restart", "is_open")],
#     [Input("restart-btn", "n_clicks")],
# )
# def restart_coiled(n_clicks):
#     if n_clicks is not None:
#         alert_restart_open = True
#
#         global client
#         client = get_client(client)
#
#         global df
#         df = load_df()
#
#         return dash.no_update, alert_restart_open
#
#     else:
#         alert_restart_open = False
#         return dash.no_update, alert_restart_open
# # ===== END - COILED CALLBACKS =====

if __name__ == "__main__":
    app.run_server(debug=False)
