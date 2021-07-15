# ========== (c) JP Hwang 2/5/21  ==========

import logging

# ===== START LOGGER =====

logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
import datashader as ds
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.express as px
import colorcet
from colorcet import bgy, fire, blues, CET_L18, kg
import datashader.transfer_functions as tf
import json
import datetime
from utils import en2ll, scheduler_url
import math
import coiled
from distributed import Client
from spatialpandas.io import read_parquet, read_parquet_dask
from retrying import retry

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
sh.setFormatter(formatter)
root_logger.addHandler(sh)

# Set app parameters / load data
mapbox_style = "carto-darkmatter"
col_options = [
    {"label": "TYPE", "value": "TYPE"},
    {"label": "STATUS", "value": "STATUS"},
]
pp_sectors = ['Commercial Non-CHP', 'Electric Utility', 'IPP Non-CHP', 'Industrial CHP', 'Industrial Non-CHP']
pp_sectord_dict = [{"label": s, "value": s} for s in pp_sectors]


# Init Dash App
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server


def mask_df_ll(df_in, lons, lats):
    lon0, lon1 = (
        min(lons),
        max(lons),
    )
    lat0, lat1 = (
        min(lats),
        max(lats),
    )
    tmp_df = (df_in
              .query(f"LAT > {lat0}")
              .query(f"LAT < {lat1}")
              .query(f"LON > {lon0}")
              .query(f"LON < {lon1}")
              )
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

    # Filter DF by voltage slider range
    if volt_range is not None:
        tmp_df = tmp_df[(tmp_df["VOLTAGE"] >= volt_range[0]) & (tmp_df["VOLTAGE"] <= volt_range[1])]

    if cap_range is not None:
        tmp_df = tmp_df[(tmp_df["Total_MW"] >= cap_range[0]) & (tmp_df["Total_MW"] <= cap_range[1])]

    if pp_sectors is not None:
        tmp_df = tmp_df[tmp_df["Sector_Name"].isin(pp_sectors)]

    return tmp_df


def get_cnr_coords(agg, coord_params):
    # agg is an xarray object, see http://xarray.pydata.org/en/stable/ for more details
    coords_lon, coords_lat = agg.coords[coord_params[0]].values, agg.coords[coord_params[1]].values
    # Corners of the image, which need to be passed to mapbox
    coords_ll = [
        [coords_lon[0], coords_lat[0]],
        [coords_lon[-1], coords_lat[0]],
        [coords_lon[-1], coords_lat[-1]],
        [coords_lon[0], coords_lat[-1]],
    ]
    curr_coords_ll_out = en2ll(coords_ll)
    return curr_coords_ll_out


def get_mapbox_layer(df_in, agg_type="points", agg_param=None, x_col="x", y_col="y", geom=None, cmap=fire, res="fine"):
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
    elif agg_type == "polygon":
        agg = cvs.polygons(df_in, agg=agg_param, geometry=geom)
    else:
        if not agg_type == "points":
            logger.warning("Inappropriate aggregation data type, defaulting to points")
        agg = cvs.points(df_in, agg=agg_param, geometry=geom, x=x_agg, y=y_agg)
        agg = tf.spread(agg, px=1)  # Spread for visibility
    img_out = tf.shade(agg, cmap=cmap)[::-1].to_pil()
    logger.info("Finished getting datashader img")

    curr_coords_ll_out = get_cnr_coords(agg, [x_col, y_col])

    return {"sourcetype": "image", "source": img_out, "coordinates": curr_coords_ll_out}, float(agg.min().values), float(agg.max().values)


def build_legend(scale_min=0.0, scale_max=1.0, colorscale_n=7, cmap=bgy, legend_title="Legend"):

    colorscale_int = int((len(cmap) - 1) / (colorscale_n - 1))
    legend_headers = list()
    legend_colors = list()
    colwidth = int(100 / (colorscale_n))
    for i in range(colorscale_n):
        tmp_col = cmap[i * colorscale_int]  # Color
        tmp_num = round(
            scale_min + (scale_max - scale_min) / (colorscale_n - 1) * i, 1
        )  # Number
        legend_headers.append(
            html.Th(
                f" ",
                style={
                    "background-color": tmp_col,
                    "color": "black",
                    "fontSize": 12,
                    "height": "1em",
                    "width": str(colwidth) + "%"
                },
            ),
        )
        legend_colors.append(
            html.Td(tmp_num, style={"fontSize": 14}),
        )

    legend_body = html.Table([
        html.Tr(legend_headers),
        html.Tr(legend_colors),
    ], style={"width": "90%"})
    legend = html.Table([
        html.Tr([
            html.Td(html.Strong(f"{legend_title}:"), style={"width": "40%"}),
            html.Td(legend_body, style={"width": "50%"})
        ])
    ], style={"width": "90%"})
    return legend


def get_lon_lat_zoom(relayout_data, prev_center, prev_zoom):
    # If there is a zoom level or relayout_data["mapbox.center"] - Update map based on the info
    # Otherwise - use default
    if relayout_data:  # Center point loc will not always be in relayout data
        relayout_lon = relayout_data.get("mapbox.center", {}).get("lon", prev_center[0])
        relayout_lat = relayout_data.get("mapbox.center", {}).get("lat", prev_center[1])
        relayout_zoom = relayout_data.get("mapbox.zoom", float(prev_zoom))
    else:
        relayout_lon = prev_center[0]
        relayout_lat = prev_center[1]
        relayout_zoom = float(prev_zoom)
    return relayout_lon, relayout_lat, relayout_zoom


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
#             scheduler_options={"idle_timeout": "1 hour"}
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
#     logger.info("Data loaded")
#     return df
#
#
# client = get_client(client)
# df = load_df()
# df = df.persist()
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
    # Init client
    logger.info(f"Connecting to cluster at {scheduler_url} ... ")
    client = Client(scheduler_url)
    logger.info("done")


# Read data into a Dask DataFrame
@retry(wait_exponential_multiplier=100, wait_exponential_max=2000, stop_max_delay=6000)
def load_df(client, name):

    df = client.get_dataset(name)
    return df


init_client()
df = load_df(client, "df")
df = df.build_sindex()  # IMPORTANT: This is required to make sure that dask can read mult-part spatialpandas file
# ====================
# END - LOCAL CLUSTER
# ====================


# Load Solar Powerplant data
logger.info("Loading solar power plant data")
solar_df = pd.read_csv("data/Power_Plants_Solar_proc.csv")

# Load Wind Powerplant data
logger.info("Loading wind power plant data")
wind_df = pd.read_csv("data/Power_Plants_Wind_proc.csv")

max_pcap = max(solar_df["Total_MW"].max(), wind_df["Total_MW"].max())
# ====================
logger.info("Setting dataset extents")
lonmin = min(df["lon_a"].min().compute(), df["lon_b"].min().compute())
lonmax = max(df["lon_a"].max().compute(), df["lon_b"].max().compute())
latmin = min(df["lat_a"].min().compute(), df["lat_b"].min().compute())
latmax = max(df["lat_a"].max().compute(), df["lat_b"].max().compute())

geo_extents_ll = np.array([[lonmin, latmin], [lonmax, latmax]])

# Set initial map view
init_zoom = 4.25
init_lon = -83.85
init_lat = 40.2

default_position = {
    "zoom": init_zoom,
    "pitch": 0,
    "bearing": 0,
    "center": {"lon": init_lon, "lat": init_lat},
}

logger.info("Building a placeholder DF with categories")
cat_colormap = colorcet.glasbey
dummy_data = [{"x": latmin, "y": lonmin}]
dummy_data.append({"x": latmax, "y": lonmax})
dummy_df = pd.DataFrame(dummy_data)


def build_base_map():
    # Build the underlying map that the Datashader overlay will be on top of
    fig = px.scatter_mapbox(
        dummy_df, lat="x", lon="y"
    )
    fig.update_layout(mapbox_style=mapbox_style)
    fig.update_traces(mode="markers", hoverinfo="skip", hovertemplate=None)
    fig.update_traces(marker=dict(size=0))
    fig["layout"]["mapbox"].update(default_position)
    fig.update_layout(showlegend=False)
    fig.update_layout(margin=dict(l=5, r=5, t=5, b=5))
    return fig


init_fig = build_base_map()

# ====================================================================
# ========== SET UP DASH APP LAYOUT ==========
# ====================================================================

# ==============================
# Actual layout
# ==============================
header_class = "mt-3"

header = html.Div(
    dbc.Container([
        html.H3(["US Power Grid Explorer"], className="py-3"),
    ]), className="bg-light"
)

body = html.Div([dbc.Container(
    [
        dbc.Row(dbc.Col(html.H5("Summary of Displayed Data", className=header_class))),
        dbc.Row(
            [
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H6("Electrical Grid", className="card-title mb-1")),
                        dbc.CardBody([f"..."], id="grid-data")
                    ], color="info", outline=True),
                ]),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H6("Solar power plants", className="card-title mb-1")),
                        dbc.CardBody([f"..."], id="solar-data")
                    ], color="info", outline=True),
                ]),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H6("Wind power plants", className="card-title mb-1")),
                        dbc.CardBody([f"..."], id="wind-data"),
                    ], color="info", outline=True),
                ]),
            ]
        ),
        dbc.Row(
            [
                dbc.Col([
                    html.H5("Visual Explorer", className=header_class),
                    dbc.Card(
                        [
                            dbc.CardHeader(html.H5("Map view", className="card-title mb-0 mt-0"), className="bg-primary text-white"),
                            dbc.CardBody([
                                dcc.Graph(figure=init_fig, id="map-graph"),
                                html.Div("", id="map-legend")
                            ], className="p-0 m-0"),
                            html.Div(
                                dbc.Badge("Datashader update time: Unknown", color="secondary", className="ml-3 mt-0 mb-2", id="update-time")
                            )
                        ], color="primary", outline=True
                    ),
                    dbc.Card(
                        [
                            dbc.CardHeader(html.H5("Legend", className="card-title mb-0 mt-0")),
                            dbc.CardBody(
                                build_legend(scale_min=0, scale_max=100)
                            , className="p-1 m-1", id="legends-card"),
                        ], color="info", outline=True, className="my-2"
                    ),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card(
                                [
                                    dbc.CardHeader(html.H5("Plant Numbers", className="card-title mb-0 mt-0")),
                                    dbc.CardBody(children=[
                                        dcc.Graph(figure=px.scatter(), id="histogram-fig-count")
                                    ], className="p-0 m-0"),
                                ], color="info", outline=True, className="mx-0 px-0"
                            ),
                        ], className="col-sm-12 col-md-6"),
                        dbc.Col([
                            dbc.Card(
                                [
                                    dbc.CardHeader(html.H5("Plant Capacities", className="card-title mb-0 mt-0")),
                                    dbc.CardBody(children=[
                                        dcc.Graph(figure=px.scatter(), id="histogram-fig-cap")
                                    ], className="p-0 m-0"),
                                ], color="info", outline=True, className="mx-0 px-0"
                            ),
                        ], className="col-sm-12 col-md-6"),
                    ]),
                    # # ===== ONLY IF USING COILED =====
                    # html.H5("Dashboard Status", className=header_class),
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
                ], className="col-sm-12 col-md-7 col-lg-9"),
                dbc.Col([
                    html.H5("Controls/Filters", className=header_class),
                    dbc.Card([
                        dbc.CardHeader("Show/hide layers"),
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
                    ]),
                    dbc.Card([
                        dbc.CardHeader("Grid voltages"),
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
                                        math.log(500): '500V',
                                        math.log(1000): '1kV',
                                    },
                                    value=[math.log(100), math.log(1000)]
                                ),
                            ]
                        )
                    ], className="mt-2"),
                    dbc.Card([
                        dbc.CardHeader("Aggregation Resolution"),
                        dbc.CardBody(
                            [
                                html.Label("Solar power plants"),
                                dcc.Dropdown(
                                    options=[{"label": "Fine", "value": "fine"}, {"label": "Medium", "value": "medium"}, {"label": "Coarse", "value": "coarse"}],
                                    value="medium",
                                    multi=False,
                                    id="solar-res"
                                ),
                                html.Label("Wind power plants"),
                                dcc.Dropdown(
                                    options=[{"label": "Fine", "value": "fine"}, {"label": "Medium", "value": "medium"}, {"label": "Coarse", "value": "coarse"}],
                                    value="medium",
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
                                    value="count",
                                    multi=False,
                                    id="solar-agg"
                                ),
                                html.Label("Wind power plants"),
                                dcc.Dropdown(
                                    options=[{"label": "Total Power", "value": "total"}, {"label": "Average Power", "value": "avg"}, {"label": "Count", "value": "count"}],
                                    value="count",
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
                        dbc.CardHeader("Power Plant - Sectors"),  # TODO - add modal to explain different types
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
        # ===== END - DEBUGGING ROWS =====
        dbc.Container([
            html.Small(["App built with ",
                        html.A("Plotly Dash", href="https://plotly.com/")])
        ], className="my-2"),
    ]
)])

app.layout = html.Div([header, body])


@app.callback(
    [
        Output("map-graph", "figure"),
        Output("prev-center", "children"),
        Output("prev-zoom", "children"),
        Output("grid-data", "children"),
        Output("solar-data", "children"),
        Output("wind-data", "children"),
        Output("pp-cap-note", "children"),
        Output("update-time", "children"),
        Output("update-time", "color"),
        Output("relayout-text-old", "children"),
        Output("legends-card", "children"),
        Output("histogram-fig-count", "figure"),
        Output("histogram-fig-cap", "figure"),
    ],
    [
        Input("map-graph", "relayoutData"),
        Input("layer-checklist", "value"),
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
def update_overlay(relayout_data, layers_list, grid_voltages, pp_caps, pp_sectors, solar_res, wind_res, solar_agg, wind_agg, prev_center_json, prev_zoom):

    grid_voltages = [int(math.exp(i)) for i in grid_voltages]
    grid_voltages = [grid_voltages[0]-1, grid_voltages[1]+1]  # Account for rounding errors

    prev_center = json.loads(prev_center_json)
    relayout_lon, relayout_lat, relayout_zoom = get_lon_lat_zoom(
        relayout_data, prev_center, prev_zoom
    )
    new_center = {"lon": relayout_lon, "lat": relayout_lat}

    if relayout_zoom is None:
        logger.info("No relayout info, getting default map...")
        relayout_zoom = init_zoom
        relayout_lon = init_lon
        relayout_lat = init_lat
    else:
        logger.info("Got relayout info, processing...")

    start_time = datetime.datetime.now()

    tmp_df = filter_df(df, relayout_zoom, relayout_lon, relayout_lat, volt_range=grid_voltages).compute()
    tmp_solar_df = filter_df(solar_df, relayout_zoom, relayout_lon, relayout_lat, cap_range=pp_caps, pp_sectors=pp_sectors)
    tmp_wind_df = filter_df(wind_df, relayout_zoom, relayout_lon, relayout_lat, cap_range=pp_caps, pp_sectors=pp_sectors)

    update_time = datetime.datetime.now() - start_time
    update_time = str(round(update_time.total_seconds(), 2))
    logger.info(f"Took {update_time}s to filter the DFs")
    fig = build_base_map()

    start_time = datetime.datetime.now()

    # Update mapbox layers & legends
    mapbox_layers = list()
    legends_div = list()
    if "grid" in layers_list:
        grid_cmap = kg
        grid_layer, agg_min, agg_max = get_mapbox_layer(tmp_df, agg_type="line", agg_param=ds.mean("VOLTAGE"), geom="geometry", cmap=grid_cmap)
        mapbox_layers.append(grid_layer)
        grid_legend = build_legend(scale_min=agg_min, scale_max=agg_max, legend_title="Average grid voltage", cmap=grid_cmap)
        legends_div.append(grid_legend)
    if "solar" in layers_list:
        solar_cmap = CET_L18
        if solar_agg == "total":
            solar_agg_func = ds.sum("Total_MW")
        elif solar_agg == "avg":
            solar_agg_func = ds.mean("Total_MW")
        elif solar_agg == "count":
            solar_agg_func = ds.count()
        else:
            logger.info("Unrecognised variable, defaulting to count")
            solar_agg_func = ds.count()
        solar_layer, agg_min, agg_max = get_mapbox_layer(tmp_solar_df, agg_type="points", agg_param=solar_agg_func, x_col="x_en", y_col="y_en", cmap=solar_cmap, res=solar_res)
        mapbox_layers.append(solar_layer)
        solar_legend = build_legend(scale_min=agg_min, scale_max=agg_max, legend_title="Solar power plants", cmap=solar_cmap)
        legends_div.append(solar_legend)
    if "wind" in layers_list:
        wind_cmap = blues
        if wind_agg == "total":
            wind_agg_func = ds.sum("Total_MW")
        elif wind_agg == "avg":
            wind_agg_func = ds.mean("Total_MW")
        elif wind_agg == "count":
            wind_agg_func = ds.count()
        else:
            logger.info("Unrecognised variable, defaulting to count")
            wind_agg_func = ds.count()
        wind_layer, agg_min, agg_max = get_mapbox_layer(tmp_wind_df, agg_type="points", agg_param=wind_agg_func, x_col="x_en", y_col="y_en", cmap=wind_cmap, res=wind_res)
        mapbox_layers.append(wind_layer)
        wind_legend = build_legend(scale_min=agg_min, scale_max=agg_max, legend_title="Wind power plants", cmap=wind_cmap)
        legends_div.append(wind_legend)

    # Build histogram
    comb_wind_df = pd.concat([wind_df, tmp_wind_df])
    comb_wind_df = comb_wind_df.assign(shown=comb_wind_df.duplicated("OBJECTID"))
    comb_wind_df = comb_wind_df.assign(cat="wind power, not shown")
    comb_wind_df.loc[comb_wind_df.duplicated("OBJECTID"), "cat"] = "wind power, shown"
    comb_solar_df = pd.concat([solar_df, tmp_solar_df])
    comb_solar_df = comb_solar_df.assign(shown=comb_solar_df.duplicated("OBJECTID"))
    comb_solar_df = comb_solar_df.assign(cat="solar power, not shown")
    comb_solar_df.loc[comb_solar_df.duplicated("OBJECTID"), "cat"] = "solar power, shown"
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
    hist_count_fig = px.histogram(
        comb_pwr_df, x="Total_MW", color="cat", log_y=True, barmode="group", nbins=10,
        color_discrete_map=hist_colormap, template="plotly_white", height=350,
        labels=hist_labels
    )
    hist_count_fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=0.99, xanchor="right", x=0.99),
        margin=dict(l=5, r=5, t=5, b=5)
    )
    hist_cap_fig = px.histogram(
        comb_pwr_df, x="Total_MW", y="Total_MW", log_y=True, color="cat", barmode="group", nbins=10, histfunc='avg',
        color_discrete_map=hist_colormap, template="plotly_white", height=350,
        labels=hist_labels
    )
    hist_cap_fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=0.99, xanchor="right", x=0.99),
        margin=dict(l=5, r=5, t=5, b=5)
    )

    fig.update_layout(mapbox_layers=mapbox_layers)
    position = {
        "zoom": relayout_zoom,
        "center": new_center,
    }
    fig["layout"]["mapbox"].update(position)

    update_time = datetime.datetime.now() - start_time
    update_time = str(round(update_time.total_seconds(), 2))
    update_txt = f"Datashader update time: {update_time}s",
    update_status = "info"

    # Calc values to display on datacards
    start_time = datetime.datetime.now()
    tot_grid_len = int(df["SHAPE_Leng"].sum().compute() * 85)  # Magic number 85 comes from a rule of thumb lat/lon to km conversion from at this latitude (See https://gis.stackexchange.com/questions/251643/approx-distance-between-any-2-longitudes-at-a-given-latitude)
    update_time = datetime.datetime.now() - start_time
    update_time = str(round(update_time.total_seconds(), 2))
    logger.info(f"Took {update_time}s for length calc")
    tot_wind_cap = int(tmp_wind_df[tmp_wind_df["Wind_MW"] > 0]["Wind_MW"].sum())
    tot_solar_cap = int(tmp_solar_df[tmp_solar_df["Solar_MW"] > 0]["Solar_MW"].sum())

    pp_cap_txt = f"{pp_caps[0]}-{pp_caps[1]} MW"

    return (
        fig,
        json.dumps([relayout_lon, relayout_lat]),
        relayout_zoom,
        [
            html.Span([dbc.Badge(f"{str(len(tmp_df))}", color="primary", className="mr-1 mb-2 large-badge"), "Grid sections"]),
            html.Br(),
            html.Span([dbc.Badge(f"{tot_grid_len}", color="info", className="mr-1 large-badge"), "km in length"])
        ],
        [
            html.Span([dbc.Badge(f"{len(tmp_solar_df)}", color="primary", className="mr-1 mb-2 large-badge"), f"Solar plants"]),
            html.Br(),
            html.Span([dbc.Badge(f"{tot_solar_cap}", color="info", className="mr-1 large-badge"), f"MW total capacity"])
        ],
        [
            html.Span([dbc.Badge(f"{len(tmp_wind_df)}", color="primary", className="mr-1 mb-2 large-badge"), f"Wind plants"]),
            html.Br(),
            html.Span([dbc.Badge(f"{tot_wind_cap}", color="info", className="mr-1 large-badge"), f"MW total capacity"])
        ],
        pp_cap_txt,
        update_txt,
        update_status,
        json.dumps([relayout_data]),
        legends_div,
        hist_count_fig,
        hist_cap_fig
    )


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
    app.run_server(debug=True)
