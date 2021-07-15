# ========== (c) JP Hwang 28/6/21  ==========

import logging
import pandas as pd
import numpy as np
from utils import ll2en

logger = logging.getLogger(__name__)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
sh.setFormatter(formatter)
root_logger.addHandler(sh)

desired_width = 320
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', desired_width)

import geopandas as gpd
from spatialpandas import GeoDataFrame

trans_lines = gpd.read_file('temp/shp/Transmission_Lines.shp', encoding='utf8')

# ==============================
# Convert to spatialpandas DF
# ==============================
df = GeoDataFrame(trans_lines)

df = df.assign(bounds=df["geometry"].apply(lambda x: x.to_shapely().bounds))
df = df.assign(lon_a=df["bounds"].apply(lambda x: x[0]))
df = df.assign(lat_a=df["bounds"].apply(lambda x: x[1]))
df = df.assign(lon_b=df["bounds"].apply(lambda x: x[2]))
df = df.assign(lat_b=df["bounds"].apply(lambda x: x[3]))
df = df.drop("bounds", axis=1)

df = df.assign(LAT=df["lat_a"])
df = df.assign(LON=df["lon_a"])
vals_en = np.array(ll2en(df[["LON", "LAT"]].values))

# Add easting/northing dims
df = df.assign(x_en=vals_en[:, 0])
df = df.assign(y_en=vals_en[:, 1])

df = df.assign(wire_type="UNKNOWN")
df.loc[df["TYPE"].str.contains("OVERHEAD"), "wire_type"] = "OVERHEAD"
df.loc[df["TYPE"].str.contains("UNDERGROUND"), "wire_type"] = "UNDERGROUND"

# ==============================
# Convert LL to EN
# ==============================
from spatialpandas import GeoSeries
df = df.assign(geometry_ll=df["geometry"])
en_geom = trans_lines["geometry"].to_crs("EPSG:3395")
df = df.assign(geometry=GeoSeries(en_geom))

df.to_parquet("data/grid/Transmission_Lines_proc.parq")

df = df[["TYPE", "STATUS", 'OWNER', 'VOLTAGE', 'VOLT_CLASS', "SHAPE_Leng", 'geometry', 'lon_a', 'lat_a', 'lon_b', 'lat_b', 'LAT', 'LON', 'x_en', 'y_en']]
df.to_parquet("data/grid/Transmission_Lines_proc_sm.parq")

import dask.dataframe as dd
ddf = dd.from_pandas(df, npartitions=2)
ddf_packed = ddf.pack_partitions(npartitions=2)
ddf_packed.to_parquet('temp/Transmission_Lines_proc_sm_packed.parq')

# Convert solar/wind data
solar_df = pd.read_csv("data/Power_Plants_Solar.csv")
vals_en = np.array(ll2en(solar_df[["Longitude", "Latitude"]].values))
solar_df = solar_df.assign(x_en=vals_en[:, 0])
solar_df = solar_df.assign(y_en=vals_en[:, 1])
solar_df = solar_df.assign(LAT=solar_df["Latitude"])
solar_df = solar_df.assign(LON=solar_df["Longitude"])
solar_df.to_csv("data/Power_Plants_Solar_proc.csv")

wind_df = pd.read_csv("data/Power_Plants_Wind.csv")
vals_en = np.array(ll2en(wind_df[["Longitude", "Latitude"]].values))
wind_df = wind_df.assign(x_en=vals_en[:, 0])
wind_df = wind_df.assign(y_en=vals_en[:, 1])
wind_df = wind_df.assign(LAT=wind_df["Latitude"])
wind_df = wind_df.assign(LON=wind_df["Longitude"])
wind_df.to_csv("data/Power_Plants_Wind_proc.csv")
