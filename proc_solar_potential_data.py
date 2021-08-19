# ========== (c) JP Hwang 18/8/21  ==========

import logging
import pandas as pd
import numpy as np
import shapely
import datetime
import random
import spatialpandas  # Necessary for reading the parquet file with multipolygon data

'''
Where does the data come from?
    Raster files from here - Solar: https://www.nrel.gov/gis/solar-resource-maps.html | Wind: https://www.nrel.gov/gis/wind-resource-maps.html

Convert the raster (TIF) file to a readable format:
    Install raster2xyz (pip install raster2xyz)
    You may need to install GDAL (https://formulae.brew.sh/formula/gdal)
    Run on shell: 
        raster2xyz path_to/nsrdb3_ghi.tif path_to/nsrdb3_ghi.csv
'''

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

from utils import ll2en
solar_df = pd.read_csv("temp/nsrdb3_ghi.csv")
vals_en = np.array(ll2en(solar_df[["x", "y"]].values))
solar_df = solar_df.assign(x_en=vals_en[:, 0])
solar_df = solar_df.assign(y_en=vals_en[:, 1])
solar_df = solar_df.assign(LON=solar_df["x"])
solar_df = solar_df.assign(LAT=solar_df["y"])
solar_df.to_csv("temp/nsrdb3_ghi_en.csv")

# Filter for continental US (-124.848974, 24.396308) - (-66.885444, 49.384358)
solar_df_us = solar_df[(solar_df["x"] > -125) & (solar_df["x"] < -66) & (solar_df["y"] > 24) & (solar_df["y"] < 50)]

# Try to filter / avg data for state
state_df = pd.read_parquet("temp/state_data_proc.parq")  # need to import spatialpandas first

# Add state column to solar_df_us
solar_df_us = solar_df_us.assign(state=None)
sample_size = min(1000000, len(solar_df_us))
solar_df_us_sm = solar_df_us.sample(n=sample_size, random_state=42)  # Sampling for speed

starttime = datetime.datetime.now()
counter = 0
looplen = len(solar_df_us_sm)

for j in range(looplen):
    tmp_x = solar_df_us_sm.iloc[j]["x"]
    tmp_y = solar_df_us_sm.iloc[j]["y"]
    tmp_pt = shapely.geometry.Point((tmp_x, tmp_y))
    tmp_states = state_df[(state_df["lonmin"] < tmp_x) & (state_df["lonmax"] > tmp_x) & (state_df["latmin"] < tmp_y) & (state_df["latmax"] > tmp_y)]  # This speeds up the process by over an order of magnitude!
    if len(tmp_states) > 0:
        for i, state_row in tmp_states.iterrows():
            if state_row.geometry_ll.to_shapely().contains(tmp_pt):
                solar_df_us_sm.loc[solar_df_us_sm.iloc[j].name, "state"] = state_row["STATEFP"]
                counter += 1
                break
    if j+1 % 1000 == 0:
        logger.info(f"Processed {j} rows")

proclen = datetime.datetime.now() - starttime
print(f"Found {counter} of {looplen} rows in the U.S.!")
print(f"Took {proclen.total_seconds()} seconds (with pre-filtering of state_df)")

solar_df_us_sm.to_csv(f"temp/nsrdb3_ghi_en_us_states_{sample_size}.csv")

solar_df_us_sm = solar_df_us_sm[solar_df_us_sm["state"].notna()]  # If no need for data from outside the U.S. at all
solar_df_us_sm.to_csv(f"data/nsrdb3_ghi_en_us_states_only.csv")
