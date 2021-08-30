# ========== (c) JP Hwang 18/8/21  ==========

import logging
import pandas as pd
import numpy as np
from pyproj import Transformer, CRS

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

# To create the CSV file, run the below command from shell with the original files from NREL
# raster2xyz temp/wtk_conus_80m_mean_masked.tif temp/wtk_conus_80m_mean_masked.csv

CRSstr = CRS.from_string("+proj=lcc +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs")  # ESRI:102009
en_outfile = f"data/resource/wtk_conus_80m_mean_masked_proc.csv"

logger.info(f"{en_outfile} not found - creating")
wind_df = pd.read_csv("temp/wtk_conus_80m_mean_masked.csv")
transformer_wind_to_4326 = Transformer.from_crs(CRSstr, "epsg:4326")
transformer_wind_to_3857 = Transformer.from_crs(CRSstr, "epsg:3857")


def wind2ll(coords):
    return [list(reversed(transformer_wind_to_4326.transform(*row))) for row in coords]


def wind2en(coords):
    return [list(transformer_wind_to_3857.transform(*row)) for row in coords]

logger.info("Transforming location data")
vals_ll = np.array(wind2ll(wind_df[["x", "y"]].values))
vals_en = np.array(wind2en(wind_df[["x", "y"]].values))
wind_df = wind_df.assign(x_en=vals_en[:, 0])
wind_df = wind_df.assign(y_en=vals_en[:, 1])
wind_df = wind_df.assign(LON=vals_ll[:, 0])
wind_df = wind_df.assign(LAT=vals_ll[:, 1])
logger.info("Saving transformed data")

wind_df = wind_df[wind_df.z > 0]  # Remove many null data rows
wind_df.to_csv(en_outfile, index=False)
