import logging

logger = logging.getLogger(__name__)

import os
import numpy as np
from distributed import Client
from spatialpandas.io import read_parquet_dask
from utils import scheduler_url


if __name__ == "__main__":
    print(f"Connecting to cluster at {scheduler_url} ... ", end="")
    client = Client(scheduler_url)
    print("done")

    grid_fpath = os.path.join("data/grid", "Transmission_Lines_proc_sm.parq")

    logger.info("Loading Parquet data")
    df = read_parquet_dask(grid_fpath)

    # Preprocessing
    df["SHAPE_Leng"] = df["SHAPE_Leng"].astype(np.float32)
    df["VOLTAGE"] = df["VOLTAGE"].astype(np.float32)
    df["lon_a"] = df["lon_a"].astype(np.float32)
    df["lat_a"] = df["lat_a"].astype(np.float32)
    df["lon_b"] = df["lon_b"].astype(np.float32)
    df["lat_b"] = df["lat_b"].astype(np.float32)
    df["LAT"] = df["LAT"].astype(np.float32)
    df["LON"] = df["LON"].astype(np.float32)
    df["x_en"] = df["x_en"].astype(np.float32)
    df["y_en"] = df["y_en"].astype(np.float32)
    df = df.assign(TYPE=df["TYPE"].astype("category"))
    df = df.assign(STATUS=df["STATUS"].astype("category"))

    # Filter out rows with unknown voltage
    df = df[df["VOLTAGE"] > 0]

    df = df.persist()

    # Clear any published datasets
    for k in client.list_datasets():
        client.unpublish_dataset(k)

    client.publish_dataset(df=df)
