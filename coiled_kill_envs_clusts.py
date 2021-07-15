# ========== (c) JP Hwang 24/5/21  ==========

import logging
import pandas as pd
import numpy as np
import coiled

logger = logging.getLogger(__name__)

desired_width = 320
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", desired_width)


def kill_coiled_clusts():
    coiled_clusts = coiled.list_clusters()
    for k, v in coiled_clusts.items():
        logger.info(f"Killing cluster {k}")
        coiled.delete_cluster(k)
    return True


def kill_coiled_envs():
    coiled_envs = coiled.list_software_environments()
    for k, v in coiled_envs.items():
        logger.info(f"Killing environment {k}")
        coiled.delete_software_environment(k)
    return True


def main():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    sh.setFormatter(formatter)
    root_logger.addHandler(sh)

    # kill_coiled_envs()
    kill_coiled_clusts()


if __name__ == "__main__":
    main()
