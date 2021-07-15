# ========== (c) JP Hwang 8/6/21  ==========

import logging
import pandas as pd
import numpy as np
import coiled

logger = logging.getLogger(__name__)

desired_width = 320
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", desired_width)


def main():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    sh.setFormatter(formatter)
    root_logger.addHandler(sh)

    coiled.create_software_environment(
        name="grid-app-env", pip="coiled_requirements.txt",
    )


if __name__ == "__main__":
    main()
