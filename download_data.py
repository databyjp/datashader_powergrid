# ========== (c) JP Hwang 16/5/21  ==========

import logging
import os
import requests
import pandas as pd
import zipfile

logger = logging.getLogger(__name__)

desired_width = 320
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', desired_width)


def main():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    sh.setFormatter(formatter)
    root_logger.addHandler(sh)

    outdir = 'data/grid'
    if not os.path.exists(outdir):
        logger.info(f"Output dir {outdir} NOT found, trying to create it")
        os.mkdir(outdir)
    else:
        logger.info(f"Output dir {outdir} found")

    for outfile, parq_url in {
        "Transmission_Lines_proc_sm.parq": "https://databyjp.s3-us-west-2.amazonaws.com/power_data/Transmission_Lines_proc_sm.parq",
    }.items():
        parq_outpath = os.path.join(outdir, outfile)
        if not os.path.exists(parq_outpath):
            logger.info(f"Parq file(s) not found at {parq_outpath}")
            if not os.path.exists(parq_outpath):
                logger.info(f"File not found - downloading from {parq_url}")
                r = requests.get(parq_url, allow_redirects=True)
                open(parq_outpath, 'wb').write(r.content)
                if os.path.exists(parq_outpath):
                    logger.info("File successfully downloaded!")
                else:
                    logger.info("File still not found!")

        else:
            logger.info(f"Parq file(s) found at {parq_outpath}, skipping download/unzipping")


if __name__ == '__main__':
    main()
