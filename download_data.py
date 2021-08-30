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

    for (outdir, outfile, file_url) in [
        ("data/grid", "Transmission_Lines_proc_sm.parq", "https://databyjp.s3-us-west-2.amazonaws.com/power_data/Transmission_Lines_proc_sm.parq"),
        ("data/resource", "nsrdb3_ghi_en_us_proc.csv", "https://databyjp.s3-us-west-2.amazonaws.com/power_data/nsrdb3_ghi_en_us_proc.csv"),
        ("data/resource", "wtk_conus_80m_mean_masked_proc.csv", "https://databyjp.s3-us-west-2.amazonaws.com/power_data/wtk_conus_80m_mean_masked_proc.csv"),
    ]:
        print(outdir, outfile, file_url)

        if not os.path.exists(outdir):
            logger.info(f"Output dir {outdir} NOT found, trying to create it")
            os.mkdir(outdir)
        else:
            logger.info(f"Output dir {outdir} found")

        file_outpath = os.path.join(outdir, outfile)
        if not os.path.exists(file_outpath):
            logger.info(f"Destination file(s) not found at {file_outpath}")
            if not os.path.exists(file_outpath):
                logger.info(f"File not found - downloading from {file_url}")
                r = requests.get(file_url, allow_redirects=True)
                open(file_outpath, 'wb').write(r.content)
                if os.path.exists(file_outpath):
                    logger.info("File successfully downloaded!")
                else:
                    logger.info("File still not found!")

        else:
            logger.info(f"Output file(s) found at {file_outpath}, skipping download")


if __name__ == '__main__':
    main()
