#!/bin/bash
# Add this directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:`dirname "$(realpath $0)"`

# TO Accommodate peculiarities of gallery server only - most people can ignore the next two lines
pip install --upgrade pip
pip install -r requirements-predeploy.txt

# # -- WHEN USING COILED - UNHIDE THE NEXT TWO LINES --
# python coiled_login.py  # Log into coiled using token saved as env variable
# python coiled_create_env.py  # Create coiled env up front
# -- END - COILED SECTION --

# -- WHEN USING LOCAL COMPUTE - UNHIDE THE NEXT FOUR LINES --
dask-scheduler &
dask-worker 127.0.0.1:8786 --nprocs 4 --local-directory work-dir &
python download_data.py
python publish_data.py
# -- END - FOR LOCAL COMPUTE --

gunicorn "app:server" --timeout 120 --workers 2
wait
