import os
coiled_token = os.environ['COILED_TOKEN']
os.system(f"coiled login --token {coiled_token}")
