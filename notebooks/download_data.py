# %%
import pandas as pd
from loguru import logger
import os
import requests
import backoff

IMAGE_DIR = "../data/clouds2/images/"

@backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=5, logger=logger)
def download_image_with_retry(url, filepath):
    response = requests.get(url, stream=True, timeout=10)
    if response.status_code == 200:
        with open(filepath, 'wb') as f:
            f.write(response.content)
        logger.success(f"Successfully downloaded {url} to {filepath}")
    else:
        logger.error(f"Failed to download {url}. Status code: {response.status_code}")

# %%
cols = ["Site ID", "Sky Color", "Sky Visibility", "Total Cloud Cover"]
image_cols = [
    "Ground Image North",
    "Ground Image East",
    "Ground Image West",
    "Ground Image South",
    "Ground Image Up",
]

data_df = pd.read_csv(
    "../data/2023_GLOBE-AnnualCloudData_2024-05-02_v3.2.csv",
    index_col=False,
    usecols=cols + image_cols,
)
logger.info(f"Loaded {len(data_df):,} rows")
os.makedirs(IMAGE_DIR, exist_ok=True)

# %%
# Filter rows: discard rows with no images FIRST
logger.info("Filtering out rows with no image URLs...")
has_image = data_df[image_cols].notna().any(axis=1)
data_df = data_df[has_image]
logger.info(f"Filtered DataFrame now has {len(data_df):,} rows with at least one image URL.")

# Calculate total images in the filtered dataset
n_images_total_available = data_df[image_cols].notna().sum().sum()
logger.info(f"Total potential images in the filtered dataset: {n_images_total_available}")

# %%
# Manual testing subset configuration
TEST_SUBSET_SIZE = 20 # Set to a small number for testing, or None to process all
# To process all, comment out or change data_df_to_process assignment below

if TEST_SUBSET_SIZE is not None:
    data_df_to_process = data_df.head(TEST_SUBSET_SIZE)
    n_images_to_download = data_df_to_process[image_cols].notna().sum().sum()
    logger.info(f"RUNNING IN TEST MODE: Processing only the first {TEST_SUBSET_SIZE} rows of the filtered DataFrame.")
    logger.info(f"This subset contains {n_images_to_download} potential images out of {n_images_total_available} total in the filtered dataset.")
else:
    data_df_to_process = data_df
    n_images_to_download = n_images_total_available
    logger.info("Processing the full filtered dataset.")

# %%
# Download logic
logger.info(f"Starting image download process. Will attempt to download images from {n_images_to_download} URLs in the current selection.")
images_processed_count = 0 # Counter for URLs processed in the current selection

for index, row in data_df_to_process.iterrows():
    for direction_column_name in image_cols:
        url = row[direction_column_name]
        if pd.notna(url):
            images_processed_count += 1 # Increment for each valid URL encountered
            logger.info(f"Attempting to download: URL {images_processed_count}/{n_images_to_download} in current selection.")
            simple_direction_str = direction_column_name.split(" ")[-1]
            filename = f"{index}_{simple_direction_str}.jpg"
            filepath_to_save = os.path.join(IMAGE_DIR, filename)
            download_image_with_retry(url, filepath_to_save)

logger.info("Image download process complete for the current selection.")

# %%
# Post-download summary (optional, could be removed or adjusted)
# This part might be redundant if TEST_SUBSET_SIZE is used, as n_images_total_available already gives the full count.
logger.info(f"Summary: Processed {len(data_df_to_process):,} rows.")
logger.info(f"Found {n_images_total_available:,} potential images in the original filtered dataset.")
if TEST_SUBSET_SIZE is not None:
    logger.info(f"Attempted to download {n_images_to_download} images from the subset of {TEST_SUBSET_SIZE} rows.")
logger.info(f"Estimated disk space for all {n_images_total_available} images: {n_images_total_available * 70 / 1e6:.2f} Gb total (approx 70kb/image).")

# %%
