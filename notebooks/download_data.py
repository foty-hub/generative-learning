# %%
import pandas as pd
from loguru import logger

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
# %%
# discard rows with no images
has_image = data_df[image_cols].notna().any(axis=1)
data_df = data_df[has_image]

# Find total number of images and total amount of data, at ~70kb per image
n_images = data_df[image_cols].notna().sum().sum()
logger.info(f"Found total {n_images:,} images")
logger.info(f"Estimated around {n_images * 70 / 1e6:.2f} Gb total")
# %%
