import os
import zipfile
import gdown
from loguru import logger
import numpy as np
import jax.numpy as jnp
from PIL import Image
from einops import rearrange
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt # Added

IMG_ZIP_ID = "0B7EVK8r0v71pZjFTYXZWM3FlRnM"
ATTR_TXT_ID = "0B7EVK8r0v71pblRyaVFSWGxPY0U"

# Define CelebA dataset split ranges (1-indexed as per original dataset info)
TRAIN_SPLIT_END = 162770
VALIDATION_SPLIT_END = 182637
# TEST_SPLIT_END is implicitly 202599 (total images)

# Target image dimensions and cropping for transformations
TARGET_IMG_SIZE = (128, 128)
CENTER_CROP_SIZE = 178 # CelebA original images are 218 H x 178 W, crop to width

# Helper: PIL → JAX array in [-1, 1]
class ToArrayJNP:
    """Convert a PIL Image (H, W, C) to (C, H, W) float32 jax.numpy array scaled to [-1, 1]."""

    def __call__(self, img):
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5  # normalize to [-1, 1]
        arr = np.transpose(arr, (2, 0, 1))  # HWC → CHW
        return jnp.array(arr)

# Collate function for JAX arrays
def jnp_collate(batch):
    """Collate function to process a batch of (image, attributes) pairs from CelebADataset."""
    imgs, attrs = zip(*batch)
    batched_imgs = jnp.stack(imgs)
    batched_attrs = jnp.stack(attrs)
    return batched_imgs, batched_attrs

def train_transforms_fn(augmentation=True):
    """
    Returns a composed PyTorch transform for training.
    Includes optional augmentation.
    """
    transforms_list = [
        T.CenterCrop(CENTER_CROP_SIZE),
        T.Resize(TARGET_IMG_SIZE),
    ]
    if augmentation:
        transforms_list.append(T.RandomHorizontalFlip())
    transforms_list.append(ToArrayJNP())
    return T.Compose(transforms_list)

def eval_transforms_fn():
    """
    Returns a composed PyTorch transform for evaluation/testing.
    """
    return T.Compose([
        T.CenterCrop(CENTER_CROP_SIZE),
        T.Resize(TARGET_IMG_SIZE),
        ToArrayJNP(),
    ])

def download_celeba(root_dir="."):
    """
    Downloads and extracts the CelebA dataset.
    """
    celeba_dir = os.path.join(root_dir, "celeba")
    img_dir = os.path.join(celeba_dir, "img_align_celeba")
    attr_file_path = os.path.join(celeba_dir, "list_attr_celeba.txt")
    img_zip_path = os.path.join(celeba_dir, "img_align_celeba.zip")

    os.makedirs(celeba_dir, exist_ok=True)
    logger.info(f"Ensured CelebA directory exists: {celeba_dir}")

    if not os.path.exists(attr_file_path):
        logger.info(f"Downloading list_attr_celeba.txt to {attr_file_path}...")
        try:
            gdown.download(id=ATTR_TXT_ID, output=attr_file_path, quiet=False)
            logger.info(f"Downloaded list_attr_celeba.txt to {attr_file_path}.")
        except Exception as e:
            logger.error(f"Failed to download list_attr_celeba.txt: {e}")
            # Depending on the desired behavior, you might want to re-raise or exit
            raise
    else:
        logger.info(f"Attribute file {attr_file_path} already exists. Skipping download.")

    if not os.path.isdir(img_dir):
        logger.info(f"Image directory {img_dir} not found. Proceeding with download and extraction.")
        if not os.path.exists(img_zip_path):
            logger.info(f"Downloading img_align_celeba.zip to {img_zip_path}...")
            try:
                gdown.download(id=IMG_ZIP_ID, output=img_zip_path, quiet=False)
                logger.info(f"Downloaded img_align_celeba.zip to {img_zip_path}.")
            except Exception as e:
                logger.error(f"Failed to download img_align_celeba.zip: {e}. This can happen due to download limits.")
                logger.info("If the zip file exists but is corrupted, it will be handled during extraction attempt.")
                # Don't re-raise here, allow to proceed if zip exists or extraction handles it
        else:
            logger.info(f"Zip file {img_zip_path} already exists. Skipping download, proceeding to extraction.")

        if os.path.exists(img_zip_path): # Proceed only if zip file is present
            logger.info(f"Extracting {img_zip_path} to {celeba_dir}...")
            try:
                with zipfile.ZipFile(img_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(celeba_dir)
                logger.info(f"Successfully extracted images to {img_dir}.")
                logger.info(f"Removing {img_zip_path}...")
                os.remove(img_zip_path)
                logger.info(f"Successfully removed {img_zip_path}.")
            except zipfile.BadZipFile:
                logger.error(f"Error: Failed to extract {img_zip_path}. The file might be corrupted. Please delete it and try again.")
                # Optionally remove the corrupted zip: if os.path.exists(img_zip_path): os.remove(img_zip_path)
                raise # Re-raise as this is critical for images
            except Exception as e:
                logger.error(f"Extraction error: {e}")
                raise # Re-raise for other critical extraction errors
        else:
            logger.warning(f"Zip file {img_zip_path} not found. Cannot extract images. Please ensure it's downloaded.")
            # This implies the image directory won't exist, which CelebADataset will handle by erroring.
    else:
        logger.info(f"Image directory {img_dir} already exists. Skipping download and extraction.")


class CelebADataset(Dataset):
    """CelebA Dataset compatible with PyTorch DataLoader."""
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_dir = os.path.join(root_dir, "celeba", "img_align_celeba")
        self.attr_file_path = os.path.join(root_dir, "celeba", "list_attr_celeba.txt")
        self.attributes_names = [] # To store attribute names

        if not os.path.exists(self.attr_file_path):
            logger.error(f"Attribute file not found: {self.attr_file_path}. Run download_celeba first.")
            raise FileNotFoundError(f"Attribute file not found: {self.attr_file_path}")

        if not os.path.isdir(self.img_dir):
            logger.warning(f"Image directory not found: {self.img_dir}. Dataset may be unusable if images are missing.")
            # No immediate raise, __getitem__ will fail if specific images are missing.

        try:
            with open(self.attr_file_path, 'r') as f:
                _ = int(f.readline().strip())
                self.attributes_names = f.readline().strip().split() # Store attribute names
                full_attr_df = pd.read_csv(f, sep='\s+', header=None, names=self.attributes_names, index_col=0)
        except Exception as e:
            logger.error(f"Error reading attribute file {self.attr_file_path}: {e}")
            raise

        if split == "train": split_df = full_attr_df.iloc[:TRAIN_SPLIT_END]
        elif split == "valid": split_df = full_attr_df.iloc[TRAIN_SPLIT_END:VALIDATION_SPLIT_END]
        elif split == "test": split_df = full_attr_df.iloc[VALIDATION_SPLIT_END:]
        else: raise ValueError(f"Unknown split: {split}. Choose from 'train', 'valid', 'test'.")

        self.filenames = split_df.index.tolist()
        self.attributes = jnp.array(split_df.values, dtype=jnp.int32)
        logger.info(f"Initialized CelebADataset for split '{split}' with {len(self.filenames)} images. Attributes loaded: {len(self.attributes_names)}")

    def __len__(self): return len(self.filenames)
    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        img_path = os.path.join(self.img_dir, self.filenames[idx])
        try: image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            logger.error(f"Image not found: {img_path}. Ensure dataset is fully downloaded and extracted.")
            raise
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            raise
        attrs = self.attributes[idx]
        if self.transform: image = self.transform(image)
        return image, attrs

def get_celeba_dataloader(root_dir, batch_size, split="train", shuffle=None, augmentation=None, num_workers=0, pin_memory=True):
    """
    Creates a PyTorch DataLoader for the CelebA dataset.
    """
    if shuffle is None: shuffle = (split == "train")
    if augmentation is None: augmentation = (split == "train")

    if split == "train": transform = train_transforms_fn(augmentation=augmentation)
    else: transform = eval_transforms_fn()

    dataset = CelebADataset(root_dir=root_dir, split=split, transform=transform)

    if len(dataset) == 0:
        logger.warning(f"Dataset for split '{split}' is empty. DataLoader will also be empty.")
        # Return an empty loader or handle as per requirements. Here, it will create an empty loader.

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        collate_fn=jnp_collate, pin_memory=pin_memory, drop_last=(split == "train")
    )
    logger.info(f"Created DataLoader for split '{split}': batch_size={batch_size}, shuffle={shuffle}, augmentation={augmentation if split=='train' else False}, num_workers={num_workers}, images={len(dataset)}")
    return loader

# Utility: Convert a transformed JAX array back to a Pillow Image
def jnp_to_pil(img_jnp):
    img_jnp = (img_jnp * 0.5 + 0.5) * 255.0
    img_jnp = jnp.clip(img_jnp, 0, 255).astype(jnp.uint8)
    img_jnp = rearrange(img_jnp, "c h w -> h w c")
    if img_jnp.shape[2] == 1: img_jnp = img_jnp[:, :, 0]
    return Image.fromarray(np.array(img_jnp))

if __name__ == '__main__':
    logger.info("Starting CelebA data loader script demo...")
    # Setup Matplotlib to use a non-GUI backend if no GUI is available (e.g., in a container)
    # This needs to be done before importing pyplot for the first time in some cases.
    try:
        os.environ['DISPLAY']
    except KeyError:
        logger.info("No DISPLAY environment variable found, attempting to use 'Agg' backend for Matplotlib.")
        plt.switch_backend('Agg')


    data_directory = "data"
    batch_size = 4
    os.makedirs(data_directory, exist_ok=True)

    logger.info(f"Step 1: Ensuring CelebA dataset is available in '{data_directory}/celeba'.")
    try:
        download_celeba(data_directory) # Download/verify dataset
        logger.info("Dataset download/check completed.")
    except Exception as e:
        logger.error(f"Critical error during dataset download/preparation: {e}")
        logger.error("Cannot proceed with DataLoader demo without the dataset.")
        exit(1) # Exit if dataset preparation fails

    logger.info(f"\nStep 2: Getting DataLoaders...")
    try:
        logger.info(f"--- Getting TRAIN DataLoader ---")
        train_loader = get_celeba_dataloader(data_directory, batch_size=batch_size, split="train", augmentation=True, num_workers=0)

        if len(train_loader.dataset) > 0:
            logger.info(f"Train DataLoader created. Accessing one batch...")
            train_images_batch, train_attrs_batch = next(iter(train_loader))
            logger.info(f"Train batch images shape: {train_images_batch.shape}, dtype: {train_images_batch.dtype}")
            logger.info(f"Train batch attributes shape: {train_attrs_batch.shape}, dtype: {train_attrs_batch.dtype}")

            logger.info(f"\nStep 3: Visualizing a sample from the TRAIN batch...")
            sample_image_jnp = train_images_batch[0]
            sample_attrs_jnp = train_attrs_batch[0]

            pil_image = jnp_to_pil(sample_image_jnp)
            logger.info(f"Sample image (from batch[0]) converted back to PIL: size {pil_image.size}, mode {pil_image.mode}")

            # Log attributes for the sample image
            # Attribute names are stored in the dataset object
            attr_names = train_loader.dataset.attributes_names
            if attr_names:
                logger.info("Attributes for the sample image (0/1, from -1/1):")
                attr_str_list = []
                for i, name in enumerate(attr_names):
                    if sample_attrs_jnp[i] == 1: # Or use original -1/1 values if preferred
                        attr_str_list.append(name)
                if not attr_str_list:
                    logger.info("  (No positive attributes for this sample or all attributes are -1)")
                else:
                     # Log only a few attributes if many are positive to keep log concise
                    logged_attrs = attr_str_list[:5] # Log first 5 positive attributes
                    logger.info(f"  {', '.join(logged_attrs)}{'...' if len(attr_str_list) > 5 else ''}")
            else:
                logger.warning("Attribute names not available in dataset object for detailed logging.")


            try:
                plt.figure(figsize=(6,6))
                plt.imshow(pil_image)
                plt.title(f"Sample Image from Train DataLoader\nSize: {TARGET_IMG_SIZE}")
                plt.axis('off')
                # Try to save instead of show for non-interactive environments
                plot_filename = "sample_celeba_image.png"
                plt.savefig(plot_filename)
                logger.info(f"Sample image saved to {plot_filename}")
                # plt.show() # This might block or fail in non-GUI envs
            except Exception as e:
                logger.error(f"Matplotlib visualization failed: {e}")
                logger.info("This can happen in environments without a GUI. Check the saved image if available.")

        else:
            logger.warning("Train dataset is empty. Cannot demonstrate DataLoader or visualization.")

        logger.info(f"\n--- Getting VALIDATION DataLoader ---")
        valid_loader = get_celeba_dataloader(data_directory, batch_size=batch_size, split="valid", num_workers=0)
        if len(valid_loader.dataset) > 0:
            logger.info(f"Validation DataLoader created. Accessing one batch...")
            valid_images, valid_attrs = next(iter(valid_loader))
            logger.info(f"Validation batch images shape: {valid_images.shape}")
            logger.info(f"Validation batch attributes shape: {valid_attrs.shape}")
        else:
            logger.warning("Validation dataset is empty.")

    except FileNotFoundError as e:
        logger.error(f"Could not create DataLoader: {e}. ")
        logger.error("Ensure 'list_attr_celeba.txt' exists and images are extracted in the 'data/celeba' directory.")
        logger.error("You might need to run the script once for `download_celeba` to complete fully, especially if `img_align_celeba.zip` download was interrupted previously.")
    except Exception as e:
        logger.error(f"An error occurred during DataLoader example usage: {e}")
        logger.exception("Detailed traceback:")
    logger.info("\nCelebA data loader script demo finished.")
