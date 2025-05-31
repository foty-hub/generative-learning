# %%
import os
import torchvision.transforms as T

from PIL import Image
from loguru import logger
from torch.utils.data import Dataset, DataLoader
import jax.numpy as jnp
import numpy as np
from einops import rearrange

# %%
root_folder = "../data/clouds2"
train_folder = f"{root_folder}/train"
test_folder = f"{root_folder}/test"

logger.info("Loading images")

dataset = []
# The images in the train dataset have discrete cloud type labels
for subfolder in os.listdir(train_folder):
    image_label = subfolder
    subfolder_path = f"{train_folder}/{subfolder}"
    for file in os.listdir(subfolder_path):
        img_path = f"{subfolder_path}/{file}"
        img = Image.open(img_path)
        datapoint = {
            "image": img,
            "label": image_label,
            "filepath": img_path,
        }
        dataset.append(datapoint)

n_train = len(dataset)

# The images in the test dataset have no labels and
# therefore a different folder structure
for file in os.listdir(test_folder):
    img_path = f"{test_folder}/{file}"
    img = Image.open(img_path)
    datapoint = {
        "image": img,
        "label": "NA",
        "filepath": img_path,
    }
    dataset.append(datapoint)
n_test = len(dataset) - n_train

logger.info(f"Loaded {len(dataset)} images ({n_train} train, {n_test} test)")


# %%
# Helper: PIL → JAX array in [-1, 1]
class ToArrayJNP:
    """Convert a PIL Image (H, W, C) to (C, H, W) float32 jax.numpy array scaled to [-1, 1]."""

    def __call__(self, img):
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5  # normalize to [-1, 1]
        arr = np.transpose(arr, (2, 0, 1))  # HWC → CHW
        return jnp.array(arr)


# ----------------------------------------------------------------------
# Utility: Convert a transformed JAX array back to a Pillow Image
def jnp_to_pil(img_jnp):
    """
    Convert a (C, H, W) float32 JAX array in the range [-1, 1]—as produced by
    `ToArrayJNP`—back into a standard Pillow Image.

    Args:
        img_jnp (jax.numpy.ndarray): Image tensor (C, H, W) with values in [-1, 1].

    Returns:
        PIL.Image.Image: Reconstructed image in RGB (or grayscale) format.
    """
    # De-normalize from [-1, 1] → [0, 255]
    img_jnp = (img_jnp * 0.5 + 0.5) * 255.0
    img_jnp = jnp.clip(img_jnp, 0, 255).astype(jnp.uint8)
    img_jnp = rearrange(img_jnp, "b h w -> h w b")
    # Drop the channel dim if single‑channel (grayscale)
    if img_jnp.shape[2] == 1:
        img_jnp = img_jnp[:, :, 0]
    return Image.fromarray(np.array(img_jnp))


train_transforms = T.Compose(
    [
        T.RandomCrop(256, pad_if_needed=False),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2, 0.1),
        ToArrayJNP(),
    ]
)

val_transforms = T.Compose(
    [
        T.CenterCrop(256),
        ToArrayJNP(),
    ]
)


# %%
# -------------------- DATASET + DATALOADERS --------------------
# Build mapping from string labels to integer indices using the training set
#
# Torch DataLoader collate_fn that converts each batch to jax.numpy
def jnp_collate(batch):
    imgs, labels = zip(*batch)
    return jnp.stack(imgs), jnp.array(labels)


class CloudDataset(Dataset):
    """Wraps the raw list-of-dicts produced above and applies transforms."""

    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform
        self.label_set = sorted({d["label"] for d in dataset[:n_train]})
        self.label_to_idx = {lbl: idx for idx, lbl in enumerate(self.label_set)}

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        entry = self.data_list[idx]
        img = entry["image"]
        label_str = entry["label"]
        # Convert string label to an integer; unlabeled test images get -1
        label_idx = (
            self.label_to_idx[label_str] if label_str in self.label_to_idx else -1
        )
        if self.transform:
            img = self.transform(img)
        return img, label_idx


# Split the flat list into training and validation/test portions
train_data = dataset[:n_train]
val_data = dataset[n_train:]

train_dataset = CloudDataset(train_data, transform=train_transforms)
val_dataset = CloudDataset(val_data, transform=val_transforms)


def get_dataloaders(batch_size=32, num_workers=0, with_val=True):
    logger.info("Generating dataloaders")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=jnp_collate,
    )
    if with_val:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=jnp_collate,
        )
        return train_loader, val_loader
    return train_loader


# %%
train_loader = get_dataloaders(with_val=False)
# %%
for X in train_loader:
    break
# %%
# imgarray = X[0][0]
# jnp_to_pil(imgarray)
# %%
