import os
import sys
import pytest
import numpy as np
import jax.numpy as jnp
from PIL import Image
import zipfile
from unittest.mock import patch, MagicMock

# Add the parent directory to sys.path to allow imports from celeba_loader
# This assumes the tests are run from the repository root or tests/ directory.
# A more robust solution might involve package structure or pytest path configuration.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from celeba_loader import (
    download_celeba,
    CelebADataset,
    get_celeba_dataloader,
    ToArrayJNP,
    jnp_to_pil,
    train_transforms_fn,
    eval_transforms_fn,
    jnp_collate,
    logger, # import logger to manage it
    # Constants that might be monkeypatched
    TARGET_IMG_SIZE,
    CENTER_CROP_SIZE,
    TRAIN_SPLIT_END,
    VALIDATION_SPLIT_END
)

# Suppress loguru logs during tests, or set to a higher level
logger.remove()
logger.add(sys.stderr, level="ERROR") # Only show errors and critical

# --- Fixtures ---

@pytest.fixture
def dummy_celeba_root(tmp_path):
    """Creates a dummy root directory for CelebA data within tmp_path."""
    test_data_root = tmp_path / "celeba_test_data"
    test_data_root.mkdir()
    # Create the 'celeba' subdirectory as expected by the loader
    (test_data_root / "celeba").mkdir(exist_ok=True)
    return test_data_root

@pytest.fixture
def dummy_attr_file(dummy_celeba_root):
    """Creates a dummy list_attr_celeba.txt file."""
    celeba_dir = dummy_celeba_root / "celeba"
    attr_file_path = celeba_dir / "list_attr_celeba.txt"
    content = """3
Smiling Male Young
000001.jpg 1 -1 1
000002.jpg -1 1 1
000003.jpg 1 1 -1
"""
    with open(attr_file_path, "w") as f:
        f.write(content)
    return attr_file_path

@pytest.fixture
def dummy_image_files(dummy_celeba_root):
    """Creates dummy image files in the expected img_align_celeba directory."""
    img_dir_path = dummy_celeba_root / "celeba" / "img_align_celeba"
    img_dir_path.mkdir(exist_ok=True)

    image_names = ["000001.jpg", "000002.jpg", "000003.jpg"]
    image_paths = []
    for i, name in enumerate(image_names):
        img = Image.new("RGB", (20, 20), color=f"rgb({i*50},{i*50},{i*50})") # Small, distinct images
        path = img_dir_path / name
        img.save(path)
        image_paths.append(path)
    return img_dir_path, image_paths


# --- Test Image Utilities ---

def test_to_array_jnp():
    """Tests the ToArrayJNP transform."""
    converter = ToArrayJNP()
    # Create a dummy PIL Image (H, W, C)
    pil_img = Image.new("RGB", (32, 28), color="blue") # H=28, W=32

    jnp_array = converter(pil_img)

    assert isinstance(jnp_array, jnp.ndarray)
    # Expected shape (C, H, W)
    assert jnp_array.shape == (3, 28, 32)
    assert jnp_array.dtype == jnp.float32

    # Check value range [-1, 1]
    # Original image is blue (0,0,255). Normalized: R,G=-1, B=1
    # (0/255 - 0.5) / 0.5 = -1
    # (255/255 - 0.5) / 0.5 = 1
    # Due to float precision, use approx
    assert jnp.allclose(jnp_array[0,0,0], -1.0) # R channel
    assert jnp.allclose(jnp_array[1,0,0], -1.0) # G channel
    assert jnp.allclose(jnp_array[2,0,0],  1.0) # B channel


def test_jnp_to_pil():
    """Tests the jnp_to_pil conversion utility."""
    # Create a dummy JAX array (C, H, W)
    # Example: 3 channels, 28x32 image
    # Values are in [-1, 1] range
    c, h, w = 3, 28, 32
    jnp_array = jnp.ones((c, h, w), dtype=jnp.float32)
    # Make it more interesting: R=-1, G=0, B=1
    jnp_array = jnp_array.at[0,:,:].set(-1.0) # R channel
    jnp_array = jnp_array.at[1,:,:].set(0.0)  # G channel (normalized to 0.5 for PIL)
    jnp_array = jnp_array.at[2,:,:].set(1.0)  # B channel

    pil_img = jnp_to_pil(jnp_array)

    assert isinstance(pil_img, Image.Image)
    assert pil_img.mode == "RGB"
    assert pil_img.size == (w, h) # PIL size is (W, H)

    # Check pixel values after conversion
    # R=-1 -> ( -1*0.5 + 0.5 ) * 255 = 0
    # G=0  -> (  0*0.5 + 0.5 ) * 255 = 127.5 -> 127 (uint8)
    # B=1  -> (  1*0.5 + 0.5 ) * 255 = 255
    expected_pixel = (0, 127, 255)
    assert pil_img.getpixel((0,0))[:3] == expected_pixel

    # Test grayscale conversion
    jnp_gray = jnp.ones((1, h, w), dtype=jnp.float32) * 0.5 # Value 0.5 -> (0.5*0.5+0.5)*255 = 191.25 -> 191
    pil_gray_img = jnp_to_pil(jnp_gray)
    assert pil_gray_img.mode == "L"
    assert pil_gray_img.size == (w,h)
    assert pil_gray_img.getpixel((0,0)) == 191

    # Test conversion back and forth (approximate due to float/int conversions)
    original_jnp = (jnp.array(np.random.rand(c,h,w), dtype=jnp.float32) * 2.0) - 1.0 # Random in [-1,1]
    reconverted_pil = jnp_to_pil(original_jnp)
    reconverted_jnp = ToArrayJNP()(reconverted_pil)
    assert jnp.allclose(original_jnp, reconverted_jnp, atol=0.01) # Tolerance for uint8 intermediate step


# --- Test CelebADataset ---
# We need to be able to change constants in celeba_loader for these tests
# or use very specific dummy data that matches default constants.
# For now, let's try monkeypatching constants for CelebADataset tests.

def test_celeba_dataset_splits_and_content(dummy_celeba_root, dummy_attr_file, dummy_image_files, monkeypatch):
    """Tests CelebADataset for different splits and content loading."""

    # Original image size in dummy_image_files is (20,20)
    # Original attributes: Smiling, Male, Young
    # 000001.jpg  1 -1  1
    # 000002.jpg -1  1  1
    # 000003.jpg  1  1 -1

    # Monkeypatch the split points in celeba_loader module for this test
    monkeypatch.setattr("celeba_loader.TRAIN_SPLIT_END", 1)
    monkeypatch.setattr("celeba_loader.VALIDATION_SPLIT_END", 2)
    # Test split will be image 3

    # Test Train Split
    dataset_train = CelebADataset(root_dir=dummy_celeba_root, split="train", transform=None)
    assert len(dataset_train) == 1
    assert dataset_train.attributes_names == ["Smiling", "Male", "Young"]

    img1, attrs1 = dataset_train[0]
    assert isinstance(img1, Image.Image)
    assert img1.size == (20, 20) # dummy image size
    assert jnp.array_equal(attrs1, jnp.array([1, -1, 1], dtype=jnp.int32))
    assert dataset_train.filenames[0] == "000001.jpg"

    # Test Validation Split
    dataset_valid = CelebADataset(root_dir=dummy_celeba_root, split="valid", transform=None)
    assert len(dataset_valid) == 1
    img2, attrs2 = dataset_valid[0]
    assert isinstance(img2, Image.Image)
    assert jnp.array_equal(attrs2, jnp.array([-1, 1, 1], dtype=jnp.int32))
    assert dataset_valid.filenames[0] == "000002.jpg"

    # Test Test Split
    dataset_test = CelebADataset(root_dir=dummy_celeba_root, split="test", transform=None)
    assert len(dataset_test) == 1
    img3, attrs3 = dataset_test[0]
    assert isinstance(img3, Image.Image)
    assert jnp.array_equal(attrs3, jnp.array([1, 1, -1], dtype=jnp.int32))
    assert dataset_test.filenames[0] == "000003.jpg"

    # Test with ToArrayJNP transform
    dataset_transformed = CelebADataset(root_dir=dummy_celeba_root, split="train", transform=ToArrayJNP())
    img_t, attrs_t = dataset_transformed[0]
    assert isinstance(img_t, jnp.ndarray)
    assert img_t.shape == (3, 20, 20) # C, H, W for dummy images
    assert jnp.array_equal(attrs_t, jnp.array([1, -1, 1], dtype=jnp.int32))

def test_celeba_dataset_file_not_found(dummy_celeba_root, dummy_image_files):
    """Tests FileNotFoundError if attribute file is missing."""
    with pytest.raises(FileNotFoundError, match="Attribute file not found"):
        CelebADataset(root_dir=dummy_celeba_root, split="train")

    # Test if image file is missing (requires attr file to exist first)
    dummy_attr_file(dummy_celeba_root) # Create attr file
    dataset = CelebADataset(root_dir=dummy_celeba_root, split="train")

    # Temporarily remove an image file to test __getitem__
    img_dir_path, image_paths = dummy_image_files
    os.remove(image_paths[0])

    with pytest.raises(FileNotFoundError, match="Image not found"):
        _ = dataset[0] # Accessing the removed image


# --- Test Download Functionality ---
@patch('celeba_loader.gdown.download')
@patch('celeba_loader.zipfile.ZipFile')
@patch('celeba_loader.os.remove')
def test_download_celeba_new_download_and_extract(mock_os_remove, mock_zipfile, mock_gdown_download, dummy_celeba_root):
    """Tests download_celeba when files are not present and need downloading."""
    # Configure mock_zipfile to behave like a context manager
    mock_zip_instance = MagicMock()
    mock_zipfile.return_value.__enter__.return_value = mock_zip_instance

    download_celeba(root_dir=dummy_celeba_root)

    # Assert gdown.download was called for attr file and image zip
    assert mock_gdown_download.call_count == 2
    # Check call arguments if needed, e.g., by inspecting mock_gdown_download.call_args_list

    # Assert ZipFile was opened and extractall was called
    mock_zipfile.assert_called_once()
    mock_zip_instance.extractall.assert_called_once_with(dummy_celeba_root / "celeba")

    # Assert os.remove was called for the zip file
    mock_os_remove.assert_called_once()
    # Example: mock_os_remove.assert_called_once_with(dummy_celeba_root / "celeba" / "img_align_celeba.zip")


@patch('celeba_loader.gdown.download')
@patch('celeba_loader.zipfile.ZipFile')
@patch('celeba_loader.os.remove')
def test_download_celeba_files_exist(mock_os_remove, mock_zipfile, mock_gdown_download,
                                     dummy_celeba_root, dummy_attr_file, dummy_image_files):
    """Tests download_celeba when files and image directory already exist."""

    # dummy_attr_file and dummy_image_files fixtures ensure the files/dirs are created

    download_celeba(root_dir=dummy_celeba_root)

    mock_gdown_download.assert_not_called()
    mock_zipfile.assert_not_called()
    mock_os_remove.assert_not_called()

# More tests to be added for transforms, collate, and dataloader
# For example:
# test_transforms_output_shape_type
# test_jnp_collate_fn
# test_get_celeba_dataloader_output
# test_train_transforms_augmentation_effect (more complex)

# Placeholder for future transform tests
def test_transforms_output_shape_type(monkeypatch):
    monkeypatch.setattr("celeba_loader.TARGET_IMG_SIZE", (16, 16)) # Use smaller target for test
    monkeypatch.setattr("celeba_loader.CENTER_CROP_SIZE", 18)    # Crop to 18x18 first

    # Dummy PIL image (original size 20x20, as from dummy_image_files)
    pil_img = Image.new("RGB", (20, 20))

    train_t_no_aug = train_transforms_fn(augmentation=False)
    eval_t = eval_transforms_fn()

    transformed_train = train_t_no_aug(pil_img)
    transformed_eval = eval_t(pil_img)

    expected_shape = (3, 16, 16) # C, H, W (TARGET_IMG_SIZE)
    assert transformed_train.shape == expected_shape
    assert transformed_train.dtype == jnp.float32
    assert transformed_eval.shape == expected_shape
    assert transformed_eval.dtype == jnp.float32

# Placeholder for collate function test
def test_jnp_collate_fn():
    # Create a batch of data (list of (img_jnp, attr_jnp) tuples)
    # img_jnp: (C,H,W), attr_jnp: (num_attrs,)
    batch = [
        (jnp.ones((3, 16, 16), dtype=jnp.float32), jnp.array([1, -1, 1], dtype=jnp.int32)),
        (jnp.zeros((3, 16, 16), dtype=jnp.float32), jnp.array([-1, 1, -1], dtype=jnp.int32))
    ]

    batched_imgs, batched_attrs = jnp_collate(batch)

    assert isinstance(batched_imgs, jnp.ndarray)
    assert batched_imgs.shape == (2, 3, 16, 16) # Batch, C, H, W
    assert batched_imgs.dtype == jnp.float32

    assert isinstance(batched_attrs, jnp.ndarray)
    assert batched_attrs.shape == (2, 3) # Batch, Num_Attrs
    assert batched_attrs.dtype == jnp.int32


# Placeholder for dataloader test
def test_get_celeba_dataloader_output(dummy_celeba_root, dummy_attr_file, dummy_image_files, monkeypatch):
    monkeypatch.setattr("celeba_loader.TRAIN_SPLIT_END", 1)
    monkeypatch.setattr("celeba_loader.VALIDATION_SPLIT_END", 2)
    monkeypatch.setattr("celeba_loader.TARGET_IMG_SIZE", (16, 16))
    monkeypatch.setattr("celeba_loader.CENTER_CROP_SIZE", 18)

    # Test train loader
    loader_train = get_celeba_dataloader(dummy_celeba_root, batch_size=1, split="train", augmentation=False, num_workers=0)
    assert len(loader_train.dataset) == 1

    imgs_train, attrs_train = next(iter(loader_train))

    assert imgs_train.shape == (1, 3, 16, 16) # Batch, C, H, W (target_size)
    assert imgs_train.dtype == jnp.float32
    assert attrs_train.shape == (1, 3) # Batch, Num_Attrs from dummy file
    assert attrs_train.dtype == jnp.int32
    assert jnp.array_equal(attrs_train[0], jnp.array([1, -1, 1]))

    # Test valid loader (should have 1 item: 000002.jpg)
    loader_valid = get_celeba_dataloader(dummy_celeba_root, batch_size=1, split="valid", num_workers=0)
    assert len(loader_valid.dataset) == 1
    imgs_valid, attrs_valid = next(iter(loader_valid))
    assert jnp.array_equal(attrs_valid[0], jnp.array([-1, 1, 1]))

    # Test test loader (should have 1 item: 000003.jpg)
    loader_test = get_celeba_dataloader(dummy_celeba_root, batch_size=1, split="test", num_workers=0)
    assert len(loader_test.dataset) == 1
    imgs_test, attrs_test = next(iter(loader_test))
    assert jnp.array_equal(attrs_test[0], jnp.array([1, 1, -1]))

# To run these tests, navigate to the repo root and run:
# python -m pytest
# or if tests/ is a module:
# pytest tests/
# Ensure celeba_loader.py is in the python path (e.g. by being in the root or installed)
# The sys.path modification at the top of this test file is a common way to handle this for local testing.

# Note: The dummy images created are 20x20.
# If TARGET_IMG_SIZE is (128,128) and CENTER_CROP_SIZE is 178 by default,
# transforms will try to crop 20x20 to 178x178, which will error.
# So, for transform and dataloader tests, monkeypatching these constants
# to smaller values (e.g. TARGET_IMG_SIZE=(16,16), CENTER_CROP_SIZE=18) is crucial.
# This has been done in the placeholder tests above.
