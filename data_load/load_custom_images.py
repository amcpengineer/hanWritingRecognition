"""
Load Custom Training Images

Loads images from a folder structure where filenames contain the label.
Expected filename format: {label}_{id}.png (e.g., "A_001.png", "B_002.png")

Images are preprocessed to match EMNIST format:
- Resized to 28x28
- Converted to grayscale
- Normalized to [0, 1]
- Labels converted to 0-25 (A=0, B=1, ..., Z=25)
"""

import tensorflow as tf
from pathlib import Path
import numpy as np
from PIL import Image


def extract_label_from_filename(filename: str) -> int:
    """
    Extract label from filename.
    Expects format: {letter}_{id}.png or {letter}.png
    Returns label as int (A=0, B=1, ..., Z=25)
    """
    # Get the base name without extension
    name = Path(filename).stem
    # Get the first character (the letter)
    letter = name[0].upper()
    # Convert to label (A=0, B=1, etc.)
    label = ord(letter) - ord('A')
    return label


def load_and_preprocess_image(image_path: str) -> np.ndarray:
    """
    Load an image and preprocess it to match EMNIST format.
    - Resize to 28x28
    - Convert to grayscale
    - Normalize to [0, 1]
    """
    # Load image
    img = Image.open(image_path)

    # Convert to grayscale if needed
    if img.mode != 'L':
        img = img.convert('L')

    # Resize to 28x28
    img = img.resize((28, 28), Image.Resampling.LANCZOS)

    # Convert to numpy array and normalize
    img_array = np.array(img, dtype=np.float32) / 255.0

    return img_array


def load_custom_images_dataset(
        folder_path: str,
        image_extensions: tuple = ('.png', '.jpg', '.jpeg', '.bmp')
) -> tf.data.Dataset:
    """
    Load custom images from a folder and create a TensorFlow dataset.

    Args:
        folder_path: Path to folder containing images
        image_extensions: Tuple of valid image extensions

    Returns:
        tf.data.Dataset with (image, label) pairs
    """
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    images = []
    labels = []

    # Get all image files
    image_files = [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    print(f"Found {len(image_files)} images in {folder_path}")

    for img_path in sorted(image_files):
        try:
            # Load and preprocess image
            img_array = load_and_preprocess_image(str(img_path))

            # Extract label from filename
            label = extract_label_from_filename(img_path.name)

            # Validate label
            if 0 <= label <= 25:
                images.append(img_array)
                labels.append(label)
            else:
                print(f"Warning: Invalid label for {img_path.name}, skipping")

        except Exception as e:
            print(f"Warning: Could not load {img_path.name}: {e}")

    if len(images) == 0:
        raise ValueError(f"No valid images found in {folder_path}")

    # Convert to numpy arrays
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    print(f"Loaded {len(images)} images successfully")
    print(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    return dataset


def load_custom_train_test_datasets(
        train_folder: str = "training/training_images_selected/images_to_train",
        test_folder: str = "training/training_images_selected/images_to_test"
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Load both training and test datasets from custom image folders.

    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    ds_train = load_custom_images_dataset(train_folder)
    ds_test = load_custom_images_dataset(test_folder)

    return ds_train, ds_test


# For testing
if __name__ == "__main__":
    # Test loading
    train_folder = "training/training_images_selected/images_to_train"

    if Path(train_folder).exists():
        ds = load_custom_images_dataset(train_folder)

        # Print some info
        for image, label in ds.take(3):
            print(f"Image shape: {image.shape}, Label: {label.numpy()} ({chr(label.numpy() + ord('A'))})")
    else:
        print(f"Test folder not found: {train_folder}")