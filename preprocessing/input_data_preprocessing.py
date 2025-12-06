"""
EMNIST Image Preprocessing Module

EMNIST dataset images are stored transposed and rotated compared to
natural handwriting orientation. This module provides functions to
correct the orientation so trained models work with normal handwritten input.

The fix: Transpose + Horizontal flip (or equivalently: rotate 90° + vertical flip)
"""

import tensorflow as tf
import numpy as np


def fix_emnist_orientation(image, label):
    """
    Fix EMNIST image orientation for training.

    EMNIST images are transposed and flipped compared to natural writing.
    This function corrects them to match how people naturally write.

    Args:
        image: Image tensor with shape (28, 28, 1) or (28, 28)
        label: Label tensor (passed through unchanged)

    Returns:
        Tuple of (corrected_image, label)
    """
    # Transpose (swap rows and columns)
    image = tf.transpose(image, perm=[1, 0, 2] if len(image.shape) == 3 else [1, 0])

    # Flip horizontally
    image = tf.image.flip_left_right(image)

    return image, label


def fix_emnist_orientation_image_only(image):
    """
    Fix EMNIST image orientation (image only, no label).

    Use this for inference preprocessing or when working with images directly.

    Args:
        image: Image tensor with shape (28, 28, 1) or (28, 28)

    Returns:
        Corrected image tensor
    """
    # Ensure 3D shape for flip_left_right
    original_shape = tf.shape(image)
    if len(image.shape) == 2:
        image = tf.expand_dims(image, axis=-1)

    # Transpose (swap rows and columns)
    image = tf.transpose(image, perm=[1, 0, 2])

    # Flip horizontally
    image = tf.image.flip_left_right(image)

    return image


def preprocess_and_fix_orientation(image, label):
    """
    Combined preprocessing: fix orientation + normalize.

    This is a drop-in replacement for SimpleModel.preprocess that also
    fixes the EMNIST orientation issue.

    Args:
        image: Image tensor (uint8, 0-255)
        label: Label tensor

    Returns:
        Tuple of (processed_image, adjusted_label)
    """
    # Convert to float and normalize to 0-1
    image = tf.cast(image, tf.float32) / 255.0

    # Ensure shape is (28, 28, 1)
    if len(image.shape) == 2:
        image = tf.expand_dims(image, axis=-1)

    # Fix EMNIST orientation
    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)

    # EMNIST letters labels are 1-26, convert to 0-25 for training
    label = label - 1

    return image, label


def create_preprocessing_pipeline(normalize=True, fix_orientation=True, adjust_labels=True):
    """
    Factory function to create a custom preprocessing function.

    Args:
        normalize: Whether to normalize pixel values to 0-1
        fix_orientation: Whether to fix EMNIST rotation/transpose
        adjust_labels: Whether to convert labels from 1-26 to 0-25

    Returns:
        Preprocessing function compatible with tf.data.Dataset.map()
    """

    def preprocess_fn(image, label):
        # Normalize
        if normalize:
            image = tf.cast(image, tf.float32) / 255.0

        # Ensure 3D shape
        if len(image.shape) == 2:
            image = tf.expand_dims(image, axis=-1)

        # Fix orientation
        if fix_orientation:
            image = tf.transpose(image, perm=[1, 0, 2])
            image = tf.image.flip_left_right(image)

        # Adjust labels (EMNIST letters are 1-26, we need 0-25)
        if adjust_labels:
            label = label - 1

        return image, label

    return preprocess_fn


# NumPy versions for non-TensorFlow use cases
def fix_orientation_numpy(image):
    """
    Fix EMNIST orientation using NumPy.

    Args:
        image: NumPy array with shape (28, 28) or (28, 28, 1)

    Returns:
        Corrected image as NumPy array
    """
    # Remove channel dim if present
    if len(image.shape) == 3:
        image = image.squeeze()

    # Transpose and flip
    image = np.transpose(image)
    image = np.fliplr(image)

    return image


def fix_orientation_batch_numpy(images):
    """
    Fix EMNIST orientation for a batch of images using NumPy.

    Args:
        images: NumPy array with shape (batch, 28, 28) or (batch, 28, 28, 1)

    Returns:
        Corrected images as NumPy array
    """
    # Remove channel dim if present
    if len(images.shape) == 4:
        images = images.squeeze(axis=-1)

    # Transpose each image (swap axes 1 and 2)
    images = np.transpose(images, axes=(0, 2, 1))

    # Flip horizontally (along axis 2)
    images = np.flip(images, axis=2)

    return images


# Visualization helper
def visualize_orientation_fix(dataset, num_samples=5):
    """
    Visualize before/after orientation fix.

    Args:
        dataset: TensorFlow dataset with (image, label) tuples
        num_samples: Number of samples to display
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))

    for i, (image, label) in enumerate(dataset.take(num_samples)):
        image_np = image.numpy()

        # Original (top row)
        axes[0, i].imshow(image_np.squeeze(), cmap='gray')
        axes[0, i].set_title(f'Original\nLabel: {label.numpy()}')
        axes[0, i].axis('off')

        # Fixed (bottom row)
        fixed = fix_orientation_numpy(image_np)
        axes[1, i].imshow(fixed, cmap='gray')
        letter = chr(label.numpy() - 1 + ord('A'))  # Convert 1-26 to A-Z
        axes[1, i].set_title(f'Fixed\nLetter: {letter}')
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel('Before', fontsize=12)
    axes[1, 0].set_ylabel('After', fontsize=12)

    plt.suptitle('EMNIST Orientation Fix', fontsize=14)
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    print("EMNIST Preprocessing Module")
    print("=" * 50)
    print("\nUsage in train_model.py:")
    print("-" * 50)
    print("""
from preprocessing.input_data_preprocessing import preprocess_and_fix_orientation

# Replace SimpleModel.preprocess with preprocess_and_fix_orientation:
ds_train = ds_train.map(preprocess_and_fix_orientation).batch(32).prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.map(preprocess_and_fix_orientation).batch(32).prefetch(tf.data.AUTOTUNE)
    """)

    print("\nOr use the factory function for custom options:")
    print("-" * 50)
    print("""
from preprocessing.input_data_preprocessing import create_preprocessing_pipeline

preprocess_fn = create_preprocessing_pipeline(
    normalize=True,
    fix_orientation=True,
    adjust_labels=True
)

ds_train = ds_train.map(preprocess_fn).batch(32).prefetch(tf.data.AUTOTUNE)
    """)