#File: preprocessing/preprocessing_img.py
"""
Image Preprocessing Module for EMNIST Model Predictions

This module provides utilities to preprocess images before feeding them
to a trained EMNIST handwriting recognition model.
"""

import numpy as np
from PIL import Image


class ImagePreprocessor:
    """
    Preprocesses images for EMNIST model predictions.

    Handles loading, resizing, grayscale conversion, normalization,
    and reshaping to match model input requirements.
    """

    def __init__(self, target_size=(28, 28), normalize=True, invert=True):
        """
        Initialize the preprocessor.

        Args:
            target_size: Tuple (height, width) for resizing. Default (28, 28).
            normalize: Whether to normalize pixel values to 0-1. Default True.
            invert: Whether to invert colors (for white-on-black). Default True.
        """
        self.target_size = target_size
        self.normalize = normalize
        self.invert = invert

    def load_image(self, image_path):
        """
        Load an image from file path.

        Args:
            image_path: Path to the image file.

        Returns:
            PIL Image object in grayscale.
        """
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        return image

    def preprocess(self, image):
        """
        Preprocess a single image for model prediction.

        Args:
            image: Can be a file path (str), PIL Image, or numpy array.

        Returns:
            Numpy array with shape (1, 28, 28, 1) ready for model.predict()
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = self.load_image(image)

        # Convert PIL Image to numpy array
        if isinstance(image, Image.Image):
            image = image.convert('L')  # Ensure grayscale
            image = image.resize(self.target_size, Image.Resampling.LANCZOS)
            image = np.array(image, dtype=np.float32)

        # Handle numpy array input
        if isinstance(image, np.ndarray):
            # Resize if needed (for raw numpy arrays)
            if image.shape[:2] != self.target_size:
                pil_img = Image.fromarray(image.astype(np.uint8))
                pil_img = pil_img.convert('L')
                pil_img = pil_img.resize(self.target_size, Image.Resampling.LANCZOS)
                image = np.array(pil_img, dtype=np.float32)

        # Invert colors if needed (EMNIST expects white on black)
        if self.invert:
            image = 255.0 - image

        # Normalize to 0-1 range
        if self.normalize:
            image = image / 255.0

        # Reshape to (1, 28, 28, 1) - batch size 1, single channel
        image = image.reshape(1, self.target_size[0], self.target_size[1], 1)

        return image

    def preprocess_batch(self, images):
        """
        Preprocess multiple images for batch prediction.

        Args:
            images: List of file paths, PIL Images, or numpy arrays.

        Returns:
            Numpy array with shape (batch_size, 28, 28, 1)
        """
        processed = [self.preprocess(img) for img in images]
        return np.vstack(processed)


# Convenience function for quick single-image preprocessing
def preprocess_image(image, target_size=(28, 28), normalize=True, invert=True):
    """
    Quick preprocessing function for a single image.

    Args:
        image: File path (str), PIL Image, or numpy array.
        target_size: Tuple (height, width) for resizing. Default (28, 28).
        normalize: Whether to normalize pixel values to 0-1. Default True.
        invert: Whether to invert colors (for white-on-black). Default True.

    Returns:
        Numpy array with shape (1, 28, 28, 1) ready for model.predict()
    """
    preprocessor = ImagePreprocessor(
        target_size=target_size,
        normalize=normalize,
        invert=invert
    )
    return preprocessor.preprocess(image)


# Example usage
if __name__ == "__main__":
    # Example with class
    preprocessor = ImagePreprocessor()

    # From file path
    # processed_img = preprocessor.preprocess("path/to/image.png")

    # From numpy array (e.g., 28x28 handwritten digit)
    sample_image = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
    processed_img = preprocessor.preprocess(sample_image)

    print(f"Output shape: {processed_img.shape}")  # (1, 28, 28, 1)
    print(f"Value range: [{processed_img.min():.2f}, {processed_img.max():.2f}]")