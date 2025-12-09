"""
Augment Handwriting Images

This script creates augmented versions of handwriting images to increase
the training dataset size. Augmentations are tailored for handwriting:
- Small rotations
- Translations (shifts)
- Scaling/zoom
- Shearing
- Elastic distortions
- Noise addition
- Erosion/dilation (thickness variation)

Usage:
    python augment_images.py [--multiplier 10] [--source folder] [--dest folder]
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from pathlib import Path
import argparse
import random
from scipy.ndimage import map_coordinates, gaussian_filter, rotate, shift, zoom
from scipy.ndimage import binary_erosion, binary_dilation
import warnings

warnings.filterwarnings('ignore')


def elastic_transform(image: np.ndarray, alpha: float = 15, sigma: float = 3) -> np.ndarray:
    """
    Elastic deformation of images - very effective for handwriting augmentation.

    Args:
        image: Input image as numpy array
        alpha: Intensity of the deformation
        sigma: Smoothness of the deformation
    """
    shape = image.shape

    # Random displacement fields
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha

    # Create mesh grid
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

    # Apply displacement
    indices = (
        np.clip(y + dy, 0, shape[0] - 1).flatten(),
        np.clip(x + dx, 0, shape[1] - 1).flatten()
    )

    return map_coordinates(image, indices, order=1).reshape(shape)


def random_rotation(image: np.ndarray, max_angle: float = 15) -> np.ndarray:
    """Rotate image by a random angle."""
    angle = random.uniform(-max_angle, max_angle)
    return rotate(image, angle, reshape=False, mode='constant', cval=255)


def random_shift(image: np.ndarray, max_shift: float = 3) -> np.ndarray:
    """Shift image randomly in x and y directions."""
    shift_x = random.uniform(-max_shift, max_shift)
    shift_y = random.uniform(-max_shift, max_shift)
    return shift(image, [shift_y, shift_x], mode='constant', cval=255)


def random_zoom(image: np.ndarray, zoom_range: tuple = (0.9, 1.1)) -> np.ndarray:
    """Zoom in/out on the image."""
    zoom_factor = random.uniform(*zoom_range)
    zoomed = zoom(image, zoom_factor, mode='constant', cval=255)

    # Crop or pad to original size
    h, w = image.shape
    zh, zw = zoomed.shape

    if zoom_factor > 1:
        # Crop center
        start_h = (zh - h) // 2
        start_w = (zw - w) // 2
        return zoomed[start_h:start_h + h, start_w:start_w + w]
    else:
        # Pad with white
        result = np.full((h, w), 255, dtype=image.dtype)
        start_h = (h - zh) // 2
        start_w = (w - zw) // 2
        result[start_h:start_h + zh, start_w:start_w + zw] = zoomed
        return result


def random_shear(image: np.ndarray, max_shear: float = 0.2) -> np.ndarray:
    """Apply random shear transformation."""
    shear = random.uniform(-max_shear, max_shear)
    h, w = image.shape

    # Create shear transformation
    result = np.full((h, w), 255, dtype=np.float32)
    for y in range(h):
        for x in range(w):
            new_x = int(x + shear * (y - h / 2))
            if 0 <= new_x < w:
                result[y, new_x] = image[y, x]

    return result


def add_noise(image: np.ndarray, noise_level: float = 10) -> np.ndarray:
    """Add random Gaussian noise."""
    noise = np.random.normal(0, noise_level, image.shape)
    noisy = image + noise
    return np.clip(noisy, 0, 255)


def adjust_thickness(image: np.ndarray, mode: str = 'random') -> np.ndarray:
    """
    Adjust stroke thickness using morphological operations.
    mode: 'erode', 'dilate', or 'random'
    """
    # Convert to binary (assuming dark text on light background)
    binary = image < 128

    if mode == 'random':
        mode = random.choice(['erode', 'dilate', 'none'])

    if mode == 'erode':
        # Make strokes thinner
        transformed = binary_erosion(binary, iterations=1)
    elif mode == 'dilate':
        # Make strokes thicker
        transformed = binary_dilation(binary, iterations=1)
    else:
        return image

    # Convert back to grayscale
    result = np.where(transformed, 0, 255).astype(np.float32)
    return result


def adjust_brightness(image: np.ndarray, factor_range: tuple = (0.8, 1.2)) -> np.ndarray:
    """Adjust image brightness."""
    factor = random.uniform(*factor_range)
    adjusted = image * factor
    return np.clip(adjusted, 0, 255)


def adjust_contrast(image: np.ndarray, factor_range: tuple = (0.8, 1.2)) -> np.ndarray:
    """Adjust image contrast."""
    factor = random.uniform(*factor_range)
    mean = image.mean()
    adjusted = (image - mean) * factor + mean
    return np.clip(adjusted, 0, 255)


def create_augmented_image(image: np.ndarray, augmentation_strength: str = 'medium') -> np.ndarray:
    """
    Apply a random combination of augmentations to an image.

    Args:
        image: Input image as numpy array (grayscale)
        augmentation_strength: 'light', 'medium', or 'heavy'
    """
    # Parameters based on strength
    params = {
        'light': {
            'rotation': 8, 'shift': 2, 'zoom': (0.95, 1.05),
            'shear': 0.1, 'elastic_alpha': 8, 'noise': 5
        },
        'medium': {
            'rotation': 15, 'shift': 3, 'zoom': (0.9, 1.1),
            'shear': 0.2, 'elastic_alpha': 15, 'noise': 10
        },
        'heavy': {
            'rotation': 20, 'shift': 4, 'zoom': (0.85, 1.15),
            'shear': 0.3, 'elastic_alpha': 20, 'noise': 15
        }
    }
    p = params.get(augmentation_strength, params['medium'])

    result = image.astype(np.float32)

    # Randomly apply augmentations
    augmentations = [
        (0.7, lambda img: random_rotation(img, p['rotation'])),
        (0.6, lambda img: random_shift(img, p['shift'])),
        (0.5, lambda img: random_zoom(img, p['zoom'])),
        (0.4, lambda img: random_shear(img, p['shear'])),
        (0.5, lambda img: elastic_transform(img, p['elastic_alpha'])),
        (0.3, lambda img: add_noise(img, p['noise'])),
        (0.3, lambda img: adjust_thickness(img)),
        (0.4, lambda img: adjust_brightness(img)),
        (0.4, lambda img: adjust_contrast(img)),
    ]

    for probability, transform in augmentations:
        if random.random() < probability:
            result = transform(result)

    return np.clip(result, 0, 255).astype(np.uint8)


def augment_dataset(
        source_folder: str,
        dest_folder: str,
        multiplier: int = 10,
        include_original: bool = True,
        strength: str = 'medium'
) -> int:
    """
    Augment all images in source folder and save to destination.

    Args:
        source_folder: Path to source images
        dest_folder: Path to save augmented images
        multiplier: Number of augmented versions per image
        include_original: Whether to copy original images too
        strength: Augmentation strength ('light', 'medium', 'heavy')

    Returns:
        Total number of images created
    """
    source = Path(source_folder)
    dest = Path(dest_folder)

    if not source.exists():
        raise FileNotFoundError(f"Source folder not found: {source_folder}")

    dest.mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
    image_files = [f for f in source.iterdir()
                   if f.is_file() and f.suffix.lower() in image_extensions]

    print(f"Found {len(image_files)} images in source folder")
    print(f"Creating {multiplier} augmented versions per image")
    print(f"Augmentation strength: {strength}")

    total_created = 0

    for img_path in sorted(image_files):
        # Load image
        img = Image.open(img_path).convert('L')
        img_array = np.array(img)

        # Get label from filename (first character)
        label = img_path.stem[0].upper()
        base_name = img_path.stem

        # Save original if requested
        if include_original:
            original_dest = dest / f"{base_name}_orig{img_path.suffix}"
            img.save(original_dest)
            total_created += 1

        # Create augmented versions
        for i in range(multiplier):
            augmented = create_augmented_image(img_array, strength)
            aug_img = Image.fromarray(augmented)

            # Save with augmentation index
            aug_dest = dest / f"{base_name}_aug{i:03d}{img_path.suffix}"
            aug_img.save(aug_dest)
            total_created += 1

        print(f"  Processed: {img_path.name} -> {multiplier + (1 if include_original else 0)} images")

    return total_created


def main():
    parser = argparse.ArgumentParser(description="Augment handwriting images")
    parser.add_argument(
        "--source",
        type=str,
        default="training/training_images_selected/images_to_train",
        help="Source folder with original images"
    )
    parser.add_argument(
        "--dest",
        type=str,
        default="training/training_images_selected/images_to_train_augmented",
        help="Destination folder for augmented images"
    )
    parser.add_argument(
        "--multiplier",
        type=int,
        default=10,
        help="Number of augmented versions per image (default: 10)"
    )
    parser.add_argument(
        "--strength",
        type=str,
        choices=['light', 'medium', 'heavy'],
        default='medium',
        help="Augmentation strength (default: medium)"
    )
    parser.add_argument(
        "--no-original",
        action="store_true",
        help="Don't include original images in output"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"Using random seed: {args.seed}")

    print(f"\nSource: {args.source}")
    print(f"Destination: {args.dest}")
    print()

    total = augment_dataset(
        source_folder=args.source,
        dest_folder=args.dest,
        multiplier=args.multiplier,
        include_original=not args.no_original,
        strength=args.strength
    )

    print(f"\n{'=' * 50}")
    print(f"AUGMENTATION COMPLETE")
    print(f"{'=' * 50}")
    print(f"Total images created: {total}")
    print(f"Output folder: {args.dest}")


if __name__ == "__main__":
    main()