#File: training/select_images_to_train.py

"""
Select Images to Train/Test Split

This script moves images from the source folder to separate training and test folders
based on a configurable split percentage. Images are shuffled randomly before splitting
to ensure a balanced distribution.

Usage:
    python select_images_to_train.py [--train-percent 50] [--copy]
"""

import os
import shutil
import random
import argparse
from pathlib import Path

# Configuration
SOURCE_DIR = Path("input_data/onenote_raw/letter_images")
DEST_BASE_DIR = Path("training/training_images_selected")
TRAIN_DIR = DEST_BASE_DIR / "images_to_train"
TEST_DIR = DEST_BASE_DIR / "images_to_test"

# Supported image extensions
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp'}


def get_image_files(source_dir: Path) -> list[Path]:
    """Get all image files from the source directory."""
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    image_files = [
        f for f in source_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return image_files


def create_destination_folders(train_dir: Path, test_dir: Path) -> None:
    """Create destination folders if they don't exist."""
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created/verified folders:")
    print(f"  - Train: {train_dir}")
    print(f"  - Test:  {test_dir}")


def clear_destination_folders(train_dir: Path, test_dir: Path) -> None:
    """Clear existing files in destination folders."""
    for folder in [train_dir, test_dir]:
        if folder.exists():
            for file in folder.iterdir():
                if file.is_file():
                    file.unlink()
            print(f"Cleared existing files in: {folder}")


def split_images(
        image_files: list[Path],
        train_percent: float,
        train_dir: Path,
        test_dir: Path,
        copy_mode: bool = False
) -> tuple[int, int]:
    """
    Split and move/copy images to train and test folders.

    Args:
        image_files: List of image file paths
        train_percent: Percentage of images for training (0-100)
        train_dir: Destination folder for training images
        test_dir: Destination folder for test images
        copy_mode: If True, copy files instead of moving them

    Returns:
        Tuple of (train_count, test_count)
    """
    # Shuffle images randomly for unbiased split
    shuffled_files = image_files.copy()
    random.shuffle(shuffled_files)

    # Calculate split point
    total_images = len(shuffled_files)
    train_count = int(total_images * (train_percent / 100))

    # Split the files
    train_files = shuffled_files[:train_count]
    test_files = shuffled_files[train_count:]

    # Choose operation (move or copy)
    operation = shutil.copy2 if copy_mode else shutil.move
    operation_name = "Copying" if copy_mode else "Moving"

    print(f"\n{operation_name} images...")

    # Move/copy training images
    for file in train_files:
        dest = train_dir / file.name
        operation(str(file), str(dest))

    # Move/copy test images
    for file in test_files:
        dest = test_dir / file.name
        operation(str(file), str(dest))

    return len(train_files), len(test_files)


def print_summary(total: int, train_count: int, test_count: int, train_percent: float) -> None:
    """Print a summary of the split operation."""
    actual_train_percent = (train_count / total * 100) if total > 0 else 0
    actual_test_percent = (test_count / total * 100) if total > 0 else 0

    print("\n" + "=" * 50)
    print("SPLIT SUMMARY")
    print("=" * 50)
    print(f"Total images found:    {total}")
    print(f"Requested train split: {train_percent:.1f}%")
    print("-" * 50)
    print(f"Training images:       {train_count} ({actual_train_percent:.1f}%)")
    print(f"Test images:           {test_count} ({actual_test_percent:.1f}%)")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Split images into training and test sets"
    )
    parser.add_argument(
        "--train-percent",
        type=float,
        default=50.0,
        help="Percentage of images for training (default: 50)"
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of moving them"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible splits"
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Don't clear existing files in destination folders"
    )

    args = parser.parse_args()

    # Validate train percentage
    if not 0 < args.train_percent < 100:
        raise ValueError("Train percentage must be between 0 and 100 (exclusive)")

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")

    print(f"\nSource directory: {SOURCE_DIR}")
    print(f"Train percentage: {args.train_percent}%")
    print(f"Test percentage:  {100 - args.train_percent}%")
    print(f"Mode: {'Copy' if args.copy else 'Move'}")

    # Get image files
    image_files = get_image_files(SOURCE_DIR)
    print(f"\nFound {len(image_files)} images in source directory")

    if len(image_files) == 0:
        print("No images found. Exiting.")
        return

    # Create destination folders
    create_destination_folders(TRAIN_DIR, TEST_DIR)

    # Clear existing files unless --no-clear is specified
    if not args.no_clear:
        clear_destination_folders(TRAIN_DIR, TEST_DIR)

    # Split and move/copy images
    train_count, test_count = split_images(
        image_files,
        args.train_percent,
        TRAIN_DIR,
        TEST_DIR,
        copy_mode=args.copy
    )

    # Print summary
    print_summary(len(image_files), train_count, test_count, args.train_percent)

    print("\nDone!")


if __name__ == "__main__":
    main()