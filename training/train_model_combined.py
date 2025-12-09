"""
Train Model with Combined Dataset

This script trains a model using:
1. EMNIST dataset (with orientation fix preprocessing) for train + test
2. Custom images (without orientation fix) added to training only

The custom test images (images_to_test) are reserved for inference testing,
not used during training validation.
"""

import tensorflow as tf
from models.simple_model import SimpleModel
from data_load.load_dataset import load_emnist_letters
from data_load.load_custom_images import load_custom_images_dataset
from tensorflow import keras
from preprocessing.input_data_preprocessing import preprocess_and_fix_orientation


def preprocess_custom_image(image, label):
    """
    Preprocess custom images - NO orientation fix needed.
    Just ensures correct shape for the model.
    Images already come as (28, 28, 1) from load_custom_images.
    """
    # Ensure shape is (28, 28, 1) to match EMNIST
    image = tf.ensure_shape(image, (28, 28, 1))
    return image, label


def train_model_combined(
    custom_train_folder: str = "training/training_images_selected/images_to_train_augmented",
    epochs: int = 50,
    batch_size: int = 32
) -> tuple[keras.callbacks.History, SimpleModel]:
    """
    Train model with EMNIST + custom images combined.

    Custom images are added to training set only.
    Validation uses EMNIST test set only.

    Args:
        custom_train_folder: Path to custom training images
        epochs: Number of training epochs
        batch_size: Batch size for training

    Returns:
        Tuple of (training history, trained model)
    """
    # Initialize model
    model = SimpleModel(num_classes=26)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Load EMNIST data
    print("Loading EMNIST dataset...")
    (ds_emnist_train, ds_emnist_test), ds_info = load_emnist_letters()

    # Preprocess EMNIST (needs orientation fix)
    ds_emnist_train_processed = ds_emnist_train.map(
        preprocess_and_fix_orientation,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds_emnist_test_processed = ds_emnist_test.map(
        preprocess_and_fix_orientation,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Load custom training images
    print(f"\nLoading custom training images from: {custom_train_folder}")
    try:
        ds_custom_train = load_custom_images_dataset(custom_train_folder)
        ds_custom_train_processed = ds_custom_train.map(
            preprocess_custom_image,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        has_custom_train = True
        custom_train_size = ds_custom_train.cardinality().numpy()
        print(f"Custom training images: {custom_train_size}")
    except (FileNotFoundError, ValueError) as e:
        print(f"No custom training images found: {e}")
        has_custom_train = False

    # Combine training datasets
    print("\nCombining datasets...")
    if has_custom_train:
        ds_train_combined = ds_emnist_train_processed.concatenate(ds_custom_train_processed)
        print("Training: EMNIST + Custom images")
    else:
        ds_train_combined = ds_emnist_train_processed
        print("Training: EMNIST only")

    print("Validation: EMNIST test set only")

    # Shuffle, batch, and prefetch
    ds_train_final = (
        ds_train_combined
        .shuffle(buffer_size=10000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    ds_test_final = (
        ds_emnist_test_processed
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Train
    print(f"\nStarting training for {epochs} epochs...")
    history = model.get_model().fit(
        ds_train_final,
        epochs=epochs,
        validation_data=ds_test_final
    )

    # Save
    save_path = "models/saved/simple_model_combined2.keras"
    model.save(save_path)
    print(f"\nModel saved to: {save_path}")

    return history, model


if __name__ == "__main__":
    # Train with combined dataset
    history, model = train_model_combined(
        epochs=50,
        batch_size=32
    )

    # Print final metrics
    print("\nFinal Results:")
    print(f"  Training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"  Validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
