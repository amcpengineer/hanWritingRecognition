"""
Train Model with Combined Dataset - Optimized
"""

import tensorflow as tf
from models.vgg_model import VGGNetModel
from data_load.load_dataset import load_emnist_letters
from data_load.load_custom_images import load_custom_images_dataset
from tensorflow import keras
from preprocessing.input_data_preprocessing import preprocess_and_fix_orientation

# Enable mixed precision for RTX 3050 (significant speedup)
tf.keras.mixed_precision.set_global_policy('mixed_float16')


class NaNDetector(keras.callbacks.Callback):
    """Stop training immediately if loss becomes NaN."""
    def on_batch_end(self, batch, logs=None):
        loss = logs.get('loss')
        if loss is not None and (tf.math.is_nan(loss) or tf.math.is_inf(loss)):
            print(f"\n⚠️ NaN/Inf detected at batch {batch}! Stopping training.")
            self.model.stop_training = True


def preprocess_custom_image(image, label):
    """Preprocess custom images - NO orientation fix needed."""
    # Ensure 3D shape: (28, 28) -> (28, 28, 1)
    image = tf.reshape(image, (28, 28, 1))
    # Cast label to int64 to match EMNIST
    label = tf.cast(label, tf.int64)
    return image, label


def train_model_combined(
    custom_train_folder: str = "training/training_images_selected/images_to_train_augmented",
    epochs: int = 50,
    batch_size: int = 64  # Increased from 32 for better GPU utilization
) -> tuple[keras.callbacks.History, VGGNetModel]:
    """Train model with EMNIST + custom images combined."""

    # Initialize model
    model = VGGNetModel(num_classes=26)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        learning_rate=0.001
    )

    # Load EMNIST data
    print("Loading EMNIST dataset...")
    (ds_emnist_train, ds_emnist_test), ds_info = load_emnist_letters()

    # Preprocess EMNIST and CACHE (big speedup!)
    ds_emnist_train_processed = (
        ds_emnist_train
        .map(preprocess_and_fix_orientation, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()  # Cache after preprocessing
    )
    ds_emnist_test_processed = (
        ds_emnist_test
        .map(preprocess_and_fix_orientation, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
    )

    # Load custom training images
    print(f"\nLoading custom training images from: {custom_train_folder}")
    try:
        ds_custom_train = load_custom_images_dataset(custom_train_folder)
        ds_custom_train_processed = (
            ds_custom_train
            .map(preprocess_custom_image, num_parallel_calls=tf.data.AUTOTUNE)
            .cache()
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

    # Callbacks for stable & efficient training
    callbacks = [
        NaNDetector(),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'models/saved/vggnet_combined_best.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    # Train
    print(f"\nStarting training for {epochs} epochs...")
    history = model.get_model().fit(
        ds_train_final,
        epochs=epochs,
        validation_data=ds_test_final,
        callbacks=callbacks
    )

    # Save final model
    save_path = "models/saved/vggnet_model_combined.keras"
    model.save(save_path)
    print(f"\nModel saved to: {save_path}")

    return history, model


if __name__ == "__main__":
    history, model = train_model_combined(epochs=50, batch_size=64)

    print("\nFinal Results:")
    print(f"  Training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"  Validation accuracy: {history.history['val_accuracy'][-1]:.4f}")