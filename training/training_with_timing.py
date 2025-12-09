import tensorflow as tf
from models.simple_model import SimpleModel
from data_load.load_dataset import load_emnist_letters
from tensorflow import keras
import matplotlib.pyplot as plt
import time
from preprocessing.input_data_preprocessing import preprocess_and_fix_orientation


def train_model(use_gpu=True) -> keras.callbacks.History:
    # Configure device
    if not use_gpu:
        tf.config.set_visible_devices([], 'GPU')  # Hide GPUs
        print("Training on CPU (GPU disabled)")
    else:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"Training on GPU: {gpus}")
        else:
            print("No GPU found, training on CPU")

    # Initialize model
    model = SimpleModel(num_classes=26)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Load data
    print("Loading data...")
    load_start = time.time()
    (ds_train, ds_test), ds_info = load_emnist_letters()
    load_time = time.time() - load_start
    print(f"Data loading time: {load_time:.2f} seconds")

    # Preprocess data
    ds_train = ds_train.map(preprocess_and_fix_orientation).batch(32).prefetch(tf.data.AUTOTUNE)
    ds_test = ds_test.map(preprocess_and_fix_orientation).batch(32).prefetch(tf.data.AUTOTUNE)

    # Train with timing
    print("Starting training...")
    train_start = time.time()

    history = model.get_model().fit(
        ds_train,
        epochs=50,
        validation_data=ds_test
    )

    train_time = time.time() - train_start

    # Print timing summary
    print("\n" + "=" * 50)
    print("TIMING SUMMARY")
    print("=" * 50)
    print(f"Data loading time: {load_time:.2f} seconds")
    print(f"Training time:     {train_time:.2f} seconds ({train_time / 60:.2f} minutes)")
    print(f"Time per epoch:    {train_time / 50:.2f} seconds")
    print(f"Total time:        {load_time + train_time:.2f} seconds")
    print("=" * 50 + "\n")


    return history, model, train_time


if __name__ == "__main__":
    # Run comparison
    print("\n" + "=" * 60)
    print("RUNNING GPU vs CPU COMPARISON")
    print("=" * 60)

    # Train with GPU
    print("\n>>> Training WITH GPU <<<")
    _, _, gpu_time = train_model(use_gpu=True)

    # Train without GPU
    print("\n>>> Training WITHOUT GPU <<<")
    _, _, cpu_time = train_model(use_gpu=False)

    # Final comparison
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print(f"GPU training time: {gpu_time:.2f} seconds ({gpu_time / 60:.2f} min)")
    print(f"CPU training time: {cpu_time:.2f} seconds ({cpu_time / 60:.2f} min)")
    print(f"Speedup:           {cpu_time / gpu_time:.2f}x faster with GPU")
    print("=" * 60)