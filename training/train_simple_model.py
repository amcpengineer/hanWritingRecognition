# Example of how train_model could use this:
import tensorflow as tf
from models.simple_model import SimpleModel
from data_load.load_dataset import load_emnist_letters
from tensorflow import keras
import matplotlib.pyplot as plt

def train_model() -> keras.callbacks.History:
    # Initialize model
    model = SimpleModel(num_classes=26)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Load data
    (ds_train, ds_test), ds_info = load_emnist_letters()

    # Preprocess data
    ds_train = ds_train.map(SimpleModel.preprocess).batch(32).prefetch(tf.data.AUTOTUNE)
    ds_test = ds_test.map(SimpleModel.preprocess).batch(32).prefetch(tf.data.AUTOTUNE)

    # Train
    history = model.get_model().fit(
        ds_train,
        epochs=50,
        validation_data=ds_test
    )

    # Plot learning curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Learning Curves - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['sparse_categorical_accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_sparse_categorical_accuracy'], label='Validation Accuracy')
    plt.title('Learning Curves - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.show()

    # Save
    model.save("models/saved/simple_model.keras")

    return history, model