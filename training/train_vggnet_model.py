#train_vggnet_model.py
# Example of how train_model could use this:
import tensorflow as tf
from models.vgg_model import VGGNetModel
from data_load.load_dataset import load_emnist_letters
from tensorflow import keras
import matplotlib.pyplot as plt
from preprocessing.input_data_preprocessing import preprocess_and_fix_orientation

learning_rate: float = 0.001

def train_vggnet_model() -> keras.callbacks.History:
    # Initialize model
    model = VGGNetModel(num_classes=26)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        learning_rate=learning_rate
    )

    # Load data
    (ds_train, ds_test), ds_info = load_emnist_letters()

    # Preprocess data
    ds_train = ds_train.map(preprocess_and_fix_orientation).batch(32).prefetch(tf.data.AUTOTUNE)
    ds_test = ds_test.map(preprocess_and_fix_orientation).batch(32).prefetch(tf.data.AUTOTUNE)

    # Train
    history = model.get_model().fit(
        ds_train,
        epochs=50,
        validation_data=ds_test
    )

    # Save
    model.save("models/saved/vggnet_model_preprocessed.keras")

    return history, model