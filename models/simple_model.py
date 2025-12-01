# models/simple_model.py
from typing import Tuple, Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


class SimpleModel:
    """
    Simple CNN model for EMNIST Letters classification.
    """

    def __init__(
            self,
            num_classes: int = 26,  # EMNIST Letters: A-Z
            input_shape: Tuple[int, int, int] = (28, 28, 1),
            filters_1: int = 32,
            filters_2: int = 64,
            dense_units: int = 64
    ):
        """
        Initialize the model architecture.

        Args:
            num_classes: Number of output classes (26 for EMNIST Letters)
            input_shape: Shape of input images (height, width, channels)
            filters_1: Number of filters in first Conv2D layer
            filters_2: Number of filters in second Conv2D layer
            dense_units: Number of units in dense layer
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.filters_1 = filters_1
        self.filters_2 = filters_2
        self.dense_units = dense_units

        self.model = self._build_model()

    def _build_model(self) -> keras.Model:
        """
        Build the CNN architecture.

        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            # First convolutional block
            layers.Conv2D(
                self.filters_1,
                (3, 3),
                activation="relu",
                input_shape=self.input_shape,
                name="conv2d_1"
            ),
            layers.MaxPooling2D((2, 2), name="maxpool_1"),

            # Second convolutional block
            layers.Conv2D(
                self.filters_2,
                (3, 3),
                activation="relu",
                name="conv2d_2"
            ),
            layers.MaxPooling2D((2, 2), name="maxpool_2"),

            # Dense layers
            layers.Flatten(name="flatten"),
            layers.Dense(self.dense_units, activation="relu", name="dense"),
            layers.Dropout(0.5, name="dropout"),  # Added dropout for regularization
            layers.Dense(self.num_classes, activation="softmax", name="output")
        ], name="simple_cnn")

        return model

    def compile(
            self,
            optimizer: str = "adam",
            loss: str = "sparse_categorical_crossentropy",
            metrics: Optional[list] = None
    ) -> None:
        """
        Compile the model with optimizer, loss, and metrics.

        Args:
            optimizer: Optimizer name or instance
            loss: Loss function name or instance
            metrics: List of metrics to track
        """
        if metrics is None:
            metrics = ["accuracy"]

        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

    @staticmethod
    def preprocess(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Preprocess a single image and label.

        Args:
            image: Input image tensor
            label: Label tensor

        Returns:
            Preprocessed image and label
        """
        # Normalize pixel values to [0, 1]
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    def get_model(self) -> keras.Model:
        """
        Get the underlying Keras model.

        Returns:
            Keras model
        """
        return self.model

    def summary(self) -> None:
        """
        Print model architecture summary.
        """
        self.model.summary()

    def save(self, filepath: str) -> None:
        """
        Save the model to disk.

        Args:
            filepath: Path to save the model
        """
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'SimpleModel':
        """
        Load a saved model from disk.

        Args:
            filepath: Path to the saved model

        Returns:
            SimpleModel instance with loaded weights
        """
        loaded_model = keras.models.load_model(filepath)

        # Create instance and replace model
        instance = cls()
        instance.model = loaded_model

        print(f"Model loaded from {filepath}")
        return instance
