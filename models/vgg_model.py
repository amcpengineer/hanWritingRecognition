# models/vgg_model.py
from typing import Tuple, Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, initializers


class VGGNetModel:
    """
    VGGNet variant model for EMNIST Letters classification.
    Fixed version with BatchNorm in convolutional layers to prevent
    activation explosion.
    """

    def __init__(
            self,
            num_classes: int = 26,
            input_shape: Tuple[int, int, int] = (28, 28, 1)
    ):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = self._build_model()

    def _build_model(self) -> keras.Model:
        """Build the VGGNet architecture with proper normalization."""

        # Weight initializer
        conv_initializer = initializers.HeNormal()
        dense_initializer = initializers.HeNormal()  # Changed from RandomNormal

        model = keras.Sequential([
            # ========== FEATURE EXTRACTOR ==========

            # Block 1
            layers.Conv2D(
                16, (3, 3), padding='same', use_bias=False,
                kernel_initializer=conv_initializer,
                input_shape=self.input_shape,
                name="block1_conv"
            ),
            layers.BatchNormalization(name="block1_bn"),
            layers.Activation('relu', name="block1_relu"),

            # Block 2
            layers.Conv2D(
                32, (3, 3), padding='same', use_bias=False,
                kernel_initializer=conv_initializer,
                name="block2_conv"
            ),
            layers.BatchNormalization(name="block2_bn"),
            layers.Activation('relu', name="block2_relu"),
            layers.MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool"),

            # Block 3
            layers.Conv2D(
                64, (3, 3), padding='same', use_bias=False,
                kernel_initializer=conv_initializer,
                name="block3_conv1"
            ),
            layers.BatchNormalization(name="block3_bn1"),
            layers.Activation('relu', name="block3_relu1"),
            layers.Conv2D(
                64, (3, 3), padding='same', use_bias=False,
                kernel_initializer=conv_initializer,
                name="block3_conv2"
            ),
            layers.BatchNormalization(name="block3_bn2"),
            layers.Activation('relu', name="block3_relu2"),

            # Block 4
            layers.Conv2D(
                128, (3, 3), padding='same', use_bias=False,
                kernel_initializer=conv_initializer,
                name="block4_conv1"
            ),
            layers.BatchNormalization(name="block4_bn1"),
            layers.Activation('relu', name="block4_relu1"),
            layers.Conv2D(
                128, (3, 3), padding='same', use_bias=False,
                kernel_initializer=conv_initializer,
                name="block4_conv2"
            ),
            layers.BatchNormalization(name="block4_bn2"),
            layers.Activation('relu', name="block4_relu2"),
            layers.MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool"),

            # Block 5
            layers.Conv2D(
                128, (3, 3), padding='same', use_bias=False,
                kernel_initializer=conv_initializer,
                name="block5_conv1"
            ),
            layers.BatchNormalization(name="block5_bn1"),
            layers.Activation('relu', name="block5_relu1"),
            layers.Conv2D(
                128, (3, 3), padding='same', use_bias=False,
                kernel_initializer=conv_initializer,
                name="block5_conv2"
            ),
            layers.BatchNormalization(name="block5_bn2"),
            layers.Activation('relu', name="block5_relu2"),

            # Adaptive Average Pooling equivalent
            layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name="avgpool"),

            # ========== CLASSIFIER ==========

            layers.Flatten(name="flatten"),

            # FC1: 128 * 3 * 3 = 1152 -> 512
            layers.Dense(512, use_bias=False, kernel_initializer=dense_initializer, name="fc1"),
            layers.BatchNormalization(name="bn1"),
            layers.Activation('relu', name="relu1"),
            layers.Dropout(0.5, name="dropout1"),

            # FC2: 512 -> 512
            layers.Dense(512, use_bias=False, kernel_initializer=dense_initializer, name="fc2"),
            layers.BatchNormalization(name="bn2"),
            layers.Activation('relu', name="relu2"),
            layers.Dropout(0.5, name="dropout2"),

            # Output: 512 -> num_classes
            layers.Dense(
                self.num_classes,
                activation='softmax',
                kernel_initializer=dense_initializer,
                name="output"
            )
        ], name="vggnet")

        return model

    def compile(
            self,
            optimizer: str = "adam",
            loss: str = "sparse_categorical_crossentropy",
            metrics: Optional[list] = None,
            learning_rate: float = 0.001
    ) -> None:
        """Compile the model with optimizer, loss, and metrics."""
        if metrics is None:
            metrics = ["accuracy"]

        # Create optimizer with gradient clipping for stability
        if optimizer == "adam":
            opt = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
        elif optimizer == "sgd":
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, clipnorm=1.0)
        else:
            opt = optimizer

        self.model.compile(optimizer=opt, loss=loss, metrics=metrics)

    @staticmethod
    def preprocess(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Preprocess a single image and label."""
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    def get_model(self) -> keras.Model:
        return self.model

    def summary(self) -> None:
        self.model.summary()

    def save(self, filepath: str) -> None:
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'VGGNetModel':
        loaded_model = keras.models.load_model(filepath)
        instance = cls()
        instance.model = loaded_model
        print(f"Model loaded from {filepath}")
        return instance