# main.py
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import mixed_precision
from tensorflow import keras
from data_load.load_dataset import load_emnist_letters
from training.train_simple_model import train_model
import matplotlib.pyplot as plt

print(f"TensorFlow version: {tf.__version__}")
print(f"TensorFlow Datasets version: {tfds.__version__}")
#TensorFlow version: 2.10.1 TensorFlow Datasets version: 4.8.3

# 1 set gpu memory growth so tf does not reserve all memory
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print('GPU working')
    except Exception as e:
        print('set memory growth error', e)

# 2 enable mixed precision to speed up training on ampere family
mixed_precision.set_global_policy('mixed_float16')

#history
#history = train_model()

#make predictions on test set
model = keras.models.load_model('models/saved/simple_model.keras')

(ds_train, ds_test), ds_info = load_emnist_letters()
#get only 1 img from ds_test
for image, label in ds_test.take(1):
    print(f"\nImage shape: {image.shape}")
    print(f"Label: {label.numpy()}")
    print(f"Letter: {chr(label.numpy() + ord('A'))}")
    # Transpose (swap width and height)
    image = tf.transpose(image, perm=[1, 0, 2])
    # Flip horizontally
    image = tf.image.flip_left_right(image)
    plt.imshow(image.numpy().squeeze(), cmap='gray')
    plt.title(f"Label: {chr(label.numpy() + ord('A'))}")
    plt.axis('off')
    plt.show()
