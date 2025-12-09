# main.py
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import mixed_precision
from training.train_vggnet_model_combined import train_model_combined
import time
from tensorflow import keras
from data_load.load_dataset import load_emnist_letters
from training.train_simple_model2 import train_model2
import matplotlib.pyplot as plt
from preprocessing.preprocessing_img import preprocess_image
import numpy as np

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

print("Starting training...")
train_start = time.time()

#history
history, model = train_model_combined(
        custom_train_folder="training/training_images_selected/images_to_train_augmented",  # ← Fix this
        epochs=50,
        batch_size=64
    )

train_time = time.time() - train_start

# Print timing summary
print("\n" + "=" * 50)
print("TIMING SUMMARY")
print("=" * 50)
#print(f"Data loading time: {load_time:.2f} seconds")
print(f"Training time:     {train_time:.2f} seconds ({train_time / 60:.2f} minutes)")
print(f"Time per epoch:    {train_time / 50:.2f} seconds")
#print(f"Total time:        {load_time + train_time:.2f} seconds")
print("=" * 50 + "\n")
