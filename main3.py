# main.py
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import mixed_precision
from tensorflow import keras
from data_load.load_dataset import load_emnist_letters
from training.training_with_timing import train_model
import matplotlib.pyplot as plt
from preprocessing.preprocessing_img import preprocess_image
import numpy as np

print(f"TensorFlow version: {tf.__version__}")
print(f"TensorFlow Datasets version: {tfds.__version__}")
#TensorFlow version: 2.10.1 TensorFlow Datasets version: 4.8.3

# Run comparison
print("\n" + "=" * 60)
print("RUNNING GPU vs CPU COMPARISON")
print("=" * 60)

# Train without GPU
print("\n>>> Training WITHOUT GPU <<<")
_, _, gpu_time = train_model(use_gpu=True)
