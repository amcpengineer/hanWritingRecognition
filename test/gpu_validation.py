import os

# CRITICAL: Add CUDA bin to DLL search path BEFORE importing TensorFlow
cuda_bin = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin'
if os.path.exists(cuda_bin):
    os.add_dll_directory(cuda_bin)
    print(f"Added CUDA bin to DLL directory: {cuda_bin}")

import tensorflow as tf

print(f"\nTensorFlow version: {tf.__version__}")
print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")

if tf.config.list_physical_devices('GPU'):
    print("\n✓ GPU is available!")
    # Enable memory growth
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Test GPU computation
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
        c = tf.matmul(a, b)
        print(f"Test computation on GPU successful!")
        print(c)
else:
    print("\n✗ No GPU found")