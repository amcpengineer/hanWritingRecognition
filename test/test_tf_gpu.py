import tensorflow as tf
print("TF version:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
try:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        x = tf.random.uniform((1500,1500))
        y = tf.matmul(x, x)
        print("Simple GPU matmul result shape:", y.shape)
    else:
        print("No GPU detected by TensorFlow")
except Exception as e:
    print("Error during GPU test:", e)
