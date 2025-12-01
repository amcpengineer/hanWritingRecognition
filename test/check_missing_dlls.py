import os
import ctypes

print("=== Checking for Missing DLLs ===\n")

cuda_bin = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin'

# Add CUDA bin to DLL search
os.add_dll_directory(cuda_bin)

# List of DLLs TensorFlow needs
required_dlls = [
    'cudart64_110.dll',
    'cublas64_11.dll',
    'cublasLt64_11.dll',
    'cufft64_10.dll',
    'curand64_10.dll',
    'cusolver64_11.dll',
    'cusparse64_11.dll',
    'cudnn64_8.dll',
    'zlibwapi.dll'  # Often missing!
]

print("Checking for required DLLs in CUDA bin:")
for dll in required_dlls:
    dll_path = os.path.join(cuda_bin, dll)
    if os.path.exists(dll_path):
        # Try to load it
        try:
            ctypes.CDLL(dll_path)
            print(f"  ✓ {dll} - Found and loadable")
        except Exception as e:
            print(f"  ⚠ {dll} - Found but can't load: {e}")
    else:
        print(f"  ✗ {dll} - MISSING")

# Now try importing TensorFlow with verbose logging
print("\n=== Importing TensorFlow with Debug Info ===")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # Show all logs

import tensorflow as tf
print(f"\nTensorFlow version: {tf.__version__}")
print(f"GPUs detected: {len(tf.config.list_physical_devices('GPU'))}")