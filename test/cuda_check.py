import os
import sys

print("=== CUDA Installation Check ===\n")

# Check CUDA_PATH environment variable
cuda_path = os.environ.get('CUDA_PATH')
print(f"CUDA_PATH environment variable: {cuda_path}")

# Check if CUDA 11.8 is installed
cuda_11_8_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
print(f"\nCUDA 11.8 directory exists: {os.path.exists(cuda_11_8_path)}")

if os.path.exists(cuda_11_8_path):
    bin_path = os.path.join(cuda_11_8_path, 'bin')
    print(f"CUDA bin directory exists: {os.path.exists(bin_path)}")

    # Check for important DLLs
    important_dlls = ['cudart64_110.dll', 'cublas64_11.dll', 'cublasLt64_11.dll', 'cusparse64_11.dll']
    print("\nChecking for CUDA DLLs in bin:")
    for dll in important_dlls:
        dll_path = os.path.join(bin_path, dll)
        exists = os.path.exists(dll_path)
        print(f"  {dll}: {'✓ Found' if exists else '✗ Missing'}")

    # Check for cuDNN
    cudnn_dll = os.path.join(bin_path, 'cudnn64_8.dll')
    print(f"\ncuDNN library (cudnn64_8.dll): {'✓ Found' if os.path.exists(cudnn_dll) else '✗ Missing'}")

# Check PATH
print("\n=== Checking System PATH ===")
path_env = os.environ.get('PATH', '')
cuda_in_path = any('CUDA' in p and 'v11.8' in p for p in path_env.split(';'))
print(f"CUDA 11.8 in PATH: {'✓ Yes' if cuda_in_path else '✗ No'}")

if cuda_in_path:
    cuda_paths = [p for p in path_env.split(';') if 'CUDA' in p and 'v11.8' in p]
    print("CUDA paths found:")
    for p in cuda_paths:
        print(f"  {p}")

print("\n=== Now testing TensorFlow ===")
import tensorflow as tf

print(f"TensorFlow version: {tf.__version__}")
print(f"GPUs detected: {len(tf.config.list_physical_devices('GPU'))}")