import torch
import os
import psutil

def get_size(bytes, suffix="B"):
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

print(f"System RAM: {get_size(psutil.virtual_memory().total)}")
print(f"Available RAM: {get_size(psutil.virtual_memory().available)}")

try:
    import bitsandbytes as bnb
    print(f"bitsandbytes version: {bnb.__version__}")
    # Basic check for 4-bit config or similar
    from transformers import BitsAndBytesConfig
    config = BitsAndBytesConfig(load_in_4bit=True)
    print("bitsandbytes/transformer config load: Ok")
except ImportError as e:
    print(f"bitsandbytes check failed: {e}")
except Exception as e:
    print(f"bitsandbytes error: {e}")
