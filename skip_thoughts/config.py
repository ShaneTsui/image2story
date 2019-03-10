"""
Configuration file.
"""
import torch

# Check if your system supports CUDA
use_cuda = torch.cuda.is_available()

# Setup GPU optimization if CUDA is supported
if use_cuda:
    computing_device = torch.device("cuda")
    extras = {"num_workers": 1, "pin_memory": True}
    print("CUDA is supported")
else: # Otherwise, train on the CPU
    computing_device = torch.device("cpu")
    extras = False
    print("CUDA NOT supported")
VOCAB_SIZE = 20000
USE_CUDA = True
DEVICES = [1]
CUDA_DEVICE = computing_device
VERSION = 1
MAXLEN = 30