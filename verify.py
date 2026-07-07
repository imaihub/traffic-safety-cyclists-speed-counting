import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"cuDNN: {torch.backends.cudnn.version()}")