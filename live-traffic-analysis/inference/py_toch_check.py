import torch

def check_hardware_acceleration():
  if torch.cuda.is_available():
    print("PyTorch is hardware accelerated")
  else:
    print("PyTorch is not hardware accelerated")

check_hardware_acceleration()
