import torch

CUDA_AVAILABLE = torch.cuda.is_available()
MPS_AVAILABLE = torch.backends.mps.is_available()
if MPS_AVAILABLE:
    torch.mps.empty_cache()
    torch.mps.set_per_process_memory_fraction(0.)
DEVICE_NAME = "cuda" if CUDA_AVAILABLE else "mps" if MPS_AVAILABLE else "cpu"
DEVICE = torch.device(DEVICE_NAME)
# DEVICE = torch.device("cpu")

def get_device() -> torch.device:
    '''Get the device'''
    return DEVICE

def get_cuda_allocation():
    if DEVICE == torch.device("cuda"):
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r-a 
        print("Total memory", t)
        print("Reserved memory", r)
        print("Allocated memory", a)
        print("Free memory")
    else:
        print("No cuda device")