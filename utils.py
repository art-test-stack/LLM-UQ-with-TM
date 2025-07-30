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

def get_cuda_allocation(verbose=False):
    if DEVICE == torch.device("cuda"):
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r-a 
        if verbose:
            print(f"Total memory {t:,}")
            print(f"Reserved memory {r:,}")
            print(f"Allocated memory {a:,}")
            print(f"Free memory {r-a:,}")
        return r-a
    else:
        print("No cuda device")

from llm.utils import get_model_dir
from pathlib import Path

def get_model_training_fetched_data_csv(model_name: str, training_type: str = "batch") -> Path:
    model_dir = get_model_dir(model_name=model_name, training_type=training_type)
    fetched_data_csv = model_dir.joinpath("fetched_training_data.csv")

    assert fetched_data_csv.exists(), f"Fetched data file {fetched_data_csv} does not exist. Please run the training first."
    
    return fetched_data_csv, model_dir

import os
mplstyle_file = os.getenv("MPLSTYLE")
mplplots_dir = os.getenv("MPLPLOTS")

pt = 1./72.27
golden = (1 + 5 ** 0.5) / 2

width = 337.33545 * pt
height = width / golden