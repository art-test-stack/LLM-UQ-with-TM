from tm_data.features import get_tensor_stats, compute_beta_dist_params, get_grad_spectrums
import torch
from torch import nn

from typing import List, Union
from dataclasses import dataclass


@dataclass
class InputData:
    grad_mean: float
    grad_median: float
    grad_std: float
    grad_max: float
    grad_min: float
    grad_alpha: float
    grad_beta: float
    spectrum_mean: float
    spectrum_median: float
    spectrum_std: float
    spectrum_max: float
    spectrum_min: float
    spectrum_alpha: float
    spectrum_beta: float
    grad_ll_mean: float
    grad_ll_median: float
    grad_ll_std: float
    grad_ll_max: float
    grad_ll_min: float
    grad_ll_alpha: float
    grad_ll_beta: float
    spectrum_ll_mean: float
    spectrum_ll_median: float
    spectrum_ll_std: float
    spectrum_ll_max: float
    spectrum_ll_min: float
    spectrum_ll_alpha: float
    spectrum_ll_beta: float
    epoch: int
    batch: int


def get_all_spectrums(spectra):
    all_spectras = torch.cat([spectrum.view(-1) for spectrum in spectra])
    return all_spectras

def get_last_layers_spectrums(spectra: List[torch.Tensor], nb_last_layers: int = 5) -> List[torch.Tensor]:
    last_layers_spectrums = [ spectrum for spectrum in spectra[-nb_last_layers:]]
    return last_layers_spectrums

def get_last_layers_grads(grads: List[torch.Tensor], nb_last_layers: int = 5) -> List[torch.Tensor]:
    last_layers_grads = [ grad for grad in grads[-nb_last_layers:] ]
    return last_layers_grads

def create_csv(path: str) -> None:
    with open(path, 'w') as f:
        f.write("mean,median,std,max,min\n")
    
# def update_csv(new_input: InputData,path: str) -> None:
#     with open(path, 'a') as f:
#         f.write(f"{mean},{median},{std},{max},{min}\n")
    
def store_data(model: nn.Module, device: torch.device) -> None:
    try: 
        grads = model.get_grads()
    except:
        grads = [p.grad for p in model.parameters() if p.grad is not None]

    spectrums = get_grad_spectrums(grads, model)

