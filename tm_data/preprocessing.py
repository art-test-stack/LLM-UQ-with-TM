from tm_data.features import get_tensor_stats, compute_beta_dist_params, get_grad_spectrums

import pandas as pd
import numpy as np
import torch
from torch import nn

from typing import List, Union
from dataclasses import dataclass
from pathlib import Path

@dataclass
class GradStats:
    grad_mean: float | None = None
    grad_median: float | None = None
    grad_std: float | None = None
    grad_max: float | None = None
    grad_min: float | None = None
    # grad_alpha: float | None = None
    # grad_beta: float | None = None
    spectrum_mean: float | None = None
    spectrum_median: float | None = None
    spectrum_std: float | None = None
    spectrum_max: float | None = None
    spectrum_min: float | None = None
    spectrum_alpha: float | None = None
    spectrum_beta: float | None = None
    grad_ll_mean: float | None = None
    grad_ll_median: float | None = None
    grad_ll_std: float | None = None
    grad_ll_max: float | None = None
    grad_ll_min: float | None = None
    # grad_ll_alpha: float | None = None
    # grad_ll_beta: float | None = None
    spectrum_ll_mean: float | None = None
    spectrum_ll_median: float | None = None
    spectrum_ll_std: float | None = None
    spectrum_ll_max: float | None = None
    spectrum_ll_min: float | None = None
    spectrum_ll_alpha: float | None = None
    spectrum_ll_beta: float | None = None

@dataclass
class InputData(GradStats):
    test_loss: float | None = None
    epoch: int | None = None
    batch_size: int | None = None

class InputCSV:
    def __init__(self, model: nn.Module, path: Union[Path, str]):
        if type(path) == str:
            self.path = Path(path)
        if not self.path.suffix == '.csv':
            self.path = self.path.with_suffix('.csv')
        if not self.path.exists():
            create_csv(self.path)
        self.current = InputData()
        self.current_grads = []
        self.model = model
        self.clean_grads()
        
    def __call__(self, test_loss: float) -> None:
        assert self.current.batch_size, "You must update the hyperparameters before updating the model. Call 'update_hyperparameters' first."
        assert self.current.epoch or self.current.epoch == 0, "You must update the hyperparameters before updating the model. Call 'update_hyperparameters' first."

        self.current.test_loss = test_loss
        # add a way to store loss variations
        self.compute_model_stats()
        self.update_csv()
        self.clean_grads()
    
    def update_hyperparameters(self, epoch: int, batch_size: int) -> None:
        self.current.epoch = epoch
        self.current.batch_size = batch_size

    def update_model(self) -> None:
        # PB 1: can't do torch(grads) or np.array(grads) because of the different shapes for each layer
        # PB 2: store grads of all epochs maybe very heavy computationally
        # Temporary solution: take the stats at each batch iteration at store the mean of each stats at the end of the epoch
        try: 
            grads = self.model.get_grads()
        except:
            if None in grads:
                return
            else:
                grads = [p.grad for p in self.model.parameters()]

        self.compute_grad_stats(grads)

    def compute_grad_stats(self, grads) -> None:
        all_grads = get_concat_tensor(grads)
        last_layers_grads = get_last_layers_grads(grads)
        all_ll_grads = get_concat_tensor(last_layers_grads)

        spectrums = get_grad_spectrums(grads)
        all_spectras = get_concat_tensor(spectrums)
        last_layers_spectrums = get_last_layers_spectrums(spectrums)
        all_ll_spectrums = get_concat_tensor(last_layers_spectrums)

        grad_mean, grad_median, grad_std, grad_max, grad_min = get_tensor_stats(all_grads)
        grad_ll_mean, grad_ll_median, grad_ll_std, grad_ll_max, grad_ll_min = get_tensor_stats(all_ll_grads)
        spectrum_mean, spectrum_median, spectrum_std, spectrum_max, spectrum_min = get_tensor_stats(all_spectras)
        spectrum_ll_mean, spectrum_ll_median, spectrum_ll_std, spectrum_ll_max, spectrum_ll_min = get_tensor_stats(all_ll_spectrums)

        spectrum_alpha, spectrum_beta = compute_beta_dist_params(spectrum_mean, spectrum_std)
        spectrum_ll_alpha, spectrum_ll_beta = compute_beta_dist_params(spectrum_ll_mean, spectrum_ll_std)

        current_grad = GradStats(
            grad_mean, grad_median, grad_std, grad_max, grad_min,
            spectrum_mean, spectrum_median, spectrum_std, spectrum_max, spectrum_min, spectrum_alpha, spectrum_beta,
            grad_ll_mean, grad_ll_median, grad_ll_std, grad_ll_max, grad_ll_min,
            spectrum_ll_mean, spectrum_ll_median, spectrum_ll_std, spectrum_ll_max, spectrum_ll_min, spectrum_ll_alpha, spectrum_ll_beta
        )
        self.current_grads.append(current_grad)

    def compute_model_stats(self) -> None:
        grad_stats = GradStats()
        for fn in grad_stats.__dataclass_fields__.keys():
            fn_values = [getattr(grad, fn) for grad in self.current_grads]
            fn_mean = np.mean(fn_values)
            grad_stats.__setattr__(fn, fn_mean)
        self.current = InputData(
            **{fn: getattr(grad_stats, fn) for fn in grad_stats.__dataclass_fields__.keys()},
            test_loss=self.current.test_loss, epoch=self.current.epoch, batch_size=self.current.batch_size
        )

    def update_csv(self) -> None:
        for fn in InputData.__dataclass_fields__.keys():
            assert hasattr(self.current, fn), f"Attribute {fn} is missing from the current data."

        fields = InputData.__dataclass_fields__.keys()
        if not self.path.exists():
            create_csv(self.path)
        df = pd.DataFrame([{fn: getattr(self.current, fn) for fn in fields}])
        df.to_csv(self.path, mode='a', header=False, index=False)
    
    def clean_grads(self) -> None:
        self.current_grads = []
        
def get_concat_tensor(tensor: torch.Tensor) -> torch.Tensor:
    all_tensors = torch.cat([t.view(-1) for t in tensor])
    return all_tensors

def get_last_layers_spectrums(spectra: List[torch.Tensor], nb_last_layers: int = 5) -> List[torch.Tensor]:
    last_layers_spectrums = [ spectrum for spectrum in spectra[-nb_last_layers:]]
    return last_layers_spectrums

def get_last_layers_grads(grads: List[torch.Tensor], nb_last_layers: int = 5) -> List[torch.Tensor]:
    last_layers_grads = [ grad for grad in grads[-nb_last_layers:] ]
    return last_layers_grads

def create_csv(path: str) -> None:
    with open(path, 'w') as f:
        f.write(f'{",".join(InputData.__dataclass_fields__.keys())}\n')
    
def update_csv(new_input: InputData, path: str) -> None:
    fields = InputData.__dataclass_fields__.keys()
    path = Path(f'{path}.csv')
    if not path.exists():
        create_csv(path)
    df = pd.DataFrame([{fn: getattr(new_input, fn) for fn in fields}])
    df.to_csv(path, mode='a', header=False, index=False)

def store_data(
        model: nn.Module, 
        epoch:int, 
        batch_size: int, 
        device: torch.device
    ) -> None:
    try: 
        grads = model.get_grads()
    except:
        grads = [p.grad for p in model.parameters() if p.grad]
        if None in grads:
            return
    
    all_grads = get_concat_tensor(grads)
    last_layers_grads = get_last_layers_grads(grads)
    all_ll_grads = get_concat_tensor(last_layers_grads)

    spectrums = get_grad_spectrums(grads)
    all_spectras = get_concat_tensor(spectrums)
    last_layers_spectrums = get_last_layers_spectrums(spectrums)
    all_ll_spectrums = get_concat_tensor(last_layers_spectrums)

    grad_mean, grad_median, grad_std, grad_max, grad_min = get_tensor_stats(all_grads)
    grad_ll_mean, grad_ll_median, grad_ll_std, grad_ll_max, grad_ll_min = get_tensor_stats(all_ll_grads)
    spectrum_mean, spectrum_median, spectrum_std, spectrum_max, spectrum_min = get_tensor_stats(all_spectras)
    spectrum_ll_mean, spectrum_ll_median, spectrum_ll_std, spectrum_ll_max, spectrum_ll_min = get_tensor_stats(all_ll_spectrums)

    # grad_alpha, grad_beta = compute_beta_dist_params(grad_mean, grad_std)
    # grad_ll_alpha, grad_ll_beta = compute_beta_dist_params(grad_ll_mean, grad_ll_std)
    spectrum_alpha, spectrum_beta = compute_beta_dist_params(spectrum_mean, spectrum_std)
    spectrum_ll_alpha, spectrum_ll_beta = compute_beta_dist_params(spectrum_ll_mean, spectrum_ll_std)

    input_data = InputData(
        grad_mean, grad_median, grad_std, grad_max, grad_min, # grad_alpha, grad_beta,
        spectrum_mean, spectrum_median, spectrum_std, spectrum_max, spectrum_min, spectrum_alpha, spectrum_beta,
        grad_ll_mean, grad_ll_median, grad_ll_std, grad_ll_max, grad_ll_min, # grad_ll_alpha, grad_ll_beta,
        spectrum_ll_mean, spectrum_ll_median, spectrum_ll_std, spectrum_ll_max, spectrum_ll_min, spectrum_ll_alpha, spectrum_ll_beta,
        epoch, batch_size
    )
    update_csv(input_data, 'dataset/uq_features')

