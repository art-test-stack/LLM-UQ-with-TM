from tm_data.features import get_tensor_stats, compute_beta_dist_params, get_grad_spectrums, compute_cosine_similarity

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

    # spectrum_mean: float | None = None
    # spectrum_median: float | None = None
    # spectrum_std: float | None = None
    # spectrum_max: float | None = None
    # spectrum_min: float | None = None
    # spectrum_alpha: float | None = None
    # spectrum_beta: float | None = None

    # grad_ll_mean: float | None = None
    # grad_ll_median: float | None = None
    # grad_ll_std: float | None = None
    # grad_ll_max: float | None = None
    # grad_ll_min: float | None = None

    # grad_ll_alpha: float | None = None
    # grad_ll_beta: float | None = None

    # spectrum_ll_mean: float | None = None
    # spectrum_ll_median: float | None = None
    # spectrum_ll_std: float | None = None
    # spectrum_ll_max: float | None = None
    # spectrum_ll_min: float | None = None
    # spectrum_ll_alpha: float | None = None
    # spectrum_ll_beta: float | None = None


@dataclass
class InputData(GradStats):
    test_loss: float | None = None
    var_test_loss: float | None = None
    epoch: int | None = None
    batch_size: int | None = None


class InputCSV:
    def __init__(self, model: nn.Module, path: Union[Path, str], world_size: int = 0):
        if type(path) == str:
            self.path = Path(path)
        if not self.path.suffix == '.csv':
            self.path = self.path.with_suffix('.csv')
        if not self.path.exists():
            create_csv(self.path)
        self.model = model
        self.init_data()
        self.world_size = world_size
        self.last_layers = None

    def init_data(self, last_test_loss: Union[float, None] = None) -> None:
        self.current = InputData()
        self.clean_grads()
        self.last_test_loss = last_test_loss

    def __call__(self, test_loss: float) -> None:
        assert self.current.batch_size, "You must update the hyperparameters before updating the model. Call 'update_hyperparameters' first."
        assert self.current.epoch or self.current.epoch == 0, "You must update the hyperparameters before updating the model. Call 'update_hyperparameters' first."

        self.current.test_loss = test_loss
        self.current.var_test_loss = test_loss - self.last_test_loss if self.last_test_loss else test_loss
        self.compute_grad_stats()
        # add a way to store loss variations
        self.compute_model_stats()
        self.update_csv()
        self.init_data(last_test_loss=test_loss)
    
    def update_hyperparameters(self, epoch: int, batch_size: int) -> None:
        self.current.epoch = epoch
        self.current.batch_size = batch_size

    def update_model(self) -> None:
        # PB 1: can't do torch(grads) or np.array(grads) because of the different shapes for each layer
        # PB 2: store grads of all epochs maybe very heavy computationally
        # Temporary solution: take the stats at each batch iteration at store the mean of each stats at the end of the epoch
        # self.model.clean_nan()
        grads = self.model.get_grads()
        grads = [ torch.clone(grad) for grad in grads ]
    
        # for i, grad in enumerate(grads):
        #     print(f"grad {i}.shape", grad.shape)
        self.current_grads = grads if not self.current_grads else [ grad + current for grad, current in zip(grads, self.current_grads) ]
        # print("current_grads:", self.current_grads)

    def compute_grad_stats(self) -> None:
        grads = self.current_grads[0].view(-1) if len(self.current_grads) == 1 else get_concat_tensor(self.current_grads)
        # last_layers_grads = get_last_layers(grads)
        # all_ll_grads = get_concat_tensor(last_layers_grads)

        # grads_cosine_sim = compute_cosine_similarity(all_grads, self.last_layers)
        # self.last_layers = all_grads # [ grad_layer.copy().cpu() for grad_layer in all_grads ] 

        # spectrums = get_grad_spectrums(grads)
        # all_spectras = get_concat_tensor(spectrums)
        # last_layers_spectrums = get_last_layers_spectrums(spectrums)
        # all_ll_spectrums = get_concat_tensor(last_layers_spectrums)

        grad_mean, grad_median, grad_std, grad_max, grad_min = get_tensor_stats(grads)
        # grad_ll_mean, grad_ll_median, grad_ll_std, grad_ll_max, grad_ll_min = get_tensor_stats(all_ll_grads)
        # spectrum_mean, spectrum_median, spectrum_std, spectrum_max, spectrum_min = get_tensor_stats(all_spectras)
        # spectrum_ll_mean, spectrum_ll_median, spectrum_ll_std, spectrum_ll_max, spectrum_ll_min = get_tensor_stats(all_ll_spectrums)

        # spectrum_alpha, spectrum_beta = compute_beta_dist_params(spectrum_mean, spectrum_std)
        # spectrum_ll_alpha, spectrum_ll_beta = compute_beta_dist_params(spectrum_ll_mean, spectrum_ll_std)

        grad_stats = GradStats(
            grad_mean, grad_median, grad_std, grad_max, grad_min,
            # spectrum_mean, spectrum_median, spectrum_std, spectrum_max, spectrum_min, spectrum_alpha, spectrum_beta,
            # grad_ll_mean, grad_ll_median, grad_ll_std, grad_ll_max, grad_ll_min,
            # spectrum_ll_mean, spectrum_ll_median, spectrum_ll_std, spectrum_ll_max, spectrum_ll_min, spectrum_ll_alpha, spectrum_ll_beta
        )
        self.current_grad_stats.append(grad_stats)

    def compute_model_stats(self) -> None:
        grad_stats = GradStats()
        for fn in grad_stats.__dataclass_fields__.keys():
            fn_values = [getattr(grad, fn) for grad in self.current_grad_stats]
            fn_mean = np.mean(fn_values)
            grad_stats.__setattr__(fn, fn_mean)
        self.current = InputData(
            **{fn: getattr(grad_stats, fn) for fn in grad_stats.__dataclass_fields__.keys()},
            test_loss=self.current.test_loss, var_test_loss=self.current.var_test_loss, 
            epoch=self.current.epoch, batch_size=self.current.batch_size
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
        self.current_grad_stats = []
        self.current_grads = None
        
def get_concat_tensor(tensor: torch.Tensor) -> torch.Tensor:
    all_tensors = torch.cat([t.view(-1) for t in tensor])
    return all_tensors

def get_last_layers(grads: List[torch.Tensor], nb_last_layers: int = 5) -> List[torch.Tensor]:
    return [ grads[-layer] for layer in range(1, nb_last_layers + 1)]

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


