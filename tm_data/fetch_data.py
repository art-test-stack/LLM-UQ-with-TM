from tm_data.features import get_tensor_stats, compute_beta_dist_params, get_grad_spectrums, compute_cosine_similarity, compute_grad_dir

import pandas as pd
import numpy as np
import torch
from torch import nn

from typing import Callable, Dict, List, Union
from dataclasses import dataclass, fields, make_dataclass, field
from pathlib import Path

@dataclass
class GradStats:
    grad_mean: float | None = None
    grad_median: float | None = None
    grad_std: float | None = None
    grad_max: float | None = None
    grad_min: float | None = None
    grad_dir: float | None = None
    grad_cos_dist: float | None = None
    grad_noise_scale: float | None = None


@dataclass
class InputData(GradStats):
    test_loss: float | None = None
    var_test_loss: float | None = None
    train_loss: float | None = None
    var_train_loss: float | None = None
    # confidence_score: float | None = None
    epoch: int | None = None
    batch_size: int | None = None
    mean_lr: float | None = None


class TrainingDataFetcher:
    def __init__(
            self, 
            model: nn.Module, 
            path: Union[Path, str], 
            world_size: int = 0,
            train_metrics: List[str] = None,
            val_metrics: List[str] = None
        ) -> None:
        if type(path) == str:
            path = Path(path)
        self.path = path
        if not self.path.suffix == '.csv':
            self.path = self.path.with_suffix('.csv')
        self.model = model

        self.eval_metrics = [ 
            f"{metric}_train" if metric in val_metrics else metric for metric in train_metrics 
            ] + [ f"{metric}_val" if metric in train_metrics else metric for metric in val_metrics ]
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.current_grads = None
        self.init_data()
        self.world_size = world_size
        self.last_layers = None
        print("Look for csv at", self.path)
        if not self.path.exists():
            self.create_csv(self.path)
            print("CSV created at:", self.path)

    def init_data(self, last_values: Union[Dict[str,float], None] = None) -> None:
        self.clean_grads()
        self.last_test_loss = last_values["test_loss"] if last_values else None
        self.last_train_loss = last_values["train_loss"] if last_values else None
        self.CurrentInput = make_dataclass(
            "CurrentInput",
            fields=[(f.name, f.type, field(default=None)) for f in fields(InputData)] +
                [(name, float, field(default=None)) for name in self.eval_metrics]
        )
        self.current = self.CurrentInput()

    def __call__(self, losses, train_metrics: Dict, val_metrics: Dict) -> None:
        assert self.current.batch_size, "You must update the hyperparameters before updating the model. Call 'update_hyperparameters' first."
        assert self.current.epoch or self.current.epoch == 0, "You must update the hyperparameters before updating the model. Call 'update_hyperparameters' first."

        self.current.test_loss = losses["test"]
        self.current.var_test_loss = losses["test"] - self.last_test_loss if self.last_test_loss else losses["test"]
        self.current.train_loss = losses["train"]
        self.current.var_train_loss = losses["train"] - self.last_train_loss if self.last_train_loss else losses["train"]
        metrics = {
            **{f"{metric}_train" if metric in val_metrics else metric: value for metric, value in train_metrics.items()},
            **{f"{metric}_val" if metric in train_metrics else metric: value for metric, value in val_metrics.items()}
        }
        for metric in self.eval_metrics:
            setattr(self.current, metric, metrics[metric])
        last_values = {
            "test_loss": losses["test"],
            "train_loss": losses["train"]
        }
        self.compute_grad_stats()
        # add a way to store loss variations
        self.compute_model_stats()
        self.update_csv()
        self.init_data(last_values)
    
    def update_hyperparameters(self, epoch: int, batch_size: int, mean_lr: float) -> None:
        #TODO fine a new name
        self.current.epoch = epoch
        self.current.batch_size = batch_size
        self.current.mean_lr = mean_lr

    def update_model(self) -> None:
        # PB 1: can't do torch(grads) or np.array(grads) because of the different shapes for each layer
        # PB 2: store grads of all epochs maybe very heavy computationally
        # Temporary solution: take the stats at each batch iteration at store the mean of each stats at the end of the epoch
        # self.model.clean_nan()
        try:
            grads = self.model.get_grads()
        except:
            grads = [ p.grad for p in self.model.parameters() if p.grad is not None ]
        grads = [ torch.clone(grad) for grad in grads ]
    
        self.current_grads = grads if not self.current_grads else [ grad + current for grad, current in zip(grads, self.current_grads) ]
        # print("current_grads:", self.current_grads)

    @torch.inference_mode()
    def compute_grad_stats(self) -> None:
        grads = self.current_grads[0].view(-1) if len(self.current_grads) == 1 else get_concat_tensor(self.current_grads)

        # grad_mean, grad_median, grad_std, grad_max, grad_min, grad_noise_scale = get_tensor_stats(grads)
        grad_stats = GradStats(
            grad_dir=compute_grad_dir(grads=grads, last_grads=self.last_iter_grads), 
            grad_cos_dist=compute_cosine_similarity(grads1=grads, grads2=self.last_iter_grads), 
            **get_tensor_stats(grads)
        )
        self.current_grad_stats.append(grad_stats)

    def compute_model_stats(self) -> None:
        grad_stats = GradStats()
        for fn in grad_stats.__dataclass_fields__.keys():
            fn_values = [getattr(grad, fn) for grad in self.current_grad_stats]
            fn_mean = np.mean(fn_values)
            grad_stats.__setattr__(fn, fn_mean)
        for fn in grad_stats.__dataclass_fields__.keys():
            setattr(self.current, fn, getattr(grad_stats, fn)) 

    def update_csv(self) -> None:
        for fn in self.CurrentInput.__dataclass_fields__.keys():
            assert hasattr(self.current, fn), f"Attribute {fn} is missing from the current data."

        fields = self.CurrentInput.__dataclass_fields__.keys()
        if not self.path.exists():
            self.create_csv(self.path)
        df = pd.DataFrame([{fn: getattr(self.current, fn) for fn in fields}])
        df.to_csv(self.path, mode='a', header=False, index=False)
    
    def clean_grads(self) -> None:
        self.current_grad_stats = []
        self.last_iter_grads = self.current_grads
        self.current_grads = None

    def create_csv(self, path: str) -> None:
        with open(path, 'w') as f:
            f.write(f'{",".join(self.CurrentInput.__dataclass_fields__.keys())}\n')
    
        

def get_concat_tensor(tensor: torch.Tensor) -> torch.Tensor:
    all_tensors = torch.cat([t.view(-1) for t in tensor])
    return all_tensors

def get_last_layers(grads: List[torch.Tensor], nb_last_layers: int = 5) -> List[torch.Tensor]:
    return [ grads[-layer] for layer in range(1, nb_last_layers + 1)]



