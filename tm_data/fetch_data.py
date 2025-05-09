from tm_data.features import get_tensor_stats, compute_cosine_similarity, compute_grad_dir

import pandas as pd
import numpy as np
import torch
from torch import nn

from typing import Dict, List, Union
from dataclasses import dataclass, fields, make_dataclass, field
from pathlib import Path


@dataclass
class GradStats:
    grad_abs_mean: float | None = None
    grad_mean: float | None = None
    grad_median: float | None = None
    grad_std: float | None = None
    grad_max: float | None = None
    grad_min: float | None = None
    grad_dir: float | None = None
    grad_cos_dist: float | None = None
    grad_noise_scale: float | None = None

@dataclass
class BaseData(GradStats):
    train_loss: float | None = None
    var_train_loss: float | None = None
    epoch: int | None = None

@dataclass
class BatchData(BaseData):
    batch_ids: List[int] | None = None

@dataclass
class InputData(BaseData):
    val_loss: float | None = None
    var_val_loss: float | None = None
    mean_lr: float | None = None
    batch_size: int | None = None


class TrainingDataFetcher:
    def __init__(
            self, 
            model: nn.Module, 
            model_dir: Union[Path, str], 
            world_size: int = 0,
            train_metrics: List[str] = None,
            val_metrics: List[str] = None
        ) -> None:
        if type(model_dir) == str:
            model_dir = Path(model_dir)
        self.path = { "epoch": model_dir / "fetched_training_data", "batch": model_dir / "fetched_batch_data" }
        for doc in self.path.keys():
            if not self.path[doc].suffix == '.csv':
                self.path[doc] = self.path[doc].with_suffix('.csv')
                
        self.model = model

        self.eval_metrics = [ 
            f"{metric}_train" if metric in val_metrics else metric for metric in train_metrics 
            ] + [ f"{metric}_val" if metric in train_metrics else metric for metric in val_metrics ]
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.current_grads = None

        self.CurrentInput = make_dataclass(
            "CurrentInput",
            fields=[(f.name, f.type, field(default=None)) for f in fields(InputData)] +
                [(name, float, field(default=None)) for name in self.eval_metrics]
        )
        self.CurrentBatch = make_dataclass(
            "CurrentBatch",
            fields=[(f.name, f.type, field(default=None)) for f in fields(BatchData)]
        )

        self.init_data()
        self.world_size = world_size
        self.last_layers = None
        print("Look for csv at", self.path)
        for doc in self.path.keys():
            if not self.path[doc].exists():
                self.create_csv(doc)
                print("CSV created at:", self.path[doc])

        self.acc_steps = 0


    def init_data(self, last_values: Union[Dict[str,float], None] = None) -> None:
        self.clean_grads()
        self.last_val_loss = last_values["val_loss"] if last_values else None
        self.last_train_loss = last_values["train_loss"] if last_values else None

        self.current = self.CurrentInput()


    def __call__(self, losses, train_metrics: Dict, val_metrics: Dict) -> None:
        assert self.current.batch_size, "You must update the hyperparameters before updating the model. Call 'update_hyperparameters' first."
        assert self.current.epoch or self.current.epoch == 0, "You must update the hyperparameters before updating the model. Call 'update_hyperparameters' first."

        self.current.val_loss = losses["val"]
        self.current.var_val_loss = losses["val"] - self.last_val_loss if self.last_val_loss else losses["val"]
        self.current.train_loss = losses["train"]
        self.current.var_train_loss = losses["train"] - self.last_train_loss if self.last_train_loss else losses["train"]
        metrics = {
            **{f"{metric}_train" if metric in val_metrics else metric: value for metric, value in train_metrics.items()},
            **{f"{metric}_val" if metric in train_metrics else metric: value for metric, value in val_metrics.items()}
        }
        for metric in self.eval_metrics:
            setattr(self.current, metric, metrics[metric])
        last_values = {
            "val_loss": losses["val"],
            "train_loss": losses["train"]
        }
        self.compute_grad_stats("epoch")
        # add a way to store loss variations
        # self.compute_model_stats("epoch")
        self.update_csv("epoch")
        self.init_data(last_values)
        self.acc_steps = 0
    
    def update_batch_csv(self, current_loss, current_metrics, batch_ids: List[int]) -> None:
        self.acc_steps += 1
        self.current_batch = self.CurrentBatch()
        self.current_batch.epoch = self.epoch
        self.current_batch.batch_ids = batch_ids
        self.current_batch.train_loss = current_loss
        self.current_batch.var_train_loss = current_loss - self.last_batch_train_loss if hasattr(self, "last_batch_train_loss") else current_loss

        batch_grads = [ torch.clone(p.grad) for p in self.model.parameters() if p.grad is not None ]
        self.batch_grads = batch_grads[0].view(-1) if len(batch_grads) == 1 else get_concat_tensor(batch_grads)
        self.update_model()

        metrics = {f"{metric}_train": value for metric, value in current_metrics.items()}
        
        for metric in metrics.keys():
            setattr(self.current_batch, metric, metrics[metric])

        self.compute_grad_stats("batch")
        self.update_csv("batch")

        self.last_batch_grads = self.batch_grads
        del self.batch_grads
        self.last_batch_train_loss = current_loss

    def update_hyperparameters(self, epoch: int, batch_size: int, mean_lr: float) -> None:
        #TODO fine a new name
        self.epoch = epoch
        self.current.epoch = epoch
        self.current.batch_size = batch_size
        self.current.mean_lr = mean_lr
        
    def update_model(self) -> None:
        # PB 1: can't do torch(grads) or np.array(grads) because of the different shapes for each layer
        # PB 2: store grads of all epochs maybe very heavy computationally
        # Temporary solution: take the stats at each batch iteration at store the mean of each stats at the end of the epoch
        # self.model.clean_nan()
        grads = self.batch_grads
        self.current_grads = grads if not hasattr(self, "current_grads") else [ grad + current for grad, current in zip(grads, self.current_grads) ]
        # print("current_grads:", self.current_grads)

    @torch.inference_mode()
    def compute_grad_stats(self, doc: str = "epoch") -> None:
        grads = self.current_grads / self.acc_steps if doc == "epoch" else self.batch_grads
        # self.current_grads = torch.clone(grads)

        last_grads = self.last_iter_grads if doc == "epoch" else self.last_batch_grads if hasattr(self, "last_batch_grads") else None
        # grad_mean, grad_median, grad_std, grad_max, grad_min, grad_noise_scale = get_tensor_stats(grads)
        grad_stats = GradStats(
            grad_dir=compute_grad_dir(grads=grads, last_grads=last_grads), 
            grad_cos_dist=compute_cosine_similarity(grads1=grads, grads2=last_grads), 
            **get_tensor_stats(grads)
        )
        curr = self.current if doc == "epoch" else self.current_batch
        for fn in grad_stats.__dataclass_fields__.keys():
            setattr(curr, fn, getattr(grad_stats, fn))
        
        if doc == "epoch":
            self.current = curr
        elif doc == "batch":
            self.current_batch = curr


    def update_csv(self, doc = "epoch") -> None:
        if doc == "epoch":
            current = self.current
            currentClass = self.CurrentInput
        elif doc == "batch":
            current = self.current_batch
            currentClass = self.CurrentBatch
        
        for fn in currentClass.__dataclass_fields__.keys():
            assert hasattr(current, fn), f"Attribute {fn} is missing from the current data."

        fields = currentClass.__dataclass_fields__.keys()
        if not self.path[doc].exists():
            self.create_csv(doc)
        df = pd.DataFrame([{fn: getattr(current, fn) for fn in fields}])
        df.to_csv(self.path[doc], mode='a', header=False, index=False)
    
    def clean_grads(self) -> None:
        # self.current_grad_stats = []
        self.last_iter_grads = self.current_grads
        del self.current_grads

    def create_csv(self, doc: str = "epoch") -> None:
        currentClass = self.CurrentInput if doc == "epoch" else self.CurrentBatch
        if not self.path[doc].exists():
            self.path[doc].parent.mkdir(parents=True, exist_ok=True)
            with open(self.path[doc], 'w') as f:
                f.write(f'{",".join(currentClass.__dataclass_fields__.keys())}\n')
    
        

def get_concat_tensor(tensor: torch.Tensor) -> torch.Tensor:
    all_tensors = torch.cat([t.view(-1) for t in tensor])
    return all_tensors

def get_last_layers(grads: List[torch.Tensor], nb_last_layers: int = 5) -> List[torch.Tensor]:
    return [ grads[-layer] for layer in range(1, nb_last_layers + 1)]



