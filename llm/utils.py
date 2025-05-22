from typing import Optional, Union
import os
from pathlib import Path

from enum import Enum

class TrainingType(Enum):
    BATCH = "batch"
    ITER = "iter"

class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.05):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.save_model = True

    def __call__(self, test_loss):
        self.save_model = False
        if self.best_loss is None:
            self.best_loss = test_loss
            self.save_model = True
        elif test_loss < self.best_loss - self.min_delta:
            self.best_loss = test_loss
            self.counter = 0
            self.save_model = True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def get_model_dir(model_name: Union[str, Path], model_dir: Optional[str] = None, training_type: Optional[str] = "batch", return_error: bool = False) -> Path:
    """
    Get the model directory for a given model name.

    Args:
        model_name (str or Path): The name of the model.
        model_dir (str, optional): The directory where the model is stored. Defaults to None.
        training_type (str, optional): The type of training ('batch' or 'iter'). Defaults to None.
    
    Returns:
        model_dir (Path): The path to the model directory.
    """
    assert training_type in TrainingType._value2member_map_, "Training type not supported. Choose between 'batch' and 'iter'"
    training_type = TrainingType(training_type)
    
    model_name += f".{training_type.value}"
    if model_dir is None:
        model_dir = os.getenv("MODEL_DIR", "/tmp/models")

    model_dir = Path(model_dir).joinpath(model_name)
    if isinstance(model_dir, str):
        model_dir = Path(model_dir)
    if not model_dir.exists():
        if return_error:
            raise FileNotFoundError(f"Model directory {model_dir} does not exist.")
        else:
            # Create the directory if it doesn't exist
            # Print a warning message
            print(Warning(f"Model directory {model_dir} does not exist. Creating it."))
            model_dir.mkdir(parents=True, exist_ok=True)

    return model_dir

class WarmUp:
    def __init__(self, optimizer, warmup_steps: int = 1000, current_step: int = 0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.current_step = current_step
        self.mean_lr = 0
        self.initial_lr = [ param['lr'] for param in optimizer.param_groups ]
        
        self.step()

    def step(self):
        mean_lr = 0
        if self.current_step < self.warmup_steps:
            self.current_step += 1
            for initial_lr, param_group in zip(self.initial_lr, self.optimizer.param_groups):
                param_group['lr'] = (self.current_step / self.warmup_steps) * initial_lr
                mean_lr += param_group['lr']
        else:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['initial_lr']
                mean_lr += param_group['lr']
        
        
        self.mean_lr = mean_lr / len(self.optimizer.param_groups)
        
