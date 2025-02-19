from tm_data.preprocessing import InputCSV
from llm.utils import EarlyStopping
from llm.model import LLM

import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

from typing import Callable
from tqdm import tqdm

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class ParallelTrainer:
    def __init__(
            self,
            model: FSDP,
            optimizer: optim.Optimizer,
            criterion: nn.Module,
            csv_object: InputCSV,
            rank: torch.device,
            world_size: int,
            name: str = "model"
        ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.csv_object = csv_object
        self.rank = rank
        self.world_size = world_size
        self.name = name

    def fit(
            self, 
            train_set, 
            val_set, 
            epochs: int = 100,
            batch_size: int = 32,
            patience: int = 5,
            min_delta: float = 0.05,
            verbose: bool = True,
            train_sampler = None,
        ) -> None:

        history = {"train_loss": [], "test_loss": []}
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, drop_last=True)

        early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

        with tqdm(range(epochs), unit="epoch", disable=not verbose) as tepoch:
            for epoch in tepoch:
                self.csv_object.update_hyperparameters(epoch, batch_size)

                if train_sampler:
                    train_sampler.set_epoch(epoch)
                train_loss = self._train_epoch(train_loader)
                test_loss = self._test_epoch(val_loader)
                self.csv_object(test_loss)

                history["train_loss"].append(train_loss)
                history["test_loss"].append(test_loss)

                tepoch.set_postfix(train_loss=train_loss, test_loss=test_loss)

                early_stopping(test_loss)
                if self.rank == 0 and early_stopping.early_stop:
                    break

                # history["test_loss"].append(test_loss / len(dev_loader.dataset))

            tepoch.set_postfix(
                loss = history["train_loss"][-1],
                test_loss = history["test_loss"][-1],
            )
        if self.rank == 0:
            self.save_model()

    def _train_epoch(self, train_set) -> float:
        self.model.train()
        total_loss = 0

        ddp_loss = torch.zeros(2).to(self.rank)

        for src, tgt in train_set:
            assert not torch.isnan(src).any(), "NaN found in sources!"
            assert not torch.isnan(tgt).any(), "NaN found in targets!"

            src, tgt = src.to(self.rank), tgt.to(self.rank)
            self.optimizer.zero_grad()
            output = self.model(src, tgt)
            loss = self.criterion(output.view(-1, output.size(-1)), tgt.view(-1))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            ddp_loss[0] += loss.item()
            ddp_loss[1] += len(src)
            self.csv_object.update_model()
            break
        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
        if self.rank == 0:
            # print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, ddp_loss[0] / ddp_loss[1]))
            return ddp_loss[0] / ddp_loss[1]
    
    def _test_epoch(self, val_set) -> float:
        self.model.eval()
        total_loss = 0
        ddp_loss = torch.zeros(2).to(self.rank) # can use other dims for other metrics eg. accuracy

        with torch.no_grad():
            for src, tgt in val_set:
                assert not torch.isnan(src).any(), "NaN found in sources!"
                assert not torch.isnan(tgt).any(), "NaN found in targets!"
                src, tgt = src.to(self.rank), tgt.to(self.rank)
                output = self.model(src, tgt)
                loss = self.criterion(output.view(-1, output.size(-1)), tgt.view(-1))
                total_loss += loss.item()
                ddp_loss[0] += self.criterion(output, tgt).item()
                ddp_loss[1] += len(src)
                break
        
        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
        if self.rank == 0:
            return ddp_loss[0] / ddp_loss[1]
    
    def save_model(self):
        path = f"models/{self.name}.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.criterion
        }, path)
        # self.save_grads()
    
    def load_model(self):
        path = f"models/{self.name}.pt"
        checkpoint = torch.load(path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.criterion = checkpoint['loss']
        self.load_grads()
        return self.model
    
    # def save_grads(self):
    #     path = f"models/{self.name}.pt"
    #     for id, param in enumerate(self.model.parameters()):
    #         if param.grad is not None:
    #             # print(param.grad.shape)
    #             torch.save(param.grad, path + f"_{id}_{param.shape}.grad.pt")
    
    # def load_grads(self):
    #     path = f"models/{self.name}.pt"
    #     for id, param in enumerate(self.model.parameters()):
    #         grad = torch.load(path + f"_{id}_{param.shape}.grad.pt")
    #         param.grad = grad
    #     return self.model


