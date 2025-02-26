from data.tokenizer import CONTROL_TOKENS
from tm_data.preprocessing import InputCSV
from llm.utils import EarlyStopping
from llm.model import LLM

import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

from typing import Callable, Dict, Union
from tqdm import tqdm
from pathlib import Path

class ParallelTrainer:
    def __init__(
            self,
            model: Union[Callable,FSDP,nn.Module],
            optimizer: optim.Optimizer,
            criterion: nn.Module,
            csv_object: InputCSV,
            rank: torch.device,
            world_size: int,
            name: str = "model",
            model_dir: str = "models/",
            soa_token_id: int = 0,
            eoa_token_id: int = 0,
            pad_token_id: int = 0
        ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.csv_object = csv_object
        self.rank = rank
        self.world_size = world_size
        self.name = name
        self.path = Path(f"{model_dir}/{self.name}.pt")
        assert not soa_token_id == eoa_token_id, "Start of answer and end of answer tokens should be different"
        self.soa_token_id = soa_token_id
        self.eoa_token_id = eoa_token_id
        self.pad_token_id = pad_token_id
        self.verbose = True

    def fit(
            self, 
            train_set, 
            val_set, 
            epochs: int = 100,
            batch_size: int = 32,
            patience: int = 5,
            min_delta: float = 0.05,
            verbose: bool = True,
            train_kwargs: Dict = {},
            val_kwargs: Dict = {}
        ) -> None:

        history = {"train_loss": [], "test_loss": []}
        train_loader = DataLoader(train_set, **train_kwargs)
        val_loader = DataLoader(val_set, **val_kwargs)

        early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

        with tqdm(range(epochs), unit="epoch", disable=not verbose or not self.rank==0) as tepoch:
            for epoch in tepoch:
                self.csv_object.update_hyperparameters(epoch, batch_size)

                if train_kwargs["sampler"]:
                    train_kwargs["sampler"].set_epoch(epoch)
                train_loss = self._train_epoch(train_loader)
                val_loss = self._val_epoch(val_loader)
                if self.rank == 0:
                    self.csv_object(val_loss)

                history["train_loss"].append(train_loss)
                history["test_loss"].append(val_loss)

                tepoch.set_postfix(train_loss=train_loss, test_loss=val_loss)

                early_stopping(val_loss)
                if self.rank == 0 and early_stopping.save_model:
                    if self.world_size > 1:
                        self.save_modelcheckpoint()
                    else:
                        self.save_model()
                torch.cuda.empty_cache()
                if early_stopping.early_stop:
                    print(f"Early stopping at epoch {epoch}, patience is {early_stopping.patience}") if self.rank == 0 else None
                    break
                
            tepoch.set_postfix(
                loss = history["train_loss"][-1],
                test_loss = history["test_loss"][-1],
            )
        if self.rank == 0:
            self.save_model()

    def _train_epoch(self, train_set, accumulation_steps: int = 1) -> float:
        # return 0
        self.model.train()
        ddp_loss = torch.zeros(2).to(self.rank)

        for i, (src, tgt) in enumerate(train_set):
            # self.model.zero_grad()
            assert not torch.isnan(src).any(), "NaN found in sources!"
            assert not torch.isnan(tgt).any(), "NaN found in targets!"
            src, tgt = src.to(self.rank), tgt.to(self.rank)
            self.optimizer.zero_grad()
            output = self.model(src, tgt)
            loss = self.criterion(output.view(-1, output.size(-1)), tgt.view(-1))
            loss.backward()
            if (i + 1) % accumulation_steps == 0:
                self.optimizer.step()
                if self.rank == 0 or True:
                    self.csv_object.update_model()
                self.optimizer.zero_grad()
            # self.optimizer.step()
            ddp_loss[0] += loss.item()
            ddp_loss[1] += len(src)
            
            
        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
        test_loss = ddp_loss[0] / ddp_loss[1]
        return float(test_loss.cpu().numpy())
    
    def _val_epoch(self, val_set, use_beam_search=False, beam_width=3) -> float:
        self.model.eval()
        ddp_loss = torch.zeros(2, device=self.rank)  # [total loss, total tokens]

        with torch.no_grad():
            # for src, tgt in val_set:
            src, tgt = next(iter(val_set))
            assert not torch.isnan(src).any(), "NaN found in sources!"
            assert not torch.isnan(tgt).any(), "NaN found in targets!"

            src, tgt = src.to(self.rank), tgt.to(self.rank)
            batch_size, seq_len = tgt.shape

            output = torch.full((batch_size, 1), self.soa_token_id, dtype=torch.long, device=self.rank)
            # pad_tokens = torch.full((batch_size, seq_len - 1), self.pad_token_id, dtype=torch.long, device=self.rank)
            # output = torch.cat([output, pad_tokens], dim=1)

            total_loss = 0.0
            total_tokens = 0
            if use_beam_search:
                # TODO
                predicted_output = self._beam_search(src, beam_width)
            else:
                for i in range(seq_len - 1):
                    logits = self.model(src, output, has_mask=False)
                    logits = logits[:, -1, :] 

                    loss = self.criterion(logits, tgt[:, i])
                    total_loss += loss.item()
                    total_tokens += batch_size

                    next_token = logits.argmax(dim=-1, keepdim=True)
                    output = torch.cat([output, next_token], dim=1)

            ddp_loss[0] += total_loss 
            ddp_loss[1] += total_tokens
            

        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)        
        test_loss = ddp_loss[0] / ddp_loss[1]
        return float(test_loss.cpu().numpy())

    def _beam_search(self, src, beam_width=3):
        """Performs beam search decoding."""
        batch_size = src.shape[0]
        beams = [(torch.full((batch_size, 1), self.soa_token_id, device=self.rank), 0)]  
        
        for _ in range(self.model.max_content - 1):
            candidates = []
            for seq, score in beams:
                logits = self.model(src, seq)[:, -1, :] 
                probs = F.log_softmax(logits, dim=-1)
                top_k_probs, top_k_indices = probs.topk(beam_width, dim=-1) 

                for i in range(beam_width):
                    new_seq = torch.cat([seq, top_k_indices[:, i].unsqueeze(-1)], dim=1)
                    new_score = score + top_k_probs[:, i].sum().item()
                    candidates.append((new_seq, new_score))
            
            beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]

        return beams[0][0] 
    
    def save_model(self):
        print(f"Try to save model at {self.path}")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.criterion
        }, self.path)
        # self.save_grads()
    
    def load_model(self):
        checkpoint = torch.load(self.path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.criterion = checkpoint['loss']
        self.load_grads()
        return self.model
    
    def save_modelcheckpoint(self):
        print(f"Try to save model checkpoint at {self.path}")
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
            model_cpu_state = self.model.state_dict()
            opt_cpu_state = self.optimizer.state_dict()
            loss_cpu_state = self.criterion

        if self.rank == 0:
        # save_name = file_save_name + "-" + time_of_run + "-" + currEpoch
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.criterion
            }, self.path)


    def infer(self, seq, tokenizer=None, testing=False):
        self.model.eval() 
        device = seq.device

        if tokenizer:
            assert type(seq) == str or type(seq) == list, "If Tokenizer given for inference, seq type should be str or list of str"
            encodings = tokenizer(seq, return_tensors=True, padding=True, truncation=True)
            seq = encodings.to(device)

        batch_size = seq.shape[0]

        generated_tokens = torch.full((batch_size, 1), self.soa_token_id, device=device, dtype=torch.long)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        propabilities = nn.functional.one_hot(generated_tokens, num_classes=self.model.vocab_size).cpu()

        for _ in range(self.model.max_content - 1):
            with torch.no_grad():
                output = self.model(seq, generated_tokens, has_mask=True) 
            propabilities = torch.cat([propabilities, output.cpu()], dim=1)
            next_tokens = torch.argmax(output[:, -1, :], dim=-1, keepdim=True)
            generated_tokens = torch.cat([generated_tokens, next_tokens], dim=1)

            if testing:
                finished |= (next_tokens.squeeze(1) == self.eoa_token_id)
                if finished.all():
                    break
        
        if tokenizer:
            return [tokenizer.decode(seq.tolist()) for seq in generated_tokens]

        return propabilities.to(self.rank)

