from llm.data.tokenizer import CONTROL_TOKENS
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

import pickle

class ParallelTrainer:
    def __init__(
            self,
            model: Union[Callable,FSDP,nn.Module],
            optimizer: optim.Optimizer,
            criterion: nn.Module,
            csv_object: InputCSV,
            rank: torch.device,
            world_size: int,
            eval_task: Callable = None,
            name: str = "model",
            model_dir: str = "models/",
            lr_scheduler: optim.lr_scheduler = None,
            bos_token_id: int = 0,
            eos_token_id: int = 0,
            pad_token_id: int = 0,
            len_answer: int = 0
        ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler or torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        self.criterion = criterion
        self.csv_object = csv_object
        self.rank = rank
        self.world_size = world_size
        self.name = name
        self.model_dir = Path(model_dir).joinpath(self.name)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        assert not bos_token_id == eos_token_id, "Start of answer and end of answer tokens should be different"
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.len_answer = len_answer
        self.verbose = True
        self.eval_task = eval_task

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
            val_kwargs: Dict = {},
        ) -> None:
        history = {"train_loss": [], "test_loss": []}
        train_loader = DataLoader(train_set, **train_kwargs)
        val_loader = DataLoader(val_set, **val_kwargs)

        early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

        with tqdm(range(epochs), unit="epoch", disable=not verbose or not self.rank==0) as tepoch:
            for epoch in tepoch:
                self.csv_object.update_hyperparameters(epoch, batch_size)

                if "sampler" in train_kwargs.keys():
                    train_kwargs["sampler"].set_epoch(epoch)
                train_loss = self._train_epoch(train_loader)
                val_loss = self._val_epoch(val_loader)
                self.lr_scheduler.step()
                
                if self.rank == 0:
                    losses = {
                        "train": train_loss,
                        "test": val_loss
                    }
                    self.csv_object(losses, self.eval_task.compute())

                history["train_loss"].append(train_loss)
                history["test_loss"].append(val_loss)

                tepoch.set_postfix(train_loss=train_loss, test_loss=val_loss)

                early_stopping(val_loss)
                if self.rank == 0 and early_stopping.save_model:
                    if self.world_size > 1:
                        self.save_modelcheckpoint(best=True)
                    else:
                        self.save_model(best=True)
                torch.cuda.empty_cache()
                if early_stopping.early_stop:
                    print(f"Early stopping at epoch {epoch}, patience is {early_stopping.patience}") if self.rank == 0 else None
                    break
                
            tepoch.set_postfix(
                loss = history["train_loss"][-1],
                test_loss = history["test_loss"][-1],
            )
        # if self.rank == 0:
        #     self.save_model()
       
        if self.world_size > 1:
            self.save_modelcheckpoint()
        else:
            self.save_model()
            
        with open(self.model_dir / "history.pickle", 'wb') as handle:
            pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _train_epoch(self, train_set, accumulation_steps: int = 1) -> float:
        # return 0
        self.model.train()
        ddp_loss = torch.zeros(2).to(self.rank)

        for i, seq in enumerate(train_set):
            # TODO handle Llama
            assert not torch.isnan(seq).any(), "NaN found in sources!"
            # assert not torch.isnan(tgt).any(), "NaN found in targets!"
            seq = seq.to(self.rank) #, tgt.to(self.rank)
            self.optimizer.zero_grad()
            output = self.model(seq, seq.shape[1] - self.len_answer)[:, -self.len_answer:, :]
            loss = self.criterion(output.view(-1, output.size(-1)), seq[:, -self.len_answer:, :].view(-1))
            loss.backward()
            if (i + 1) % accumulation_steps == 0:
                self.optimizer.step()
                if self.rank == 0 or True:
                    self.csv_object.update_model()
                self.optimizer.zero_grad()
                self.model.zero_grad()
            # self.optimizer.step()
            ddp_loss[0] += loss.item()
            ddp_loss[1] += seq.size(0) * seq.size(1)
            
        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
        test_loss = ddp_loss[0] / ddp_loss[1]
        return float(test_loss.cpu().numpy())
    
    def _val_epoch(self, val_set: DataLoader, mode: str ="greedy") -> float:
        self.model.eval()
        ddp_loss = torch.zeros(2, device=self.rank)  # [total loss, total tokens]

        with torch.no_grad():
            for seq in val_set:
                assert not torch.isnan(seq).any(), "NaN found in sources!"
                seq = seq.to(self.rank)

                batch_size = seq.shape[0]
                seq_len = self.len_answer

                output = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=self.rank)
                finished = torch.zeros(batch_size, dtype=torch.bool, device=self.rank)
                for i in range(seq_len - 1):
                    logits = self.model(seq, seq.shape[1] - seq_len)
                    logits = logits[:, -1, :] 

                    loss = self.criterion(logits, seq[:, seq.shape[1] - seq_len + i])
                    ddp_loss[0] += loss.item()
                    ddp_loss[1] += batch_size 

                    next_token = logits.argmax(dim=-1, keepdim=True)
                    output = torch.cat([output, next_token], dim=1)

                    finished |= (next_token.squeeze(1) == self.eos_token_id)
                    if finished.all():
                        padding = torch.full((batch_size, seq_len - output.size(1)), self.pad_token_id, dtype=torch.long, device=self.rank)
                        output = torch.cat([output, padding], dim=1)
                        break
                self.eval_task.update(refs=seq[:,-seq_len], preds=output)

        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)        
        test_loss = ddp_loss[0] / ddp_loss[1]
        return float(test_loss.cpu().numpy())
    
    @DeprecationWarning
    def _infer(self, seq, mode: str = "greedy"):
        self.model.eval() 
        device = seq.device

        batch_size = seq.shape[0]
        if mode == "greedy":
            return self._greedy(seq, batch_size, device)
        elif mode == "beam":

            return self._beam_search(seq, batch_size, device)
        else:
            return self._greedy(seq, batch_size, device)

    @DeprecationWarning
    def _greedy(self, seq, batch_size, device):
        generated_tokens = torch.full((batch_size, 1), self.bos_token_id, device=device, dtype=torch.long)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        propabilities = nn.functional.one_hot(generated_tokens, num_classes=self.model.vocab_size).cpu()

        for _ in range(self.len_answer - 1):
            with torch.no_grad():
                output = self.model(seq, generated_tokens, has_mask=True) 
            propabilities = torch.cat([propabilities, output.cpu()], dim=1)
            next_tokens = torch.argmax(output[:, -1, :], dim=-1, keepdim=True)
            generated_tokens = torch.cat([generated_tokens, next_tokens], dim=1)

            finished |= (next_tokens.squeeze(1) == self.eos_token_id)
            if finished.all():
                padding = torch.full((batch_size, self.len_answer - output.size(1)), self.pad_token_id, dtype=torch.long, device=self.rank)
                output = torch.cat([output, padding], dim=1)
                break
            
        return propabilities.to(self.rank)

    # def _beam_search(self, src, beam_width=3):
    #     """Performs beam search decoding."""
    #     batch_size = src.shape[0]
    #     beams = [(torch.full((batch_size, 1), self.bos_token_id, device=self.rank), 0)]  
        
    #     for _ in range(self.len_answer):
    #         candidates = []
    #         for seq, score in beams:
    #             logits = self.model(src, seq, has_mask=True)[:, -1, :] 
    #             probs = F.log_softmax(logits, dim=-1)
    #             top_k_probs, top_k_indices = probs.topk(beam_width, dim=-1) 

    #             for i in range(beam_width):
    #                 new_seq = torch.cat([seq, top_k_indices[:, i].unsqueeze(-1)], dim=1)
    #                 new_score = score + top_k_probs[:, i].sum().item()
    #                 candidates.append((new_seq, new_score))
            
    #         beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]

        # return beams[0][0] 
    
    def evaluate(self, tgt, pred):
        self.model.eval()
        with torch.no_grad():
            loss = self.criterion(pred.view(-1, pred.size(-1)), tgt.view(-1))
        return loss.item()
    
    def save_model(self, best: bool = False):
        path = "model.pth" if not best else "best_model.pth"
        path = self.model_dir.joinpath(path)
        print(f"Try to save model at {path}")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            # 'optimizer_state_dict': self.optimizer.state_dict(),
            # 'loss': self.criterion
        }, path)
        # self.save_grads()
    
    def load_model(self, best: bool = False):
        path = "model.pth" if not best else "best.pth"
        path = self.model_dir.joinpath(path)
        try:
            checkpoint = torch.load(self.path, weights_only=False)
        except:
            print(f"Load model on GPU failed, trying to load on CPU")
            checkpoint = torch.load(self.path, weights_only=False, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # self.criterion.load_state_dict(checkpoint['loss'])
        print(f"Model loaded from {path}!")
        return self.model
    
    def save_modelcheckpoint(self, best: bool = False):
        path = "model.pth" if not best else "best.pth"
        path = self.model_dir.joinpath(path)
        print(f"Try to save model checkpoint at {path}")
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
            model_cpu_state = self.model.state_dict()
            # opt_cpu_state = self.optimizer.state_dict()
            # loss_cpu_state = self.criterion

        if self.rank == 0:
        # save_name = file_save_name + "-" + time_of_run + "-" + currEpoch
            torch.save({
                'model_state_dict': model_cpu_state, #self.model.state_dict(),
                # 'optimizer_state_dict': opt_cpu_state, #self.optimizer.state_dict(),
                # 'loss': self.criterion
            }, path)
        print("Model checkpoint saved!")


    def infer(self, seq, tokenizer=None, testing=False):
        self.model.eval() 
        device = seq.device

        if tokenizer:
            assert type(seq) == str or type(seq) == list, "If Tokenizer given for inference, seq type should be str or list of str"
            encodings = tokenizer(seq, return_tensors=True, padding=True, truncation=True)
            seq = encodings.to(device)

        batch_size = seq.shape[0]

        generated_tokens = torch.full((batch_size, 1), self.bos_token_id, device=device, dtype=torch.long)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        propabilities = nn.functional.one_hot(generated_tokens, num_classes=self.model.vocab_size).cpu()

        for _ in range(self.model.max_seq_len - 1):
            with torch.no_grad():
                output = self.model(seq, generated_tokens, has_mask=True) 
            propabilities = torch.cat([propabilities, output.cpu()], dim=1)
            next_tokens = torch.argmax(output[:, -1, :], dim=-1, keepdim=True)
            generated_tokens = torch.cat([generated_tokens, next_tokens], dim=1)

            if testing:
                finished |= (next_tokens.squeeze(1) == self.eos_token_id)
                if finished.all():
                    break
        
        if tokenizer:
            return [tokenizer.decode(seq.tolist()) for seq in generated_tokens]

        return propabilities.to(self.rank)

