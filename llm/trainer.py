from tm_data.preprocessing import InputCSV
from llm.handlers.handler import ModelType
from llm.utils import EarlyStopping
from utils import get_cuda_allocation, get_device


import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from typing import Callable, Dict, Union
from tqdm import tqdm
from pathlib import Path

import math
import pickle
import time

from enum import Enum


class TrainingType(Enum):
    BATCH = "batch"
    ITER = "iter"


class Trainer:
    """
    Trainer class for training the model.
    
    Args:
        model: nn.Module, model
        optimizer: torch.optim.Optimizer, optimizer
        loss_fn: nn.Module, loss function
        rank: torch.device, rank of the process
        world_size: int, world size
        csv_object: InputCSV, object for saving training metrics
        lr_scheduler: torch.optim.lr_scheduler, learning rate scheduler
        model_type: str, type of model
        no_cuda: bool, no cuda
        training_type: str, training type
        name: str, name of the model
        model_dir: str, directory for saving the model
        bos_token_id: int, beginning of sentence token id
        eos_token_id: int, end of sentence token id
        pad_token_id: int, padding token id
        verbose: bool, verbosity
    """
    def __init__(
            self,
            model: Union[Callable,FSDP,nn.Module],
            optimizer: optim.Optimizer,
            loss_fn: nn.Module,
            rank: torch.device,
            world_size: int,
            csv_object: InputCSV = None,
            lr_scheduler: optim.lr_scheduler = None,
            **kwargs
        ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler or torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        self.loss_fn = loss_fn
        self.rank = rank
        self.world_size = world_size

        self.model_type = kwargs.get("model_type", "torch")
        self.no_cuda = kwargs.get("no_cuda", False)

        training_type = kwargs.get("training_type", "normal")
        assert training_type in TrainingType._value2member_map_, "Training type not supported. Choose between 'normal' and 'iter'"
        self.training_type = TrainingType(training_type)

        self.name = kwargs.get("name", "model") + f".{self.training_type.value}"

        model_dir = kwargs.get("model_dir", "models/")
        self.model_dir = Path(model_dir).joinpath(self.name)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.bos_token_id = kwargs.get("bos_token_id", 0)
        self.eos_token_id = kwargs.get("eos_token_id", 0)
        self.pad_token_id = kwargs.get("pad_token_id", 0)

        self.verbose = True

        self.csv_object = csv_object
        self.eval_train = kwargs.get("eval_train", None)
        self.eval_val = kwargs.get("eval_val", None)

        self.history = {
            "loss_train": [], 
            "loss_val": [], 
            "confidence_train": [], 
            "confidence_val": [],
            "accuracy_train": [], 
            "accuracy_val": [],
            # "f1_train": [],
            # "f1_val": []
        }
        try:
            self.load_last_session()
            print("Previous session loaded!")
        except:
            print("No previous session found!")

    def load_last_session(self):
        with open(self.model_dir / "history.pickle", 'rb') as handle:
            self.history = pickle.load(handle)
        self.load_model()

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
            **kwargs
        ) -> None:
        """
        Fit the model on the training set.
        
        Args:
            train_set: Dataset, training dataset
            val_set: Dataset, validation dataset
            epochs: int, number of epochs
            batch_size: int, batch size
            patience: int, patience for early stopping
            min_delta: float, minimum delta for early stopping
            verbose: bool, verbosity
            train_kwargs: Dict, kwargs for DataLoader of training set
            val_kwargs: Dict, kwargs for DataLoader of validation set
        """
        
        train_loader = DataLoader(train_set, **train_kwargs)
        val_loader = DataLoader(val_set, **val_kwargs)

        early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
        
        compute_train_loss, compute_val_loss = self._get_train_val_loops()
        print("Training type: ", self.training_type)

        get_cuda_allocation(verbose=True)
        print("Start training")
        disable = (not verbose) or not (self.rank==0 or get_device().type == "mps")
        with tqdm(range(epochs), unit="epoch", disable=disable) as tepoch:
            for epoch in tepoch:
                if self.csv_object:
                    self.csv_object.update_hyperparameters(epoch, batch_size)

                if not self.no_cuda:
                    torch.cuda.empty_cache()
                if "sampler" in train_kwargs.keys():
                    train_kwargs["sampler"].set_epoch(epoch)

                t1 = time.time()
                train_loss = compute_train_loss(train_loader)
                t2 = time.time()
                dt_train_ep = f"{t2 - t1:.2f}"

                eval_train = self.eval_train.compute()
                self.eval_train.reset()

                t1 = time.time()
                val_loss = compute_val_loss(val_loader)
                t2 = time.time()
                dt_val_ep = f"{t2 - t1:.2f}"

                eval_val = self.eval_val.compute()
                self.eval_val.reset()
                
                self.history["accuracy_train"].append(eval_train["accuracy"])
                self.history["accuracy_val"].append(eval_val["accuracy"])

                self.history["confidence_train"].append(eval_train["confidence"])
                self.history["confidence_val"].append(eval_val["confidence"])

                self.lr_scheduler.step()

                if self.rank == 0 or get_device().type == "mps":
                    losses = {
                        "train": train_loss,
                        "test": val_loss
                    }
                    if self.csv_object:
                        self.csv_object(
                            losses, 
                            train_metrics=eval_train,
                            val_metrics=eval_val
                        )

                self.history["loss_train"].append(train_loss)
                self.history["loss_val"].append(val_loss)

                if self.rank == 0 or get_device().type == "mps":
                    tepoch.set_postfix(
                        loss = self.history["loss_train"][-1],
                        loss_val = self.history["loss_val"][-1],
                        accuracy_train = self.history["accuracy_train"][-1],
                        accuracy_val = self.history["accuracy_val"][-1],
                        confidence_train = self.history["confidence_train"][-1],
                        confidence_val = self.history["confidence_val"][-1],
                        dt_train_epoch = dt_train_ep,
                        dt_val_epoch = dt_val_ep,
                    )

                early_stopping(val_loss)
                if self.rank == 0 and early_stopping.save_model:
                    if self.world_size > 1:
                        self.save_modelcheckpoint(best=True)
                    else:
                        self.save_model(best=True)
                if self.no_cuda:
                    torch.cuda.empty_cache()
                if early_stopping.early_stop:
                    print(f"Early stopping at epoch {epoch}, patience is {early_stopping.patience}") if self.rank == 0 else None
                    break
                
                if epoch % 20 == 0:    
                    if self.world_size > 1:
                        self.save_modelcheckpoint()
                    else:
                        self.save_model()
                        
                    with open(self.model_dir / "history.pickle", 'wb') as handle:
                        pickle.dump(self.history, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def _get_train_val_loops(self):
        if self.model_type == "hgface":
            if self.training_type == TrainingType.BATCH:
                return self._train_hgface_epoch, self._val_hgface_epoch
            else:
                Warning("Only batch training is supported for hgface models. Continuing with batch training.")
                return self._train_hgface_epoch, self._val_hgface_epoch
        else:
            if self.training_type == TrainingType.BATCH:
                return self._train_epoch, self._val_epoch
            elif self.training_type == TrainingType.ITER:
                return self._iter_train_epoch, self._iter_val_epoch
            else:
                raise NotImplementedError(f"Training type {self.training_type} not supported.")
            

    def _train_epoch(self, train_loader, accumulation_steps: int = 1) -> float:
        self.model.train()
        ddp_loss = torch.zeros(2).to(self.rank)
        vocab_size = self.model.vocab_size

        for i, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(self.rank)
            labels = batch["labels"].to(self.rank)
            start_pos = batch["start_positions"].min().to(self.rank)
            mask = batch["mask"].to(self.rank)
            
            del batch
            assert not torch.isnan(input_ids).any(), "NaN found in sources!"

            with torch.autocast(device_type="cuda", dtype=torch.float32):
                output = self.model(input_ids, mask=mask)[:, start_pos:]
                del mask
                loss = self.loss_fn(output.reshape(-1, vocab_size), labels.reshape(-1))
                loss = loss
                
            loss.backward()

            if self.eval_train:
                self.eval_train.update(refs=labels, preds=output)
                if i % 10 == 0:
                    self.eval_val.compute()

            ddp_loss[0] += loss.item()
            ddp_loss[1] += labels.numel()

            del input_ids, labels, output, loss

            if (i + 1) % accumulation_steps == 0:
                if self.csv_object:
                    self.csv_object.update_model()
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.model.zero_grad()
            
        if not self.no_cuda and self.world_size > 1:
            dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
            torch.cuda.empty_cache()
        train_loss = ddp_loss[0] / ddp_loss[1]

        return float(train_loss.cpu().numpy() / accumulation_steps)
    

    def _val_epoch(self, val_loader: DataLoader, mode: str ="greedy") -> float:
        self.model.eval()
        ddp_loss = torch.zeros(2, device=self.rank)
        
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                input_ids = batch["input_ids"].to(self.rank)
                labels = batch["labels"].to(self.rank)
                mask = batch["mask"].to(self.rank)
                start_pos = batch["start_positions"].min().to(self.rank)

                del batch
                assert not torch.isnan(input_ids).any(), "NaN found in sources!"
                vocab_size = self.model.vocab_size

                with torch.autocast(device_type=f"cuda", dtype=torch.float32):
                    output = self.model(input_ids, mask=mask)[:,start_pos:]
                    logits = output.reshape(-1, vocab_size)

                    loss = self.loss_fn(logits, labels.reshape(-1))
                
                if self.eval_val:
                    self.eval_val.update(refs=labels, preds=output)
                    if i % 10 == 0:
                        self.eval_val.compute()

                ddp_loss[0] += loss.item()
                ddp_loss[1] += labels.numel()

                del labels, output, logits, input_ids, mask
                
        if not self.no_cuda and self.world_size > 1:
            dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
            torch.cuda.empty_cache()

        val_loss = ddp_loss[0] / ddp_loss[1]

        return float(val_loss.cpu().numpy())
    
    def _iter_train_epoch(self, train_loader, accumulation_steps: int = 1) -> float:
        self.model.train()
        ddp_loss = torch.zeros(2).to(self.rank)
        vocab_size = self.model.vocab_size

        for i, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(self.rank)
            labels = batch["labels"].to(self.rank)
            start_pos = batch["start_positions"].min().to(self.rank)
            end_positions = batch["end_positions"].to(self.rank)
            mask = batch["mask"].to(self.rank)

            del batch
            assert not torch.isnan(input_ids).any(), "NaN found in sources!"

            logits = torch.zeros((labels.size(0), labels.size(1), vocab_size), device=self.rank)

            for idx in range((end_positions.max() - start_pos - 1)):
                # with torch.autocast(device_type="cuda", dtype=torch.float32):
                output = self.model(input_ids[:, :start_pos + idx], mask=mask[:, :start_pos + idx, :start_pos + idx])[:, -1]
                loss = self.loss_fn(output.reshape(-1, vocab_size), labels[:,idx].reshape(-1))
                loss = loss
                logits[:, idx] = output
                loss.backward()
                
                ddp_loss[0] += loss.item() 
                del loss

            ddp_loss[1] += labels.numel()
            if self.eval_train:
                self.eval_train.update(refs=labels, preds=logits)
                if i % 10 == 0:
                    self.eval_val.compute()

            del mask, input_ids, output, labels, logits

            if (i + 1) % accumulation_steps == 0:
                if self.csv_object:
                    self.csv_object.update_model()
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.model.zero_grad()

        ddp_loss[0] *= accumulation_steps 

        if not self.no_cuda and self.world_size > 1:
            dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
            torch.cuda.empty_cache()

        train_loss = ddp_loss[0] / ddp_loss[1]

        return float(train_loss.cpu().numpy())


    def _iter_val_epoch(self, val_loader: DataLoader, mode: str ="greedy") -> float:
        self.model.eval()
        ddp_loss = torch.zeros(2, device=self.rank)
        vocab_size = self.model.vocab_size
        
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                input_ids = batch["input_ids"].to(self.rank)
                labels = batch["labels"].to(self.rank)
                start_pos = batch["start_positions"].min().to(self.rank)
                end_positions = batch["end_positions"].to(self.rank)
                mask = batch["mask"].to(self.rank)

                del batch
                assert not torch.isnan(input_ids).any(), "NaN found in sources!"

                logits = torch.zeros((labels.size(0), labels.size(1), vocab_size), device=self.rank)
                for idx in range(end_positions.max() - start_pos - 1):
                    with torch.autocast(device_type="cuda", dtype=torch.float32):
                        output = self.model(input_ids[:, :start_pos + idx], mask=mask[:,:start_pos + idx, :start_pos + idx])[:, -1]
                        logits[:, idx] = output
                        
                        loss = self.loss_fn(output.reshape(-1, vocab_size), labels[:, idx].reshape(-1))
                        
                    ddp_loss[0] += loss.item() 
                    
                    del output, loss

                if self.eval_val:
                    self.eval_val.update(refs=labels, preds=logits)
                    if i % 10 == 0:
                        self.eval_val.compute()
                
                ddp_loss[1] += labels.numel()

                del input_ids, labels, logits, mask
                
        if not self.no_cuda and self.world_size > 1:
            dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
            torch.cuda.empty_cache()
        val_loss = ddp_loss[0] / ddp_loss[1]

        return float(val_loss.cpu().numpy())
    
    
    
    @DeprecationWarning
    def _train_epoch_last(self, train_loader, accumulation_steps: int = 1) -> float:
        self.model.train()
        ddp_loss = torch.zeros(2).to(self.rank)
        start_pos = 0
        vocab_size = self.model.vocab_size

        for i, seq in enumerate(train_loader):
            assert not torch.isnan(seq).any(), "NaN found in sources!"
            seq = seq.to(self.rank)
            labels = seq[:,start_pos:].reshape(-1)

            with torch.autocast(device_type=f"cuda", dtype=torch.float32):
                output = self.model(seq, start_pos)[:,start_pos:].reshape(-1, vocab_size)
                loss = self.loss_fn(output, labels)

            self.history["accuracy_train"].append(compute_accuracy(labels, output))
            # self.history["f1_train"].append(compute_f1(labels, output))

            loss.backward()
            _sz = labels.size(0)
            del seq, labels, output

            if self.csv_object:
                self.csv_object.update_model()
            if (i + 1) % accumulation_steps == 0:
                if self.rank == 0 or True:
                    pass
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.model.zero_grad()
            # self.optimizer.step()
            ddp_loss[0] += loss.item()
            ddp_loss[1] += _sz
        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
        torch.cuda.empty_cache()
        train_loss = ddp_loss[0] / ddp_loss[1]

        return float(train_loss.cpu().numpy())
    
    def _train_hgface_epoch(self, train_loader, accumulation_steps: int = 1) -> float:
        self.model.train()
        ddp_loss = torch.zeros(2).to(self.rank)
        vocab_size = self.model.vocab_size

        for idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(self.rank)
            labels = batch["labels"].to(self.rank)
            start_pos = batch["start_positions"].min().to(self.rank)
            end_positions = batch["end_positions"].to(self.rank)
            attention_mask = batch["attention_mask"].to(self.rank)

            del batch
            assert not torch.isnan(input_ids).any(), "NaN found in sources!"

            with torch.autocast(device_type=f"cuda", dtype=torch.float32):
                output = self.model(input_ids, attention_mask=attention_mask, start_positions=start_pos, end_positions=end_positions)
                loss = output.loss

            logits = output.logits
            # self.history["f1_train"].append(compute_f1(labels, output))

            loss.backward()
            _sz = labels.size(0)
            del seq, labels, output

            if self.csv_object:
                self.csv_object.update_model()
            if (i + 1) % accumulation_steps == 0:
                if self.rank == 0 or True:
                    pass
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.model.zero_grad()
            # self.optimizer.step()
            ddp_loss[0] += loss.item()
            ddp_loss[1] += _sz
        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
        torch.cuda.empty_cache()
        train_loss = ddp_loss[0] / ddp_loss[1]

        return float(train_loss.cpu().numpy())

    @PendingDeprecationWarning
    def _val_hgface_epoch(self, val_loader: DataLoader) -> float:
        self.model.eval()
        ddp_loss = torch.zeros(2, device=self.rank)
        start_pos = 0
        with torch.no_grad():
            for seq in val_loader:
                assert not torch.isnan(seq).any(), "NaN found in sources!"
                seq = seq.to(self.rank)
                vocab_size = self.model.vocab_size
                labels = seq[:,start_pos:].reshape(-1)

                with torch.autocast(device_type=f"cuda", dtype=torch.float32):
                    output = self.model(seq, start_pos)[:,start_pos:]
                    logits = output.reshape(-1, vocab_size)

                    loss = self.loss_fn(logits, labels)

                
                if self.eval_val:
                    self.eval_val.update(refs=seq, preds=output)

                ddp_loss[0] += loss.item()
                ddp_loss[1] += labels.numel()

                del labels, seq, output, logits

        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
        val_loss = ddp_loss[0] / ddp_loss[1]

        return float(val_loss.cpu().numpy())
    
    @DeprecationWarning
    def _slow_val_epoch(self, val_loader: DataLoader, mode: str ="greedy") -> float:
        self.model.eval()
        ddp_loss = torch.zeros(2, device=self.rank)
        start_pos = 0
        with torch.no_grad():
            for seq in val_loader:
                assert not torch.isnan(seq).any(), "NaN found in sources!"
                seq = seq.to(self.rank)
                
                batch_size, seq_len = seq.shape
                all_logits = torch.zeros((batch_size, seq_len - start_pos, self.model.vocab_size), device=self.rank)

                for i in range(start_pos, seq_len): 
                    logits = self.model(seq[:, :i], i)
                    labels = seq[:, i].reshape(-1)
                    logits = logits[:, -1, :] 
                    all_logits[:, i - start_pos, :] = logits
                    vocab_size = logits.size(-1)

                    loss = self.loss_fn(logits.view(-1, vocab_size), labels)
                    del logits
                    
                    ddp_loss[0] += loss.item()
                    ddp_loss[1] += labels.numel()

                    # if mode == "greedy":
                    #     next_token = logits.argmax(dim=-1)
                    # else:
                    #     probs = torch.softmax(logits, dim=-1)
                    #     next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
                    #     del probs

                # self.history["f1_val"].append(compute_f1(labels, all_logits.view(-1, vocab_size)))

                if self.eval_val:
                    self.eval_val.update(refs=seq, preds=all_logits)

                del all_logits, seq

        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
        val_loss = ddp_loss[0] / ddp_loss[1]

        return float(val_loss.cpu().numpy())

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
            loss = self.loss_fn(pred.view(-1, pred.size(-1)), tgt.view(-1))
        return loss.item()
    
    def save_model(self, best: bool = False):
        path = "model.pth" if not best else "best_model.pth"
        path = self.model_dir.joinpath(path)
        print(f"Try to save model at {path}")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            # 'optimizer_state_dict': self.optimizer.state_dict(),
            # 'loss': self.loss_fn
        }, path)
        # self.save_grads()
    
    def load_model(self, best: bool = False):
        path = "model.pth" if not best else "best_model.pth"
        path = self.model_dir.joinpath(path)
        print("Try to load model from:", path)
        try:
            checkpoint = torch.load(path, weights_only=False)
        except:
            print(f"Load model on GPU failed, trying to load on CPU")
            checkpoint = torch.load(path, weights_only=False, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # self.loss_fn.load_state_dict(checkpoint['loss'])
        print(f"Model loaded from {path}!")
        return self.model
    
    def save_modelcheckpoint(self, best: bool = False):
        path = "model.pth" if not best else "best_model.pth"
        path = self.model_dir.joinpath(path)
        print(f"Try to save model checkpoint at {path}")
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
            model_cpu_state = self.model.state_dict()
            # opt_cpu_state = self.optimizer.state_dict()
            # loss_cpu_state = self.loss_fn

        if self.rank == 0:
        # save_name = file_save_name + "-" + time_of_run + "-" + currEpoch
            torch.save({
                'model_state_dict': model_cpu_state, #self.model.state_dict(),
                # 'optimizer_state_dict': opt_cpu_state, #self.optimizer.state_dict(),
                # 'loss': self.loss_fn
            }, path)
        print("Model checkpoint saved!")

    @DeprecationWarning
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

