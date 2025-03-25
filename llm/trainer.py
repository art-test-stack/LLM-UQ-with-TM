from tm_data.preprocessing import InputCSV
from llm.confidence import ConfidenceScore
from llm.utils import EarlyStopping
from llm.eval import compute_accuracy, compute_f1
from utils import get_cuda_allocation


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

class Trainer:
    """
    Trainer class for training the model.
    
    Args:
        model: nn.Module, model to train
        optimizer: optim.Optimizer, optimizer
        loss_fn: nn.Module, loss function
        csv_object: InputCSV, object to save the training history
        rank: torch.device, rank of the process
        world_size: int, number of processes
        eval_task: Callable, evaluation task
        eval_confidence: Callable, evaluation confidence
        name: str, name of the model
        model_dir: str, directory to save the model
        lr_scheduler: optim.lr_scheduler, learning rate scheduler
        bos_token_id: int, start of answer token
        eos_token_id: int, end of answer token
        pad_token_id: int, padding token
        start_pos: int, start position of the sequence
    """
    def __init__(
            self,
            model: Union[Callable,FSDP,nn.Module],
            optimizer: optim.Optimizer,
            loss_fn: nn.Module,
            rank: torch.device,
            world_size: int,
            csv_object: InputCSV = None,
            eval_task: Callable = None,
            eval_confidence: Callable = None,
            name: str = "model",
            model_dir: str = "models/",
            lr_scheduler: optim.lr_scheduler = None,
            bos_token_id: int = 0,
            eos_token_id: int = 0,
            pad_token_id: int = 0,
            start_pos: int = 0,
            **kwargs
        ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler or torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        self.loss_fn = loss_fn
        self.rank = rank
        self.world_size = world_size
        self.name = name
        self.model_dir = Path(model_dir).joinpath(self.name)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        assert not bos_token_id == eos_token_id, "Start of answer and end of answer tokens should be different"
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.start_pos = start_pos
        self.verbose = True

        self.csv_object = csv_object
        self.eval_task = eval_task
        self.eval_conf = eval_confidence or ConfidenceScore(device=self.rank)

        self.history = {
            "train_loss": [], 
            "val_loss": [], 
            "confidence": [], 
            "accuracy_train": [], 
            "accuracy_val": [],
            # "f1_train": [],
            # "f1_val": []
        }

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

        get_cuda_allocation(verbose=True)
        print("Start training")
        with tqdm(range(epochs), unit="epoch", disable=not verbose or not self.rank==0) as tepoch:
            for epoch in tepoch:
                # get_cuda_allocation(verbose=True)
                # print(torch.cuda.memory_summary(device=0, abbreviated=False))
                if self.csv_object:
                    self.csv_object.update_hyperparameters(epoch, batch_size)

                torch.cuda.empty_cache()
                if "sampler" in train_kwargs.keys():
                    train_kwargs["sampler"].set_epoch(epoch)
                import time

                t1 = time.time()
                train_loss = self._train_epoch(train_loader)
                t2 = time.time()
                dt_train_ep = f"{t2 - t1:.2f}"

                t1 = time.time()
                val_loss = self._val_epoch(val_loader)
                t2 = time.time()
                dt_val_ep = f"{t2 - t1:.2f}"

                self.lr_scheduler.step()
                
                confidence_score = math.exp(self.eval_conf.get())
                if self.rank == 0:
                    losses = {
                        "train": train_loss,
                        "test": val_loss
                    }
                    if self.csv_object:
                        self.csv_object(losses, self.eval_task.compute(), confidence_score)

                self.history["train_loss"].append(train_loss)
                self.history["val_loss"].append(val_loss)
                self.history["confidence"].append(confidence_score)

                if self.rank == 0:
                    tepoch.set_postfix(
                        loss = self.history["train_loss"][-1],
                        val_loss = self.history["val_loss"][-1],
                        confidence = confidence_score,
                        accuracy_train = self.history["accuracy_train"][-1],
                        accuracy_val = self.history["accuracy_val"][-1],
                        dt_train_epoch = dt_train_ep,
                        dt_val_epoch = dt_val_ep,
                    )

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
                
        # if self.rank == 0:
        #     self.save_model()
                if epoch % 20 == 0:    
                    if self.world_size > 1:
                        self.save_modelcheckpoint()
                    else:
                        self.save_model()
                        
                    with open(self.model_dir / "history.pickle", 'wb') as handle:
                        pickle.dump(self.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _train_epoch(self, train_loader, accumulation_steps: int = 1) -> float:
        self.model.train()
        ddp_loss = torch.zeros(2).to(self.rank)
        start_pos = self.start_pos
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
    
    # def _val_epoch(self, val_loader: DataLoader, mode: str ="greedy") -> float:
    #     self.model.eval()
    #     ddp_loss = torch.zeros(2, device=self.rank)
    #     start_pos = self.start_pos
    #     with torch.no_grad():
    #         for seq in val_loader:
    #             assert not torch.isnan(seq).any(), "NaN found in sources!"
    #             seq = seq.to(self.rank)
                
    #             batch_size, seq_len = seq.shape
    #             all_logits = torch.zeros((batch_size, seq_len - start_pos, self.model.vocab_size), device=self.rank)

    #             for i in range(start_pos, seq_len): 
    #                 logits = self.model(seq[:, :i], i)
    #                 labels = seq[:, i].reshape(-1)
    #                 logits = logits[:, -1, :] 
    #                 all_logits[:, i - start_pos, :] = logits
    #                 vocab_size = logits.size(-1)

    #                 loss = self.loss_fn(logits.view(-1, vocab_size), labels)
    #                 del logits
                    
    #                 ddp_loss[0] += loss.item()
    #                 ddp_loss[1] += labels.numel()

    #                 # if mode == "greedy":
    #                 #     next_token = logits.argmax(dim=-1)
    #                 # else:
    #                 #     probs = torch.softmax(logits, dim=-1)
    #                 #     next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
    #                 #     del probs

    #             self.history["accuracy_val"].append(
    #                 compute_accuracy(seq[:, start_pos:].reshape(-1), all_logits.view(-1, vocab_size)))
    #             # self.history["f1_val"].append(compute_f1(labels, all_logits.view(-1, vocab_size)))

    #             if self.eval_task:
    #                 self.eval_task.update(refs=seq, preds=all_logits.argmax(dim=-1))
    #             self.eval_conf.update(all_logits)

    #             del all_logits, seq

    #     dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    #     val_loss = ddp_loss[0] / ddp_loss[1]

    #     return float(val_loss.cpu().numpy())
    
    def _val_epoch(self, val_loader: DataLoader, mode: str ="greedy") -> float:
        self.model.eval()
        ddp_loss = torch.zeros(2, device=self.rank)
        start_pos = self.start_pos
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

                self.history["accuracy_val"].append(
                    compute_accuracy(labels, logits))
                
                if self.eval_task:
                    self.eval_task.update(refs=seq, preds=output.argmax(dim=-1))
                self.eval_conf.update(output)

                ddp_loss[0] += loss.item()
                ddp_loss[1] += labels.numel()

                del labels, seq, output, logits

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
        path = "model.pth" if not best else "best.pth"
        path = self.model_dir.joinpath(path)
        try:
            checkpoint = torch.load(self.path, weights_only=False)
        except:
            print(f"Load model on GPU failed, trying to load on CPU")
            checkpoint = torch.load(self.path, weights_only=False, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # self.loss_fn.load_state_dict(checkpoint['loss'])
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
            # loss_cpu_state = self.loss_fn

        if self.rank == 0:
        # save_name = file_save_name + "-" + time_of_run + "-" + currEpoch
            torch.save({
                'model_state_dict': model_cpu_state, #self.model.state_dict(),
                # 'optimizer_state_dict': opt_cpu_state, #self.optimizer.state_dict(),
                # 'loss': self.loss_fn
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

