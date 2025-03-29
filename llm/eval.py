import evaluate

import torch
from typing import Callable, List

from enum import Enum
from typing import Dict, Union

possible_metrics = [
    # "bleu",
    # "rouge",
    # "accuracy",
    # "f1",
    # "precision",
    # "recall",
]

class SampleType(Enum):
    TENSOR = "tensor"
    STRING = "string"



class Evaluate:
    def __init__(
            self, 
            padding_id: int = None,
            tokenizer: Callable = None, 
            metrics: List[str] = possible_metrics, 
            rank: int = 0
        ):
        assert tokenizer or isinstance(padding_id, int), "Tokenizer or padding_id is required"
        self.tokenizer = tokenizer
        self.padding_id = padding_id if not tokenizer else tokenizer.pad_token_id
        self.best = None
        self.best_metric = None
        self.rank = rank
        self.running_res = None
        self.has_str_values = False
        self.running_preds = []
        self.running_refs = []
        self.running_numel = 0
        self.result_keys = []
        self.init_metrics(metrics)
        self._get_results_format()
        self.reset()
    
    def init_metrics(self, metrics: List[str]):
        # HuggingFace metrics
        self.metrics = {}
        self._init_custom_metrics()
        if self.tokenizer:
            for metric in metrics:
                m = evaluate.load(metric)
                m.tokenizer = self.tokenizer
                setattr(self, f"_{metric}", m)
        self.metrics |= { metric: SampleType.STRING for metric in metrics }
        self.has_str_values = SampleType.STRING in list(self.metrics.values())

    def _init_custom_metrics(self):
        self._accuracy = Accuracy(padding_id=self.padding_id, rank=self.rank)
        self.metrics["accuracy"] = SampleType.TENSOR

        self._confidence = ConfidenceScore(padding_id=self.padding_id, rank=self.rank)
        self.metrics["confidence"] = SampleType.TENSOR

        self._perplexity = Perplexity(padding_id=self.padding_id, rank=self.rank)
        self.metrics["perplexity"] = SampleType.TENSOR
    
    @torch.inference_mode()
    def update(self, refs: torch.Tensor, preds: torch.Tensor):
        """
        Update the references and predictions for the evaluation task
        
        Args:
            refs: torch.Tensor
                The reference(s) for the evaluation task
            preds: torch.Tensor
                The prediction(s) for the evaluation task
        """
        preds = preds.detach().half()
        refs = refs.detach().long()

        self.running_numel += preds.numel()
        self.running_preds.append(preds)
        self.running_refs.append(refs)

        if self.running_numel > 1e6:  # Adjust the limit as needed
            self.compute()
            self.reset_tensor()

    @torch.inference_mode()
    def compute(self):
        results = {}
        if len(self.running_refs) == 0:
            return self.results(results)
        elif len(self.running_refs) > 1:
            self.running_refs = torch.cat(self.running_refs).to(device=self.rank)
            self.running_preds = torch.cat(self.running_preds).to(device=self.rank)
        else:
            self.running_refs = self.running_refs[0]
            self.running_preds = self.running_preds[0]
            
        if self.has_str_values:
            refs_decoded = self.tokenizer.decode(self.running_refs)
            preds_decoded = self.tokenizer.decode(self.running_preds.argmax(dim=-1))

        for metric, stype in self.metrics.items():
            if stype == SampleType.STRING:
                results |= getattr(self, f"_{metric}").compute(references=refs_decoded, predictions=preds_decoded)
            elif stype == SampleType.TENSOR:
                results[metric] = getattr(self, f"_{metric}").compute(references=self.running_refs, predictions=self.running_preds)
            else:
                raise ValueError(f"Invalid sample type: {stype}")

        res = results.copy()
        for key, value in res.items():
            if isinstance(value, list):
                for k in range(len(value)):
                    results[f"{key}_{k}"] = float(value[k])
                del results[key]
        if self.running_res is not None:
            for key, value in results.items():
                self.running_res[key].append(value)
                
        self.reset_tensor()
        return self.results(results)
    
    def results(self, results = {}):
        if self.running_res:
            return { m: sum(v) / len(v) if len(v) > 1 else None for m, v in self.running_res.items() }
        return results

    def __call__(self, refs, preds, tokenized = True):
        self.update(refs, preds, tokenized=tokenized)
        return self.compute()
    
    def reset_tensor(self):
        del self.running_preds, self.running_refs
        torch.cuda.empty_cache()
        self.running_preds, self.running_refs = [], []
        self.running_numel = 0

    def reset(self):
        self.reset_tensor()
        self.init_running_res()

    def init_running_res(self):
        self.running_res = { key: [] for key in self.result_keys }

    def _get_results_format(self):
        self.running_refs = [torch.randint(0, 100, (1, 10), device=self.rank)]
        self.running_preds = [torch.rand(1, 10, 100, device=self.rank)]
        # self.update(self.running_refs, self.running_preds)
        results = self.compute()
        self.result_keys = []
        for key, value in results.items():
            if isinstance(value, list):
                for k in range(len(value)):
                    self.result_keys.append(f"{key}_{k}")
            else:
                self.result_keys.append(key)
        self.reset()


class Accuracy:
    """
    Compute accuracy
    
    Args:
        padding_id: int
            The padding id to ignore while computing accuracy

    """
    def __init__(self, padding_id: int = 0, rank: int = 0):
        self.padding_id = padding_id 
        self.rank = rank
    
    @torch.inference_mode()
    def compute(
            self, 
            references: torch.Tensor, 
            predictions: torch.Tensor, 
        ) -> torch.Tensor:
        """
        Compute accuracy

        Args:
            references (torch.Tensor):
                The reference tensor. Shape: (batch_size, seq_len)
            predictions (torch.Tensor):
                The prediction tensor. Shape: (batch_size, seq_len)
        Returns:
            float: Accuracy in the range [0, 1]
        """
        references = references.to(device=self.rank)
        predictions = predictions.to(device=self.rank)
        padding_id = self.padding_id

        references = references.view(-1)
        predictions = predictions.argmax(dim=-1).view(-1)

        mask = references != padding_id
        references = references[mask]
        predictions = predictions[mask]
        acc = (predictions == references).to(dtype = torch.float32).mean()

        return acc.item()
    

class ConfidenceScore:
    def __init__(self, padding_id: int = 0, rank: int = 0):
        self.padding_id = padding_id 
        self.rank = rank

    @torch.inference_mode()
    def compute(
        self,
        references: torch.Tensor,
        predictions: torch.Tensor,
        ) -> torch.Tensor:
        """
        Calculate confidence score from logits and target labels, ignoring padding tokens.

        Args:
        - logits (torch.Tensor): Logits output from the model (batch_size, seq_length, vocab_size).
        - target (torch.Tensor): Ground truth labels (batch_size, seq_length).
        - pad_token_id (int): ID of the padding token to ignore.

        Returns:
        - confidence (float): The confidence score.
        """
        probs = torch.nn.functional.softmax(predictions, dim=-1)

        target_probs = probs.gather(dim=-1, index=references.unsqueeze(-1)).squeeze(-1)

        valid_mask = references != self.padding_id

        target_probs = target_probs[valid_mask]

        confidence = target_probs.mean()
        return confidence.item()
    

class Perplexity:
    def __init__(self, padding_id: int = 0, rank: int = 0):
        self.padding_id = padding_id 
        self.rank = rank

    @torch.inference_mode()
    def compute(
        self,
        references: torch.Tensor,
        predictions: torch.Tensor,
        ) -> torch.Tensor:
        """
        Calculate perplexity from logits and target labels, ignoring padding tokens.

        Args:
            logits (torch.Tensor): Logits output from the model (batch_size, seq_length, vocab_size).
            target (torch.Tensor): Ground truth labels (batch_size, seq_length).
            pad_token_id (int): ID of the padding token to ignore.

        Returns:
            perplexity (float): The perplexity score.
        """
        log_probs = torch.nn.functional.log_softmax(predictions, dim=-1)

        target_log_probs = log_probs.gather(dim=-1, index=references.unsqueeze(-1)).squeeze(-1)
        mask = references == self.padding_id
        target_log_probs[mask] = 0
        
        perplexity = torch.exp(-target_log_probs.mean(dim=-1))  
        return perplexity.mean().item()
