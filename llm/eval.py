import evaluate

from typing import Callable, List


possible_metrics = [
    "bleu",
    "rouge",
    # "accuracy",
    # "f1",
    # "precision",
    # "recall",

]

class EvalTask:
    def __init__(self, tokenizer: Callable = None, metrics: List[str] = None):
        assert tokenizer, "Tokenizer is required"
        self.tokenizer = tokenizer
        self.best = None
        self.best_metric = None
        self.clean_up()
        self.init_metrics(metrics or possible_metrics)
        self._get_results_format()
    
    def init_metrics(self, metrics: List[str]):
        for metric in metrics:
            m = evaluate.load(metric)
            m.tokenizer = self.tokenizer
            setattr(self, f"_{metric}", m)
        self.metrics = metrics
    
    def update(self, refs, preds, tokenized=True):
        if isinstance(preds, str):
            preds = [preds]
        if isinstance(refs, str):
            refs = [refs]
        if tokenized:
            preds = self.tokenizer.decode(preds)
            refs = self.tokenizer.decode(refs)
        self.preds.extend(preds)
        self.refs.extend(refs)

    def compute(self, refs=None, preds=None, clean_up=True, tokenized=True):
        results = {}
        if preds and refs:
            self.update(refs=refs, preds=preds, tokenized=tokenized)
        for metric in self.metrics:
            results |= getattr(self, f"_{metric}").compute(references=self.refs, predictions=self.preds, )
        if clean_up:
            self.clean_up()

        res = results.copy()
        for key, value in res.items():
            if isinstance(value, list):
                for k in range(len(value)):
                    results[f"{key}_{k}"] = float(value[k])
                del results[key]
            
        return results
    
    def __call__(self, refs, preds, tokenized = True):
        self.update(refs, preds, tokenized=tokenized)
        return self.compute(clean_up=False)
        
    def clean_up(self):
        self.preds = []
        self.refs = []

    def _get_results_format(self):
        refs = ["hello"]
        preds = ["hi"]
        results = self.compute(refs=refs, preds=preds, clean_up=True, tokenized=False)
        self.result_keys = []
        for key, value in results.items():
            if isinstance(value, list):
                for k in range(len(value)):
                    self.result_keys.append(f"{key}_{k}")
            else:
                self.result_keys.append(key)
