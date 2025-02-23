from data.tokenizer import Tokenizer, CONTROL_TOKENS
import datasets

import torch
from torch.utils.data import DataLoader, Dataset

from typing import Any, Tuple

class FinQADataset(Dataset):
    def __init__(self, data: Any, tokenizer: Tokenizer):
        questions = [ tokenizer.encode(question) for question in data["question"]]
        answers = [ tokenizer.encode(answer) for answer in data["answer"]]
        
        max_len = max([len(q) for q in questions] + [len(a) for a in answers])
        
        self.questions = torch.Tensor([ q + [tokenizer.special_tokens[CONTROL_TOKENS.padding]] * (max_len - len(q)) for q in questions ])
        self.answers = torch.Tensor([ a + [tokenizer.special_tokens[CONTROL_TOKENS.padding]] * (max_len - len(a)) for a in answers ])
        self.max_content = max_len

    def __len__(self):  
        return len(self.questions)

    def __getitem__(self, idx: int):
        return self.questions[idx].long(), self.answers[idx].long() #.float()
    
def get_data(tokenizer: Tokenizer) -> Tuple[FinQADataset, FinQADataset, FinQADataset]:
    dataset = datasets.load_dataset("ibm-research/finqa", "en")

    train = dataset["train"]
    test = dataset["test"]
    val = dataset["validation"]

    train = FinQADataset(train, tokenizer)
    test = FinQADataset(test, tokenizer)
    val = FinQADataset(val, tokenizer)
    
    return train, test, val