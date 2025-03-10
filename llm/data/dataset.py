from llm.data.tokenizer import Tokenizer, CONTROL_TOKENS
import datasets

import torch
from torch.utils.data import DataLoader, Dataset

from typing import Any, Tuple, Union, List


class FinQADataset(Dataset):
    def __init__(
            self, 
            data: Any, 
            tokenizer: Tokenizer = None,
            max_length: int = 1024,
            max_q_length: Union[int, None] = None,
            max_a_length: Union[int, None] = None,
            short_answer: bool = True,
        ) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_q_len = max_q_length - max_a_length if max_q_length else max_length - max_a_length
        self.max_a_len = max_a_length if max_a_length else max_length
        
        self.short_answer = short_answer

    def __len__(self):  
        return len(self.data)

    def __getitem__(self, idx: int):
        data = self.data[idx]
        # PREPARE QUESTION
        pre_text = data["pre_text"]
        while type(pre_text) == list:
            pre_text = ("").join(pre_text)
        post_text = data["post_text"]
        while type(post_text) == list:
            post_text = ("").join(post_text)
        table = str(data["table"])
        question = data["question"]

        # PREPARE ANSWER
        if self.short_answer:
            answer = data["answer"] or data["final_result"] 
        else:
            answer = data["gold_inds"]
            while type(answer) == list:
                answer = ("").join(answer)
            program = data["program_re"]

        # WRAP THEM
        def format_table(table_str):
            table_str = table_str.replace("[[", "[\n    [")
            table_str = table_str.replace("], [", "],\n    [")
            table_str = table_str.replace("]]", "]\n]")
            return table_str

        # formatted_table = format_table(table)

        question = f"""{CONTROL_TOKENS.start_of_context}{pre_text}{CONTROL_TOKENS.end_of_context}{CONTROL_TOKENS.start_of_table}{table}{CONTROL_TOKENS.end_of_table}{CONTROL_TOKENS.start_of_description}{post_text}{CONTROL_TOKENS.end_of_description}{CONTROL_TOKENS.start_of_question}{question}{CONTROL_TOKENS.end_of_question}"""
        answer = f"{CONTROL_TOKENS.start_of_text}{answer}" 
        if not self.short_answer:
            answer += f"{CONTROL_TOKENS.start_of_program}{program}{CONTROL_TOKENS.end_of_program}"
        answer += f"{CONTROL_TOKENS.end_of_text}"

        # TOKENIZE THEM
        input_ids = self.tokenizer(question, padding='none', return_tensors=True)
        
        start_pos = torch.tensor(len(input_ids)).cpu()
        labels = self.tokenizer(answer, padding='none', return_tensors=True)

        seq = torch.cat([input_ids, labels])
        seq = self.pad_sequence(seq).squeeze(0).cpu()
        return seq, start_pos
    
    def pad_sequence(self, token_ids: torch.Tensor) -> torch.Tensor:
        if len(token_ids) < self.max_length:
            pad_tokens = torch.tensor([self.tokenizer.pad_token_id] * (self.max_length - len(token_ids)), dtype=torch.long)
            return torch.cat([token_ids, pad_tokens])
        else:
            return token_ids[-self.max_length:]
  
def get_data(
        tokenizer: Tokenizer,
        max_length: int = 1024,
        max_q_length: Union[int, None] = None,
        max_a_length: Union[int, None] = None,
        short_answer: bool = True,
        **kwargs
    ) -> Tuple[FinQADataset, FinQADataset, FinQADataset]:
    dataset = datasets.load_dataset("ibm-research/finqa", "en")

    train = dataset["train"]
    test = dataset["test"]
    val = dataset["validation"]

    params = {
        "tokenizer": tokenizer,
        "max_length": max_length,
        "max_q_length": max_q_length,
        "max_a_length": max_a_length,
        "short_answer": short_answer,
    }
    train = FinQADataset(train, **params)
    test = FinQADataset(test, **params)
    val = FinQADataset(val, **params)
    
    return train, test, val