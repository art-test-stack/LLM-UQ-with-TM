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
            max_a_length: Union[int, None] = 8,
            short_answer: bool = True,
            hint: bool = False,
        ) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_q_len = max_length - max_a_length
        self.max_a_len = max_a_length
        
        if hint and not short_answer:
            print("Warning: Hint is only available for short answer. Ignoring the hint parameter.")

        self.hint = hint and not (hint and not short_answer)
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
            answer = data["final_result"] if data["answer"] == "" else min(data["answer"], data["final_result"], key=len)
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

        question = f"""{CONTROL_TOKENS.start_of_context}{pre_text}{CONTROL_TOKENS.end_of_context}{CONTROL_TOKENS.start_of_table}{table}{CONTROL_TOKENS.end_of_table}{CONTROL_TOKENS.start_of_description}{post_text}{CONTROL_TOKENS.end_of_description}"""
        if self.hint and self.short_answer:
            hint = data['gold_inds']
            question += f"{CONTROL_TOKENS.start_of_hint}{hint}{CONTROL_TOKENS.end_of_hint}"
        question += f"{CONTROL_TOKENS.start_of_question}{question}{CONTROL_TOKENS.end_of_question}"

        answer = f"{CONTROL_TOKENS.start_of_text}{answer}" 
        if not self.short_answer:
            answer += f"{CONTROL_TOKENS.start_of_program}{program}{CONTROL_TOKENS.end_of_program}"
        answer += f"{CONTROL_TOKENS.end_of_text}"

        # TOKENIZE THEM
        input_ids = self.tokenizer(question, padding='none', return_tensors=True)
        input_ids = pad_sequence(input_ids, self.max_q_len, self.tokenizer.pad_token_id)

        labels = self.tokenizer(answer, padding='none', return_tensors=True)
        labels = pad_sequence(labels, self.max_a_len, self.tokenizer.pad_token_id)

        seq = torch.cat([input_ids, labels])
        seq = seq.squeeze(0).cpu()
        return seq

def pad_sequence(token_ids: torch.Tensor, max_length: int, pad_token_id: int) -> torch.Tensor:
    if len(token_ids) < max_length:
        pad_tokens = torch.tensor([pad_token_id] * (max_length - len(token_ids)), dtype=torch.long)
        return torch.cat([token_ids, pad_tokens])
    else:
        return token_ids[-max_length:]
  
def get_data(
        tokenizer: Tokenizer,
        max_length: int = 1024,
        max_a_length: Union[int, None] = None,
        short_answer: bool = True,
        hint: bool = False,
        **kwargs
    ) -> Tuple[FinQADataset, FinQADataset, FinQADataset]:
    dataset = datasets.load_dataset("ibm-research/finqa", "en")

    train = dataset["train"]
    test = dataset["test"]
    val = dataset["validation"]

    params = {
        "tokenizer": tokenizer,
        "max_length": max_length,
        "max_a_length": max_a_length,
        "short_answer": short_answer,
        "hint": hint,
    }
    train = FinQADataset(train, **params)
    test = FinQADataset(test, **params)
    val = FinQADataset(val, **params)
    
    return train, test, val