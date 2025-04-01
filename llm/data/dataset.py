from llm.data.tokenizer import Tokenizer, CONTROL_TOKENS
import datasets

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

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
            easy_task: bool = False,
            pad_all_answers: bool = False
        ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_q_len = max_length - max_a_length
        self.max_a_len = max_a_length
        self.pad_all_answers = pad_all_answers
        
        if hint and not short_answer:
            print("Warning: Hint is only available for short answer. Ignoring the hint parameter.")

        self.hint = hint and not (hint and not short_answer)
        self.short_answer = short_answer
        self.easy_task = easy_task

        data = list(map(self.read_data, data))
        encodings = list(map(self.prepare_data, data))
        
        self.encodings = { key: [val[i] for val in encodings] for i, key in enumerate(["input_ids", "labels", "start_positions", "end_positions", "len_q"]) }
        # self.encodings = { key: [val[i] for val in encodings] for i, key in enumerate(["input_ids", "labels", "mask", "start_positions", "end_positions"]) }

    def __len__(self):  
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int):
        mask = self.make_mask(idx)
        res = { key: torch.tensor(val[idx]) for key, val in self.encodings.items() if not key == "len_q" }
        res["mask"] = mask
        return res
    
    def make_question(self, data):
        # PREPARE QUESTION
        pre_text = data["pre_text"]
        while type(pre_text) == list:
            pre_text = ("").join(pre_text)
        post_text = data["post_text"]
        while type(post_text) == list:
            post_text = ("").join(post_text)
        table = str(data["table"])
        _question = data["question"]

        # WRAP TABLE
        def format_table(table_str):
            table_str = table_str.replace("[[", "[\n    [")
            table_str = table_str.replace("], [", "],\n    [")
            table_str = table_str.replace("]]", "]\n]")
            return table_str
        # formatted_table = format_table(table)

        question = ""
        if pre_text and not self.easy_task:
            question += f"""{CONTROL_TOKENS.start_of_context}{pre_text}{CONTROL_TOKENS.end_of_context}"""
        if table and not self.easy_task:
            question += f"""{CONTROL_TOKENS.start_of_table}{table}{CONTROL_TOKENS.end_of_table}"""
        if post_text and not self.easy_task:
            question += f"""{CONTROL_TOKENS.start_of_description}{post_text}{CONTROL_TOKENS.end_of_description}"""
        hint = data['gold_inds']
        
        if (self.hint and self.short_answer and hint) or self.easy_task:
            question += f"{CONTROL_TOKENS.start_of_hint}{hint}{CONTROL_TOKENS.end_of_hint}"
        question += f"{CONTROL_TOKENS.start_of_question}{_question}{CONTROL_TOKENS.end_of_question}"

        return question

    def make_answer(self, data):
        # PREPARE ANSWER
        if self.short_answer:
            _answer = data["final_result"] if data["answer"] == "" else min(data["answer"], data["final_result"], key=len)
            _answer = _answer if _answer else data["answer"] 
        else:
            _answer = data["gold_inds"]
            while type(_answer) == list:
                _answer = ("").join(_answer)
            program = data["program_re"]

        answer = f"{CONTROL_TOKENS.start_of_text}{_answer}" 
        if (not self.short_answer) and (not self.easy_task):
            answer += f"{CONTROL_TOKENS.start_of_program}{program}{CONTROL_TOKENS.end_of_program}"
        answer += f"{CONTROL_TOKENS.end_of_text}"

        return answer
    
    def make_mask_old(self, idx: int):
        len_q = self.encodings["len_q"][idx]
        start_pos = self.encodings["start_positions"][idx]
        end_pos = self.encodings["end_positions"][idx]
        
        key_padding_mask = np.zeros(len_q, dtype=np.int64)
        key_padding_mask = pad_sequence(key_padding_mask, self.max_length, 1)
        print("key_padding_mask", key_padding_mask)
        
        attention_mask = np.triu(np.ones((self.max_length, self.max_length), dtype=np.int64), k=start_pos)

        mask = key_padding_mask * attention_mask
        mask[end_pos-start_pos:] = 1
        mask[:,end_pos:] = 1
        return mask.astype(np.bool)
    
    def make_mask(self, idx: int):
        len_q = self.encodings["len_q"][idx]
        start_pos = self.encodings["start_positions"][idx]
        end_pos = self.encodings["end_positions"][idx]

        attention_mask = np.triu(np.ones((self.max_length, self.max_length), dtype=np.int64), k=end_pos - self.max_length + 1)
        attention_mask[:,0] = 0
        attention_mask[:,len_q:start_pos] = 1
        attention_mask[:,end_pos:] = 1

        return attention_mask.astype(bool)

    def read_data(self, data):
        # GET QUESTION AND ANSWER
        question = self.make_question(data)
        answer = self.make_answer(data)

        data["question_text"] = question
        data["answer_text"] = answer
        
        return data

    def prepare_data(self, data):
        question = data["question_text"]
        answer = data["answer_text"]
        # TOKENIZE THEM
        input_ids = self.tokenizer(question, padding=False, return_tensors=True)
        labels = self.tokenizer(answer, padding=False, return_tensors=True)

        len_q = min(self.max_q_len, len(input_ids))

        if self.pad_all_answers:
            start_pos = self.max_q_len
            end_positions = min(len(labels), self.max_a_len) + start_pos
            input_ids = pad_sequence(input_ids, self.max_q_len, self.tokenizer.pad_token_id)
            labels = pad_sequence(labels, self.max_a_len + 1, self.tokenizer.pad_token_id)
            input_ids = torch.cat([input_ids, labels[:-1]])
            labels = labels[1:]
            input_ids = input_ids.squeeze(0).cpu()
        else:
            raise NotImplementedError("Only pad_all_answers=True is supported for now.")

        # return input_ids, labels, mask, start_pos, end_positions
        input_ids = input_ids.numpy()
        labels = labels.numpy()
        return input_ids, labels, start_pos, end_positions, len_q
    

def pad_sequence(token_ids: Union[torch.Tensor, np.ndarray], max_length: int, pad_token_id: int) -> torch.Tensor:
    token_instance = token_ids  # Store the original token_ids instance

    if isinstance(token_ids, np.ndarray):
        token_ids = torch.tensor(token_ids, dtype=torch.long)
    if len(token_ids) < max_length:
        pad_tokens = torch.tensor([pad_token_id] * (max_length - len(token_ids)), dtype=torch.long)
        token_ids = torch.cat([token_ids, pad_tokens])
    else:
        token_ids = token_ids[-max_length:]

    return token_ids if isinstance(token_instance, torch.Tensor) else token_ids.numpy()
  
def get_data(
        tokenizer: Tokenizer,
        max_length: int = 1024,
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
        "max_a_length": max_a_length,
        "short_answer": short_answer,
        "hint": kwargs.get("hint", False),
        "easy_task": kwargs.get("easy_task", False),
        "pad_all_answers": kwargs.get("pad_all_answers", True)
    }
    print("params", params)
    train = FinQADataset(train, **params)
    test = FinQADataset(test, **params)
    val = FinQADataset(val, **params)
    
    return train, test, val