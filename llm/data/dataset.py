from llm.data.special_tokens import SpecialTokens
from llm.data.tokenizer import Tokenizer
import datasets

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

from datetime import datetime
from typing import Any, Tuple, Union, List, Dict

special_characters = [
    # "%",
    ",",
    ":",
    ";"
]
div_operations = [":", "/"]


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
            teacher_forcing: bool = False,
            **kwargs: Dict[str, Any]
        ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_q_len = max_length - max_a_length
        self.max_a_len = max_a_length
        self.teacher_forcing = teacher_forcing
        self.special_tokens: SpecialTokens = kwargs.get("special_tokens", SpecialTokens())
        self.instruct = kwargs.get("instruct", False)
        
        if hint and not short_answer:
            print("Warning: Hint is only available for short answer. Ignoring the hint parameter.")

        self.hint = hint and not (hint and not short_answer)
        self.short_answer = short_answer
        self.easy_task = easy_task

        data = list(map(self.read_data, data))
        self.encodings = list(map(self.prepare_data, data))
        
        if not self.teacher_forcing:
            encodings = []
            keys = list(self.encodings[0].keys())
            for encoding in self.encodings:
                encodings.extend([{ key: encoding[key][i] if not key in ["start_positions", "end_positions"] else self.max_length -1 for key in keys } for i in range(len(encoding["input_ids"]))])
            
            self.encodings = encodings

        print("len(self.encodings)", len(self.encodings))
        # self.encodings = [{ key: val[i] for i, key in enumerate(["input_ids", "labels", "start_positions", "end_positions", "len_q"])  } for val in encodings]
        # self.encodings = { key: [val[i] for val in encodings] for i, key in enumerate(["input_ids", "labels", "mask", "start_positions", "end_positions"]) }

    def __len__(self):  
        return len(self.encodings)

    def __getitem__(self, idx: int):
        mask = self.make_mask(idx)
        # res = { key: torch.tensor(val[idx]) for key, val in self.encodings.items() if not key == "len_q" }
        # res = { key: torch.tensor(val[idx]) for key, val in self.encodings.items() }
        res = self.encodings[idx]
        res["mask"] = mask
        
        return res
    
    def make_qa_pair(self, data):
        # CLEAN TEXT CONTEXT
        pre_text = clean_text(data["pre_text"])
        post_text = clean_text(data["post_text"])

        # WRAP TABLE
        table = format_table(data["table"])

        # PREPARE QUESTION
        _question = data["question"]

        _answer = self.make_answer(data)

        answer_type = get_answer_formats(_answer)
        instruction = make_instruction(answer_type)
        
        if not self.easy_task:
            context_desc = f"A pre-text, a table"
            
            if self.hint and not self.short_answer:
                context_desc += f", a post-text and a hint are given to you."
            else:
                context_desc += f" and a post-text are given to you."
        
        if self.easy_task:
            context_desc = "A hint is given to you."

        # WRAP QUESTION
        
        question = f"{self.special_tokens.start_of_text}{self.special_tokens.start_of_header}"
        if self.instruct:
            question += f"system{self.special_tokens.end_of_header}"
            question += f"\n\nCutting Knowledge Date: December 2023\nToday Date: {datetime.now().strftime('%d %b %Y')}"
            question += f"\n\nYou are a financial question answering chatbot. {context_desc} You need to reasonnate on the data to answer the question. {instruction}{self.special_tokens.eot_id}{self.special_tokens.start_of_header}"
        if pre_text and not self.easy_task:
            question += f"pre_text{self.special_tokens.end_of_header}"
            question += f"\n\nPre-text: {pre_text}{self.special_tokens.eot_id}{self.special_tokens.start_of_header}"
        if table and not self.easy_task:
            question += f"table{self.special_tokens.end_of_header}"
            question += f"\n\nTable: {table}{self.special_tokens.eot_id}{self.special_tokens.start_of_header}"
        if post_text and not self.easy_task:
            question += f"table{self.special_tokens.end_of_header}"
            question += f"\n\Post-text: {post_text}{self.special_tokens.eot_id}{self.special_tokens.start_of_header}"
        hint = clean_text(data['gold_inds'])
        
        if (self.hint and self.short_answer and hint) or self.easy_task:
            question += f"hint{self.special_tokens.end_of_header}"
            question += f"\n\nHint: {post_text}{self.special_tokens.eot_id}{self.special_tokens.start_of_header}"
        question += f"user{self.special_tokens.end_of_header}"
        question += f"\n\nQuestion: {_question}{self.special_tokens.eot_id}{self.special_tokens.start_of_header}"
        question += f"assistant{self.special_tokens.end_of_header}\n\n"

        # PREPARE ANSWER
        answer = f"{self.special_tokens.start_of_answer}{_answer}{self.special_tokens.eot_id}" 
        # if (not self.short_answer) and (not self.easy_task):
        #     answer += f"{SPECIAL_TOKENS.start_of_program}{program}{SPECIAL_TOKENS.end_of_program}"
        
        # answer += f"{self.special_tokens.end_of_text}"

        return question, answer

    def make_answer(self, data):
        # PREPARE ANSWER
        if self.short_answer:
            _answer = clean_answer(data)
        else:
            raise NotImplementedError("Only short_answer is supported for now.")
            _answer = data["gold_inds"]
            while type(_answer) == list:
                _answer = ("").join(_answer)
            program = data["program_re"]

        return _answer
    
    # @DeprecationWarning("Deprecated in favor of make_mask")
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
        len_q = self.encodings[idx]["len_q"]
        start_pos = self.encodings[idx]["start_positions"]
        end_pos = self.encodings[idx]["end_positions"]

        if self.teacher_forcing:
            attention_mask = np.triu(np.ones((self.max_length, self.max_length), dtype=np.int64), k=end_pos - self.max_length + 1)
            attention_mask[:,0] = 0
            attention_mask[:,len_q:start_pos] = 1
            attention_mask[:,end_pos:] = 1

            return attention_mask.astype(bool)

        else: 
            attention_mask = self.encodings[idx]["input_ids"] == self.tokenizer.pad_token_id
            return attention_mask.astype(bool)

    def read_data(self, data):
        # GET QUESTION AND ANSWER
        question, answer = self.make_qa_pair(data)

        data["question_text"] = question
        data["answer_text"] = answer
        
        return data

    def prepare_data(self, data):
        question = data["question_text"]
        answer = data["answer_text"]
        # TOKENIZE THEM
        input_ids = self.tokenizer(question, padding=False, return_tensors=True)
        labels = self.tokenizer(answer, padding=False, return_tensors=True)

        if self.teacher_forcing:
            len_q = min(self.max_q_len, len(input_ids))
            len_a = min(self.max_a_len, len(labels))

            start_pos = self.max_q_len
            end_positions = min(len(labels), self.max_a_len) + start_pos
            input_ids = pad_sequence(input_ids, self.max_q_len, self.tokenizer.pad_token_id)
            labels = pad_sequence(labels, self.max_a_len + 1, self.tokenizer.pad_token_id)
            input_ids = torch.cat([input_ids, labels[:-1]])
            labels = labels[1:]
            input_ids = input_ids.squeeze(0).cpu()
            len_q = torch.tensor(len_q).cpu()
            len_a = torch.tensor(len_a).cpu()
        elif not self.teacher_forcing:
            x, y = [], []
            len_q, len_a = [], []
            for k in range(1, min(self.max_a_len, len(labels)) - 1):
                _x = pad_sequence(input_ids, self.max_length - k, self.tokenizer.pad_token_id)
                _x = torch.cat([_x, labels[:k]])
                _y = labels[k]
                len_q.append(self.max_length - k)
                len_a.append(k)
                x.append(_x)
                y.append(_y)
            start_pos = self.max_length - 1
            end_positions = self.max_length - 1
            input_ids = torch.stack(x).cpu()
            labels = torch.stack(y).reshape(-1,1).cpu()
        else:
            raise NotImplementedError("Only teacher_forcing=True is supported for now.")

        # return input_ids, labels, mask, start_pos, end_positions
        input_ids = input_ids.numpy()
        labels = labels.numpy()
        return { 
            "input_ids": input_ids, 
            "labels": labels, 
            "start_positions": torch.tensor(start_pos), 
            "end_positions": torch.tensor(end_positions), 
            "len_q": len_q, 
            "len_a": len_a
        }


def clean_text(corpus: List[str]) -> str:
    corpus = [text for text in corpus if text != "."]
    for cha in special_characters:
        corpus = [text.replace(f" {cha}", cha) for text in corpus]
    corpus = [text[:-2] + "." if text.endswith(" .") else text for text in corpus]
    corpus = [text.capitalize() for text in corpus]
    corpus = " ".join(corpus)
    return corpus

def format_table(table, row_separator="\n", field_separator="; "):
    """
    Converts a 2D list table into a CLM-friendly string.
    
    Args:
        table (list of list of str): The input table in the format [
            ['', 'header_1', ..., 'header_m'],
            ['entry_1', 'value_11', ..., 'value_1m'],
            ...
        ]
        row_prefix (str): Optional prefix for each entry.
        row_separator (str): Separator between rows.
        field_separator (str): Separator between fields in a row.
        
    Returns:
        str: Formatted string ready for causal language modeling.
    """
    if not table or len(table) < 2:
        return ""

    headers = table[0][1:]  # Skip the empty top-left cell
    entries = table[1:]

    formatted_rows = []
    for row in entries:
        entry_name = row[0]
        values = row[1:]
        fields = [f"{header}: {value}" for header, value in zip(headers, values)]
        row_text = f"{entry_name}: " + field_separator.join(fields)
        formatted_rows.append(row_text)

    return row_separator.join(formatted_rows)

def get_answer_formats(answer):
    # answer = answer.replace(" ", "")
    try:
        answer = float(answer)
        return "a float"
    except:
        if "%" in answer:
            return "a percentage"
        elif answer == "ye" or answer == "yes" or answer == "no":
            return "'yes' or 'no'"
        elif "$" in answer:
            return "a currency"
        else:
            return "unknown"


def make_instruction(answer_type):
    if answer_type == "unknown":
        instruction = "Please provide a short answer."
    else:
        instruction = f"Please provide the answer as {answer_type}."
    return instruction


def clean_answer(data):
    _answer = data["final_result"] if data["answer"] == "" else min(data["answer"], data["final_result"], key=len)
    _answer = _answer if _answer else data["answer"] 
    for op in div_operations:
        if op in _answer:
            quotient, div = _answer.split(op)[0], _answer.split(op)[1]
            _answer = f"{float(quotient) / float(div):.2f}"
    if get_answer_formats(_answer) == "unknown":
        _answer = data["final_result"]
    return _answer

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
        "teacher_forcing": kwargs.get("teacher_forcing", True),
        "special_tokens": kwargs.get("special_tokens", SpecialTokens()),
    }
    print("params", params)
    train = FinQADataset(train, **params)
    test = FinQADataset(test, **params)
    val = FinQADataset(val, **params)
    
    return train, test, val