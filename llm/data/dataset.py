from llm.data.tokenizer import Tokenizer, CONTROL_TOKENS
import datasets

import torch
from torch.utils.data import DataLoader, Dataset

from typing import Any, Tuple, Union


class FinQADataset(Dataset):
    def __init__(
            self, 
            data: Any, 
            tokenizer: Tokenizer,
            max_length: int = 1024,
            max_q_length: Union[int, None] = None,
            max_a_length: Union[int, None] = None,
            short_answer: bool = True,
        ) -> None:
        self.data = data
        self.questions = data["question"]
        self.answers = data["answer"]
        self.tokenizer = tokenizer
        self.max_q_len = max_q_length if max_q_length else max_length
        self.max_a_len = max_a_length if max_a_length else max_length
        if short_answer:
            self.max_a_len = 16
        self.pad_token_id = tokenizer.special_tokens[CONTROL_TOKENS.padding]
        self.q_token_id = tokenizer.special_tokens[CONTROL_TOKENS.start_of_text]
        self.short_answer = short_answer

        # questions = [ tokenizer.encode(question) for question in data["question"]]
        # answers = [ tokenizer.encode(answer) for answer in data["answer"]]

        # max_len = max([len(q) for q in questions] + [len(a) for a in answers])
        
        # # TODO: USE torch.nn.utils.rnn.pad_sequence instead
        # # https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html
        # self.questions = torch.Tensor([ q + [tokenizer.special_tokens[CONTROL_TOKENS.padding]] * (max_len - len(q)) for q in questions ])
        # self.answers = torch.Tensor([ a + [tokenizer.special_tokens[CONTROL_TOKENS.padding]] * (max_len - len(a)) for a in answers ])
        # self.max_content = max_len

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
            answer = data["final_result"] # if data["answer"] else 
        else:
            answer = data["gold_inds"]
            while type(answer) == list:
                answer = ("").join(answer)
            program = data["program_re"]

        # WRAP THEM
        question = f"{CONTROL_TOKENS.start_of_context}{pre_text}{CONTROL_TOKENS.end_of_context}{CONTROL_TOKENS.start_of_table}{table}{CONTROL_TOKENS.end_of_table}{CONTROL_TOKENS.start_of_text}{post_text}{CONTROL_TOKENS.end_of_text}{CONTROL_TOKENS.start_of_question}{question}{CONTROL_TOKENS.end_of_question}"
        answer = f"{CONTROL_TOKENS.start_of_answer}{answer}{CONTROL_TOKENS.end_of_answer}" 
        if not self.short_answer:
            answer += f"{CONTROL_TOKENS.start_of_program}{program}{CONTROL_TOKENS.end_of_program}"

        # TOKENIZE THEM
        q_enc = self.tokenizer(question, padding='max_length', 
                               max_length=self.max_q_len, return_tensors=True)
        
        a_enc = self.tokenizer(answer, padding='max_length', 
                               max_length=self.max_a_len, return_tensors=True)

        input_ids = q_enc.squeeze(0)
        # attention_mask = causal_mask(len(a_enc)) # + padding_mask(a_enc, self.pad_token_id)
        labels = a_enc.squeeze(0) 
        # labels[labels == self.tokenizer.pad_token_id] = -100  

        return input_ids, labels # , attention_mask


def causal_mask(seq_len):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask.masked_fill(mask == 1, float('-inf'))

def padding_mask(seq, pad_token_id):
    return (seq == pad_token_id).unsqueeze(1) # .unsqueeze(2)
  
def get_data(tokenizer: Tokenizer) -> Tuple[FinQADataset, FinQADataset, FinQADataset]:
    dataset = datasets.load_dataset("ibm-research/finqa", "en")

    train = dataset["train"]
    test = dataset["test"]
    val = dataset["validation"]

    train = FinQADataset(train, tokenizer)
    test = FinQADataset(test, tokenizer)
    val = FinQADataset(val, tokenizer)
    
    return train, test, val