from pathlib import Path
from typing import List, Union

import tiktoken
from tiktoken import _tiktoken

class CONTROL_TOKENS:
    unknown = '<|unknown|>'
    padding = '<|padding|>'
    tab = '<|tab|>'
    new_line = '<|new_line|>'
    start_of_text = '<|startoftext|>'
    end_of_text = '<|endoftext|>'
    start_of_table = '<|startoftable|>'
    end_of_table = '<|endoftable|>'
    start_of_description = '<|startofdescription|>'
    end_of_description = '<|endofdescription|>'
    start_of_program = '<|startofprogram|>'
    end_of_program = '<|endofprogram|>'
    start_of_question = '<|startofquestion|>'
    end_of_question = '<|endofquestion|>'
    start_of_context = '<|startofcontext|>'
    end_of_context = '<|endofcontext|>'
    start_of_answer = '<|startofanswer|>'
    end_of_answer = '<|endofanswer|>'

CONTROL_TOKENS_LIST = list(CONTROL_TOKENS.__dict__.values())[1:-3]
tiktoken_models = list(tiktoken.model.MODEL_TO_ENCODING.keys())

import numpy as np
import torch

class Tokenizer:
    def __init__(
            self,
            model_name: str = "gpt-4o",
            pad_token_id: str = CONTROL_TOKENS.padding,
            soa_token_id: str = CONTROL_TOKENS.start_of_answer,
            eoa_token_id: str = CONTROL_TOKENS.end_of_answer,
        ) -> None:
        assert model_name in tiktoken_models, f"'{model_name}' is not a provided model"
        self.model = tiktoken.encoding_for_model(model_name)
        self.pat_str = self.model._pat_str
        self.mergeable_ranks = self.model._mergeable_ranks

        self.special_tokens = self.model._special_tokens

        self.add_special_tokens([pad_token_id, soa_token_id, eoa_token_id])
        self.pad_token_id = self.special_tokens[pad_token_id]
        self.soa_token_id = self.special_tokens[soa_token_id]
        self.eoa_token_id = self.special_tokens[eoa_token_id]

    def get_vocab_size(self) -> int:
        return self.model.n_vocab + len(self.special_tokens)

    def add_special_tokens(self, special_tokens: Union[List[str],str]) -> None:
        token_ids = range(self.model.n_vocab, self.model.n_vocab + len(special_tokens))

        special_tokens = special_tokens if isinstance(special_tokens, list) else list(special_tokens)
        self.model._special_tokens = {
            token: id
            for token, id in zip(special_tokens, token_ids)
        }
        self.model._core_bpe = _tiktoken.CoreBPE(self.mergeable_ranks, self.model._special_tokens, self.pat_str)
        self.special_tokens = self.model._special_tokens
        
    def __call__(
            self, 
            text: str, 
            retrieve_splitted_text: bool = False, 
            add_control_tokens: bool = False, 
            padding: str = "max_length", 
            max_length: int = 1024,
            return_tensors: bool = False
        ) -> Union[List[int], torch.Tensor, List[torch.Tensor]]:
        return self.encode(
            text, 
            retrieve_splitted_text=retrieve_splitted_text, 
            add_control_tokens=add_control_tokens, 
            padding=padding, 
            max_length=max_length,
            return_tensors=return_tensors
        )

    def encode(
            self, 
            text: str,
            retrieve_splitted_text: bool = False, 
            add_control_tokens: bool = False,
            padding: str = "max_length", 
            max_length: int = 1024,
            return_tensors: bool = False, 
            verbose: bool = False
        ) -> Union[List[int], List[tuple[int, str]]]:
        token_ids = self.model.encode(text, allowed_special="all")
        
        if add_control_tokens:
            token_ids = [self.special_tokens[CONTROL_TOKENS.start_of_text]] + token_ids + [self.special_tokens[CONTROL_TOKENS.end_of_text]]

        if padding == "max_length":
            token_ids = self.pad_sequence(token_ids, max_length)
        
        if return_tensors:
            token_ids = torch.tensor(token_ids, dtype=torch.long)

        if retrieve_splitted_text:
            return list(zip(token_ids, self.get_words(token_ids)))
        
        return token_ids

    def pad_sequence(self, token_ids: List[int], max_length: int) -> List[int]:
        if len(token_ids) < max_length:
            return token_ids + [self.pad_token_id] * (max_length - len(token_ids))
        else:
            return token_ids[-max_length:]

    def decode(self, token_ids: Union[List[int], List[List[int]], torch.Tensor, List[torch.Tensor]]) -> Union[str, List[str]]:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if isinstance(token_ids[0], list):
            return [self.decode(tk) for tk in token_ids]
        return self.model.decode([tk for tk in token_ids if tk != self.pad_token_id]) 

    def get_words(self, token_ids: List[int]) -> List[str]:
        words = [self.decode([tk]) for tk in token_ids]
        return words
