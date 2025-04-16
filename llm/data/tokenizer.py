from llm.data.special_tokens import SpecialTokens
from pathlib import Path
from typing import List, Union

import tiktoken
from tiktoken import _tiktoken


# @DeprecationWarning("This class is deprecated. Use SPECIAL_TOKENS instead.")
class CONTROL_TOKENS:
    padding = '<|padding|>'
    unknown = '<|unknown|>'
    tab = '<|tab|>'
    new_line = '<|new_line|>'
    bos = '<|begin_of_text|>'
    eos = '<|end_of_text|>'
    shi = '<|start_of_header|>'
    eoh = '<|end_of_header|>'
    eot = '<|eot_id|>'
    # DEPRECATED PART
    start_of_text = '<|startoftext|>'
    end_of_text = ''

    # start_of_table = '<|startoftable|>'
    # end_of_table = '<|endoftable|>'
    # start_of_description = '<|startofdescription|>'
    # end_of_description = '<|endofdescription|>'
    # start_of_program = '<|startofprogram|>'
    # end_of_program = '<|endofprogram|>'
    # start_of_question = '<|startofquestion|>'
    # end_of_question = '<|endofquestion|>'
    # start_of_context = '<|startofcontext|>'
    # end_of_context = '<|endofcontext|>'
    # start_of_hint = '<|startofhint|>'
    # end_of_hint = '<|endofhint|>'

# @DeprecationWarning("This class is deprecated. Use llm.special_tokens.SpecialTokens() instead.")
BASE_SPECIAL_TOKENS = {
    "pad_token": CONTROL_TOKENS.padding,
    "bos_token": CONTROL_TOKENS.start_of_text,
    "eos_token": CONTROL_TOKENS.end_of_text,
}
tiktoken_models = list(tiktoken.model.MODEL_TO_ENCODING.keys())

import numpy as np
import torch

class Tokenizer:
    def __init__(
            self,
            model_name: str = "gpt-4o",
            special_tokens: SpecialTokens = SpecialTokens(),
        ) -> None:
        assert model_name in tiktoken_models, f"'{model_name}' is not a provided model"
        self.model = tiktoken.encoding_for_model(model_name)
        self.model = tiktoken.Encoding(
            name=self.model.name,
            pat_str=self.model._pat_str,
            mergeable_ranks=self.model._mergeable_ranks,
            special_tokens=self.model._special_tokens,
        )
        # LINE BELLOW VERY IMPORTANT FOR TIKTOKEN INIT BUT DO NOT KNOW WHY
        # IF NOT USED, THE SPECIAL TOKENS ARE NOT ADDED
        self.pad_token_id = 0 

        self.add_special_tokens(special_tokens.list())

        self._compute_vocab_size()
        self.pad_token_id = self.special_tokens[special_tokens.pad]
        self.bos_token_id = self.special_tokens[special_tokens.start_of_text]
        self.eos_token_id = self.special_tokens[special_tokens.end_of_text]

        self._compute_vocab_size()

    def get_vocab_size(self) -> int:
        return self.vocab_size
    
    def _compute_vocab_size(self) -> int:
        self.vocab_size = self.model.n_vocab # - len(self._get_missing_idx())

    def add_special_tokens(self, special_tokens: Union[List[str],str]) -> None:
        special_tokens = special_tokens if isinstance(special_tokens, list) else list(special_tokens)
        special_tokens = [str(special_token) for special_token in special_tokens if special_token not in self.model._special_tokens] 
        missing_token_idx = self._get_missing_idx()
        token_ids = missing_token_idx[:len(special_tokens)] if len(missing_token_idx) >= len(special_tokens) else missing_token_idx + list(range(self.model.n_vocab, self.model.n_vocab + len(special_tokens) - len(missing_token_idx)))
        special_tokens = self.model._special_tokens | {
            token: id
            for token, id in zip(special_tokens, token_ids)
        }
        self.model = tiktoken.Encoding(
            name=self.model.name,
            pat_str=self.model._pat_str,
            mergeable_ranks=self.model._mergeable_ranks,
            special_tokens=special_tokens,
        )
        self.pat_str = self.model._pat_str
        self.mergeable_ranks = self.model._mergeable_ranks
        self.special_tokens = self.model._special_tokens
        self._compute_vocab_size()
    
    def _get_missing_idx(self) -> List[int]:
        missing_token_idx = []
        for idx in range(self.model.n_vocab):
            try:
                self.decode([idx])
            except:
                missing_token_idx.append(idx)
        return missing_token_idx
    
    def __call__(
            self, 
            text: str, 
            retrieve_splitted_text: bool = False, 
            add_control_tokens: bool = False, 
            padding: bool = True, 
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
            padding: bool = True, 
            max_length: int = 1024,
            return_tensors: bool = False, 
            verbose: bool = False
        ) -> Union[List[int], List[tuple[int, str]]]:
        token_ids = self.model.encode(text, allowed_special="all")
        
        if add_control_tokens:
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]

        if padding:
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
