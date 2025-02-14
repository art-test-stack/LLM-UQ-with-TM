from pathlib import Path
from typing import List, Union

import tiktoken
from tiktoken import _tiktoken

class CONTROL_TOKENS:
    unknown = '<|unknown|>'
    padding = '<|padding|>'
    start_of_text = '<|startoftext|>'
    tab = '<|tab|>'
    new_line = '<|new_line|>'
    human = '<|human|>'
    system = '<|system|>'
    user = '<|user|>'
    assistant = '<|assistant|>'
    end_of_text = '<|endoftext|>'

CONTROL_TOKENS_LIST = list(CONTROL_TOKENS.__dict__.values())[1:-3]
tiktoken_models = list(tiktoken.model.MODEL_TO_ENCODING.keys())

class Tokenizer:
    def __init__(
            self,
            model_name: str = "gpt-4o",
            special_tokens: List[str] | str = CONTROL_TOKENS_LIST,
        ) -> None:
        assert model_name in tiktoken_models, f"'{model_name}' is not a provided model"
        self.model = tiktoken.encoding_for_model(model_name)

        self.pat_str = self.model._pat_str
        self.mergeable_ranks = self.model._mergeable_ranks

        token_ids = range(self.model.n_vocab, self.model.n_vocab + len(special_tokens))

        special_tokens = special_tokens if type(special_tokens) == list else list(special_tokens)
        self.model._special_tokens = {
            token: id
            for token, id in zip(special_tokens, token_ids)
        }
        self.model._core_bpe = _tiktoken.CoreBPE(self.mergeable_ranks, self.model._special_tokens, self.pat_str)
        self.special_tokens = self.model._special_tokens

    def get_vocab_size(self) -> int:
        return self.model.n_vocab

    def encode(
            self, 
            text: str,
            retrieve_splitted_text: bool = False, 
            add_control_tokens: bool = False,
            verbose: bool = False
        ) -> Union[List[int], List[tuple[int, str]]]:
        token_ids = self.model.encode(text, allowed_special="all")
        if add_control_tokens:
            token_ids = [self.special_tokens[CONTROL_TOKENS.start_of_text]] + token_ids + [self.special_tokens[CONTROL_TOKENS.end_of_text]]
        if retrieve_splitted_text:
            return list(zip(token_ids, self.get_words(token_ids)))
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        return self.model.decode(token_ids) 

    def get_words(
            self,
            token_ids: List[int]
        ) -> List[str]:
        words = [ self.decode([tk]) for tk in token_ids ]
        return words