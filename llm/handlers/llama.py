from llm.data.special_tokens import SpecialTokens
from utils import get_device

from llama_models.llama3.reference_impl.generation import Llama
from llama_models.llama3.reference_impl.model import TransformerBlock

import fairscale.nn.model_parallel.initialize as fs_init

import tiktoken
import torch

from typing import Callable
import os
from pathlib import Path


def get_reserved_special_tokens(
            special_tokens_size: int, 
            reserved_special_token_pattern: Callable = lambda x: f"<|reserved_special_token_{x}|>"
        ):
        return [reserved_special_token_pattern(i) for i in range(2, special_tokens_size)]

class TokenizerHandler:
    def __init__(self, tknzr):
        self.main = tknzr 

    def __getattr__(self, attr):
        if hasattr(self.main, attr):
            return getattr(self.main, attr)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")

    def __call__(
            self, 
            text: str, 
            padding: str = "max_length", 
            max_length: int = 1024, 
            return_tensors: bool = True
        ):
        token_ids = self.encode(
            text, 
            bos=False, 
            eos=False,
            allowed_special="all"
        )
        if padding == "max_length":
            if len(token_ids) < max_length:
                token_ids = token_ids + [self.pad_token_id] * (max_length - len(token_ids))
            else:
                token_ids = token_ids[:max_length]
        if return_tensors:
            token_ids = torch.tensor(token_ids, dtype=torch.long)
        return token_ids
    
    def get_vocab_size(self):
        return self.model.n_vocab
    
    def add_control_tokens(self, special_tokens: SpecialTokens):
        for attr, token in special_tokens.__dict__.items():
            reserved_special_tokens = get_reserved_special_tokens(self.main.num_reserved_special_tokens)
            if attr == "pad" and "<|reserved_special_token_1|>" in self.special_tokens:
                    self.special_tokens[token] = self.special_tokens.pop("<|reserved_special_token_1|>")
                    # reserved_special_tokens.remove("<|reserved_special_token_1|>")
            for reserved_token in reserved_special_tokens:
                if token not in self.special_tokens and reserved_token in self.special_tokens:
                    self.special_tokens[token] = self.special_tokens.pop(
                        reserved_token
                    )
                    reserved_special_tokens.remove(reserved_token)
                    break
        self.special_tokens = dict(sorted(self.special_tokens.items(), key=lambda item: item[1]))
        self.model._special_tokens = self.special_tokens
        self.main.special_tokens = self.special_tokens

        self.main.model = tiktoken.Encoding(
            name=self.main.model.name,
            pat_str=self.main.model._pat_str,
            mergeable_ranks=self.main.model._mergeable_ranks,
            special_tokens=self.special_tokens,
        )

        self.pad_token_id = self.special_tokens[special_tokens.pad] 
        self.bos_token_id = self.special_tokens[special_tokens.start_of_text] 
        self.eos_token_id = self.special_tokens[special_tokens.end_of_text] 

    def decode(self, token_ids):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if isinstance(token_ids[0], list):
            return [self.main.decode(ids) for ids in token_ids]
        return self.main.decode(token_ids)      
        
# def forward(self, tokens: torch.Tensor, start_pos: int):
#     print("start pos", start_pos)
#     _bsz, seq_len = tokens.shape
#     h = self.tok_embeddings(tokens)
#     self.freqs_cis = self.freqs_cis.to(h.device)
#     freqs_cis = self.freqs_cis[start_pos : start_pos + seq_len]

#     mask = None
#     if seq_len > 1:
#         mask = torch.full((seq_len, seq_len), float("-inf"), device=tokens.device)

#         mask = torch.triu(mask, diagonal=1)

#         # https://github.com/pytorch/pytorch/issues/100005
#         # torch.triu is buggy when the device is mps: filled values are
#         # nan instead of 0.
#         if mask.device.type == torch.device("mps").type:
#             mask = torch.nan_to_num(mask, nan=0.0)

#         # When performing key-value caching, we compute the attention scores
#         # only for the new sequence. Thus, the matrix of scores is of size
#         # (seq_len, cache_len + seq_len), and the only masked entries are (i, j) for
#         # j > cache_len + i, since row i corresponds to token cache_len + i.
#         mask = torch.hstack([torch.zeros((seq_len, start_pos), device=tokens.device), mask]).type_as(h)

#     for layer in self.layers:
#         h = layer(h, start_pos, freqs_cis, mask)
#     h = self.norm(h)
#     output = self.output(h).float()
#     return output

def llama_forward(self, src: torch.Tensor, start_pos: int = 1, mask: torch.Tensor = None):
    _bsz, seq_len = src.shape
    device = src.device
    h = self.tok_embeddings(src)

    self.freqs_cis = self.freqs_cis.to(h.device)
    freqs_cis = self.freqs_cis[:seq_len] 

    if mask is None:
        if seq_len > 1:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device), diagonal=start_pos-1)
            mask[seq_len - start_pos + 1:,:] = torch.zeros(start_pos - 1, seq_len, device=device)
            mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            
            if mask.device.type == "mps":
                mask = torch.nan_to_num(mask, nan=0.0)

            mask = mask.type_as(h)

    for layer in self.layers:
        h = layer(h, 0, freqs_cis, mask)

    h = self.norm(h)
    output = self.output(h).float()
    
    return output

# @DeprecationWarning("This function is deprecated. Use hugging face handler with the corresponding Llama model instead.")
def llama_handler(params):
    ckpt_dir = os.getenv(params["ckpt_dir"])
    ckpt_dir = Path(ckpt_dir).joinpath(params["name"])
    llama_obj = Llama.build(
        ckpt_dir=ckpt_dir,
        max_seq_len=params["config"]["max_seq_len"],
        max_batch_size=8,
        device=get_device()
    )
    
    model = llama_obj.model
    model.forward = llama_forward.__get__(model)
    model = model.float()
    
    tknzr = llama_obj.tokenizer
    tokenizer = TokenizerHandler(tknzr)
    special_tokens = params.get("special_tokens", {})
    special_tokens = SpecialTokens(**special_tokens)

    tokenizer.add_control_tokens(special_tokens)
    print(tokenizer.decode(tokenizer("Does it work?", padding="None",return_tensors=False)))
    return model, tokenizer, TransformerBlock, special_tokens

