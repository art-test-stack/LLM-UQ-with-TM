
from llm.data.tokenizer import Tokenizer, CONTROL_TOKENS_LIST
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForQuestionAnswering
from peft import LoraConfig, TaskType, get_peft_model

import torch
import os

from llm.data.tokenizer import CONTROL_TOKENS_LIST, CONTROL_TOKENS
from typing import Union, List
import torch


class TokenizerHGFLlama:
    def __init__(self, tokenizer):
        self.model = tokenizer
        self.model.add_special_tokens({"additional_special_tokens": CONTROL_TOKENS_LIST})
        self.model.update_post_processor()
        self.pad_token_id = self.model.vocab[CONTROL_TOKENS.padding]
        self.bos_token_id = self.model.vocab[self.model.bos_token]
        self.eos_token_id = self.model.vocab[self.model.eos_token]

    def __call__(
        self, 
        text: str, 
        add_control_tokens: bool = False,
        padding: str = "max_length", 
        max_length: int = 1024, 
        return_tensors: bool = True
        ):
        return self.encode(
            text, 
            add_control_tokens=add_control_tokens, 
            padding=padding, 
            max_length=max_length,
            return_tensors=return_tensors
        )
    
    def encode(
            self, 
            text: str, 
            add_control_tokens: bool = False,
            padding: bool = True, 
            max_length: int = 1024,
            return_tensors: bool = False, 
        ):
        # if isinstance(text, list):
        #     # token_ids = [ 
        #     #     self.model.encode(
        #     #         tk, 
        #     #         return_tensors="pt" if return_tensors else "np", 
        #     #         padding="max_length" if padding else "none", 
        #     #         add_special_tokens=add_control_tokens,
        #     #     ).ids for tk in text 
        #     # ]
        #     token_ids = (
        #         self.encode(
        #             tk, 
        #             add_control_tokens=add_control_tokens,
        #             padding=padding,
        #             max_length=max_length,
        #             return_tensors=return_tensors
        #         ) for tk in text
        #     )
        # else:
        token_ids = self.model.encode(
                text, 
                return_tensors="pt" if return_tensors else "np", 
                padding="max_length" if padding else "do_not_pad", 
                add_special_tokens=add_control_tokens,
        )
        if isinstance(text, str):
            token_ids = token_ids[0]
        return token_ids
    
    def decode(self, token_ids: Union[List[int], List[List[int]], torch.Tensor, List[torch.Tensor]]) -> Union[str, List[str]]:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        if isinstance(token_ids[0], list) or isinstance(token_ids[0], torch.Tensor):
            return [self.decode(tk) for tk in token_ids]
        
        return self.model.decode([tk for tk in token_ids if tk != self.pad_token_id]) 
    
    
    def get_vocab_size(self):
        return self.model.vocab_size

base_lora_config = {
    "target_modules": ["q_proj", "v_proj", "o_proj",],
    "r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "bias": 'none',
    "modules_to_save": ['classifier'],
}
def hgface_handler(params):
    """
    Load model and tokenizer based on the type of model.
    
    """
    base_model = params.get("base_model", "meta-llama/Llama-2-7b-hf")
    print(f"Loading model from {base_model}")
    tokenizer = params.get("tokenizer", "meta-llama/Llama-2-7b-hf")
    lora_config = params.get("lora", base_lora_config)

    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    lora_config = LoraConfig(
        **lora_config
        # target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj", "proj"],
        # target_modules=["q_proj", "v_proj", "o_proj",],
        # r=r, 
        # lora_alpha=16, 
        # lora_dropout=0.1,
        # bias='none',
        # modules_to_save=['classifier'],
    )
    model = get_peft_model(model, lora_config)
    tokenizer = TokenizerHGFLlama(tokenizer)
    return model, tokenizer, None
    