from llm.data.special_tokens import SpecialTokens
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForQuestionAnswering
from peft import LoraConfig, TaskType, get_peft_model

import torch
import os

from typing import Union, List



class TokenizerHGFLlama:
    def __init__(self, tokenizer, special_tokens: SpecialTokens = SpecialTokens()):
        self.model = tokenizer
        # self.model.add_special_tokens({"additional_special_tokens": CONTROL_TOKENS_LIST})
        # self.model.update_post_processor()
        # self.pad_token_id = self.model.vocab[CONTROL_TOKENS.padding]
        # self.pad_token_id = self.model.vocab[self.model.eos_token]

        self.pad_token_id = self.model.vocab[special_tokens.pad]
        print("Pad token id:", self.pad_token_id)
        # self.pad_token_id = self.model.vocab["<0x00>"]
        
        self.bos_token_id = self.model.vocab[special_tokens.start_of_text]
        self.eos_token_id = self.model.vocab[special_tokens.end_of_text]

        self.vocab = dict(sorted(tokenizer.vocab.items(), key=lambda item: item[1]))
        self.max_token_id = max(self.vocab.values())

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
    
    def __len__(self):
        return max(self.vocab.values())
    
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
        return max(self.vocab.values())
    
    
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
    auth_token = os.environ.get("HGF_TOKEN", None)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype="auto", device_map="auto", token=auth_token)
    if "test" in params["name"]:
        model.model.layers = model.model.layers[:2]
    special_tokens = params.get("special_tokens", SpecialTokens())
    special_tokens = SpecialTokens(**special_tokens)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer, 
        token=auth_token,
        extra_special_tokens=special_tokens.dict())
    tokenizer = TokenizerHGFLlama(tokenizer, special_tokens)
    
    # print(tokenizer.vocab)
    # TODO: Check if it is done correctly
    # model.base_model.padding_id = tokenizer.pad_token_id
    model.resize_token_embeddings(tokenizer.max_token_id)
    print("Model embeddings resized to", tokenizer.max_token_id)
    print("Model embeddings size", model.get_input_embeddings().weight.size())
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
    return model, tokenizer, None, special_tokens
    