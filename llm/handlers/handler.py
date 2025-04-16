from llm.handlers.llama import llama_handler
from llm.handlers.torch import torch_handler
from llm.handlers.hgface import hgface_handler

from typing import Dict

from enum import Enum

class ModelType(Enum):
    LLAMA = "llama"
    TORCH = "torch"
    HGFACE = "hgface"

def model_handler(params: Dict):
    """
    Load model and tokenizer based on the type of model.
    
    Args:
        params: dict, parameters for the model
    
    Returns:
        model: nn.Module, model
        tokenizer: tokenizer, tokenizer
        TransformerBlock: nn.Module, transformer block
    """

    print(f"Load model and tokenizer... Model type is {params['type']}")
    assert "type" in params, "Model type not provided"
    assert params["type"] in ModelType._value2member_map_, "Model type not supported"

    if params["type"] == "llama":
        model, tokenizer, TransformerBlock, special_tokens = llama_handler(params)
    elif params["type"] == "torch":
        model, tokenizer, TransformerBlock, special_tokens = torch_handler(params)
    elif params["type"] == "hgface":
        model, tokenizer, TransformerBlock, special_tokens = hgface_handler(params)
    else:
        raise ValueError("Model type not supported")
    print("Model and tokenizer loaded!")

    return model, tokenizer, TransformerBlock, special_tokens