from llm.handlers.llama import llama_handler
from llm.handlers.torch import torch_handler

from typing import Dict

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
    if params["type"] == "llama":
        model, tokenizer, TransformerBlock = llama_handler(params)
    elif params["type"] == "torch":
        model, tokenizer, TransformerBlock = torch_handler(params)
    else:
        raise ValueError("Model type not supported")
    print("Model and tokenizer loaded!")

    return model, tokenizer, TransformerBlock