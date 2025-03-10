from llm.handlers.llama import llama_handler
from llm.handlers.torch import torch_handler


def model_handler(params):
    if params["type"] == "llama":
        return llama_handler(params)
    elif params["type"] == "torch":
        return torch_handler(params)
    else:
        raise ValueError("Model type not supported")