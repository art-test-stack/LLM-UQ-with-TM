from llm.data.special_tokens import SpecialTokens
from llm.data.tokenizer import Tokenizer
from llm.data.glove import get_glove_tokenizer_and_embeddings
from llm.model import LLM, DecoderBlock
from utils import get_device
import os


def torch_handler(params):
    embedding_ = None
    if "tokenizer" in params:
        if params["tokenizer"].startswith("glove"):
            glove_path = os.getenv("GLOVE_DIR")
            if not glove_path:
                raise ValueError("Environment variable GLOVE_PATH is not set.")
            if hasattr(params, "special_tokens"):
                raise Warning("Custom special tokens are not supported with glove tokenizer. Using default special tokens.")
            special_tokens = SpecialTokens()
            tokenizer, embedding_ = get_glove_tokenizer_and_embeddings(
                glove_path=glove_path, 
                model_name=params["tokenizer"], 
                dim_model=params["config"]["model_size"],
                force_init=False
            )
    else:
        special_tokens = params.get("special_tokens", {})
        special_tokens = SpecialTokens(**special_tokens)
        try:
            tokenizer = Tokenizer(model_name=params["tokenizer"])
        except:
            tokenizer = Tokenizer(model_name="gpt2")
        tokenizer.add_special_tokens(special_tokens.list())

    model = LLM(
        vocab_size=tokenizer.get_vocab_size(), 
        padding_idx=tokenizer.pad_token_id, 
        embedding_=embedding_,
        **params["config"]
    )
    model.to(get_device())
    model.device = get_device()
    
    return model, tokenizer, DecoderBlock, special_tokens
