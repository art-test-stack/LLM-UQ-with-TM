from llm.data.tokenizer import Tokenizer, CONTROL_TOKENS_LIST
from llm.data.glove import get_glove_tokenizer_and_embeddings
from llm.model import LLM, DecoderBlock
import os


def torch_handler(params):
    embedding_ = None
    if "tokenizer" in params:
        if params["tokenizer"].startswith("glove"):
            glove_path = os.getenv("GLOVE_DIR")
            if not glove_path:
                raise ValueError("Environment variable GLOVE_PATH is not set.")
            tokenizer, embedding_ = get_glove_tokenizer_and_embeddings(
                glove_path=glove_path, model_name=params["tokenizer"], dim_model=params["config"]["model_size"],
                force_init=True)
    else:
        try:
            tokenizer = Tokenizer(model_name=params["tokenizer"])
        except:
            tokenizer = Tokenizer(model_name="gpt2")
        tokenizer.add_special_tokens(CONTROL_TOKENS_LIST)

    model = LLM(
        vocab_size=tokenizer.get_vocab_size(), 
        padding_idx=tokenizer.pad_token_id, 
        embedding_=embedding_,
        **params["config"]
    )
    
    return model, tokenizer, DecoderBlock
