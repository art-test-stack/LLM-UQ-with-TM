from llm.data.tokenizer import Tokenizer, CONTROL_TOKENS_LIST
from llm.model import LLM


def torch_handler(params):
    try:
        tokenizer = Tokenizer(model_name=params["tokenizer"])
    except:
        tokenizer = Tokenizer(model_name="gpt2")
    tokenizer.add_special_tokens(CONTROL_TOKENS_LIST)

    model = LLM(
        vocab_size=tokenizer.get_vocab_size(), 
        padding_idx=tokenizer.pad_token_id, 
        **params["config"]
    )
    # model.forward = forward.__get__(model)
    
    return model, tokenizer
