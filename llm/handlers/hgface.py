import transformers
import os

from llm.data.tokenizer import Tokenizer, CONTROL_TOKENS_LIST
from transformers import BertTokenizerFast, BertForQuestionAnswering

def hgface_handler(params):
    """
    Load model and tokenizer based on the type of model.
    
    """
    model_path = params.get("name", "bert-base-uncased")
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = BertForQuestionAnswering.from_pretrained(model_path)

    
    return model, tokenizer, None
    