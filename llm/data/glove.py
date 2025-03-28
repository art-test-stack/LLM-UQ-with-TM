from llm.data.tokenizer import CONTROL_TOKENS, CONTROL_TOKENS_LIST
from llm.data.pretokenizer import PreTokenizer

import torch
import numpy as np
from tokenizers import (
    Tokenizer,
    models,
    pre_tokenizers,
    normalizers,
)

import json
from typing import List, Dict, Union
from pathlib import Path


class GloVeEmbedding(torch.nn.Module):
    def __init__(self, embeddings: np.ndarray, dim_model: int):
        super().__init__()
        self.main = torch.nn.Embedding.from_pretrained(torch.from_numpy(embeddings).float())
        assert self.main.weight.shape == embeddings.shape
        self.main.weight.requires_grad = False

        self.ffn = torch.nn.Linear(embeddings.shape[1], dim_model, bias=False)

    def forward(self, x):
        return self.ffn(self.main(x))


class GloVeTokenizer:
    def __init__(
            self,
            glove_dir: str,
            special_tokens: Dict[str, int] = None,
            tokenizer: Tokenizer = None
        ):
        tokenizer_path = str(Path(glove_dir) / 'tokenizer.json')
        strip = True # "6B" in tokenizer_path
        self.special_tokens = special_tokens
        for special_token, idx in special_tokens.items():
            setattr(self, special_token, idx)
        if tokenizer:
            self.model = tokenizer
            self.model.pre_tokenizer = pre_tokenizers.PreTokenizer.custom(PreTokenizer(strip))
        else:
            vocab_path = str(Path(glove_dir) / 'vocab.json')
            self.model = Tokenizer(models.WordLevel(vocab_path, unk_token=CONTROL_TOKENS.unknown))
            self.model.normalizer = normalizers.Sequence(
                [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
            )
            self.model.pre_tokenizer = pre_tokenizers.Whitespace()
            self.model.save(tokenizer_path)
            self.model.pre_tokenizer = pre_tokenizers.PreTokenizer.custom(PreTokenizer(strip))


    @classmethod
    def from_file(cls, glove_dir: str):
        tknzr_path = str(Path(glove_dir) / 'tokenizer.json')
        tokenizer = Tokenizer.from_file(tknzr_path)
        _, special_tokens = load_glove_embs(glove_dir)
        return cls(glove_dir, special_tokens, tokenizer)
    
    def __call__(
            self, 
            text: str, 
            retrieve_splitted_text: bool = False, 
            add_control_tokens: bool = False, 
            padding: str = "max_length", 
            max_length: int = 1024,
            return_tensors: bool = False
        ) -> Union[List[int], torch.Tensor, List[torch.Tensor]]:
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
            verbose: bool = False
        ) -> Union[List[int], List[tuple[int, str]]]:
        if isinstance(text, list):
            token_ids = [ self.model.encode(tk, add_special_tokens=False).ids for tk in text ]
        else:
            token_ids = self.model.encode(text, add_special_tokens=False).ids

        if add_control_tokens:
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]

        if padding:
            token_ids = self.pad_sequence(token_ids, max_length)
        
        if return_tensors:
            token_ids = torch.tensor(token_ids, dtype=torch.long)
        
        return token_ids
    
    def decode(self, token_ids: Union[List[int], List[List[int]], torch.Tensor, List[torch.Tensor]]) -> Union[str, List[str]]:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        if isinstance(token_ids[0], list) or isinstance(token_ids[0], torch.Tensor):
            return [self.decode(tk) for tk in token_ids]
        
        return self.model.decode([tk for tk in token_ids if tk != self.pad_token_id]) 
    
    def pad_sequence(self, token_ids: List[int], max_length: int) -> List[int]:
        if isinstance(token_ids[0], list):
            return [self.pad_sequence(tk, max_length) for tk in token_ids]
        
        if len(token_ids) < max_length:
            return token_ids + [self.pad_token_id] * (max_length - len(token_ids))
        else:
            return token_ids[-max_length:]
    
    def get_vocab_size(self):
        return self.model.get_vocab_size()


def create_special_token_embedding(embeddings: np.ndarray | None = None, dim_emb: int | None = None):
    """
    Create special tokens embedding
    
    Args:
        embeddings (np.ndarray): The embeddings
        dim_emb (int): The dimension of the embeddings
    """
    if embeddings is None and dim_emb is None:
        raise ValueError("Either embeddings or dim_emb must be provided")
    if embeddings is not None and dim_emb is None:
        dim_emb = embeddings.shape[1]
    if embeddings is None:
        embeddings = []

    emb = np.random.rand(dim_emb)
    while any(np.array_equal(emb, arr) for arr in embeddings):
        emb = np.random.rand(dim_emb)
    return emb

def add_special_tokens(vocab: Dict[str, int], dim_emb: int):
    embeddings: List[np.ndarray] = []
    for idx, token in enumerate(CONTROL_TOKENS_LIST):
        if token not in vocab.keys():
            vocab[token] = idx
            emb = create_special_token_embedding(embeddings, dim_emb=dim_emb) if not token == CONTROL_TOKENS.padding else np.zeros(dim_emb)
            embeddings.append(emb)

    special_tokens = {
        'pad_token_id': vocab[CONTROL_TOKENS.padding],
        'bos_token_id': vocab[CONTROL_TOKENS.start_of_text],
        'eos_token_id': vocab[CONTROL_TOKENS.end_of_text],
    }
    return vocab, embeddings, special_tokens


def init_glove_vocab_and_embs(glove_dir: str, model_name: str):
    """
    Initialize the GloVe vocab and embeddings

    Args:
        glove_dir (str): The directory containing the GloVe tokenizer and embedding
        model_name (str): The name of the GloVe model

    Returns:
        vocab (Dict[str, int]): The GloVe vocabulary
        embeddings (np.ndarray): The GloVe embeddings
        special_tokens (Dict[str, int]): The special tokens
    """

    vocab = {}

    model_path = model_name + '.txt'
    glove_path = Path(glove_dir) / model_path
    glove_dir = Path(glove_dir) / model_name

    with open(glove_path, 'r') as f: 
        full_content = f.read().strip().split('\n')
    
    i_word = full_content[0].split(' ')[0]
    i_embeddings = [float(val) for val in full_content[0].split(' ')[1:]]
    vocab, embeddings, special_tokens = add_special_tokens(vocab, len(i_embeddings))
    _szv = len(vocab)
    for i in range(len(full_content)):
        i_word = full_content[i].split(' ')[0]
        i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
        vocab[i_word] = i + _szv
        embeddings.append(i_embeddings)

    embeddings = np.array(embeddings)

    # Check for no duplicates embeddings with special tokens
    for token, token_id in special_tokens.items():
        if embeddings[token_id] in embeddings[len(special_tokens):]:
            embeddings[token_id] = create_special_token_embedding(embeddings)

    save_glove_vocab_and_embeddings(vocab, embeddings, special_tokens, glove_dir)

    return embeddings, special_tokens


def save_glove_vocab_and_embeddings(
        vocab: Dict[str, int], embeddings: np.ndarray, special_tokens: Dict[str, int], glove_dir: str
    ):
    """
    Save the GloVe vocab and embedding to disk
    
    Args:
        vocab (Dict[str, int]): The GloVe vocabulary
        embeddings (np.ndarray): The GloVe embeddings
        special_tokens (Dict[str, int]): The special tokens
        glove_dir (str): The directory to save the GloVe tokenizer and embedding
    """
    path = Path(glove_dir)
    path.mkdir(parents=True, exist_ok=True)

    with open(path / 'vocab.json', "w") as f:
        json.dump(vocab, f)

    with open(path / 'embeddings.npy','wb') as f:
        np.save(f, embeddings)

    with open(path / 'special_tokens.json', "w") as outfile:
        json.dump(special_tokens, outfile)


def load_glove_embs(glove_dir: str):
    """
    Load the GloVe embeddings from disk
    
    Args:
        glove_path (str): The directory containing the GloVe embeddings
    
    Returns:
        embeddings (np.ndarray): The GloVe embeddings
        special_tokens (Dict[str, int]): The special tokens
    """

    with open(glove_dir / 'embeddings.npy','rb') as f:
        embeddings = np.load(f)
    
    with open(glove_dir / 'special_tokens.json', 'r') as f:
        special_tokens = json.load(f)
 
    return embeddings, special_tokens


def get_glove_tokenizer_and_embeddings(glove_path: str, model_name: str, dim_model: int = 512, force_init: bool = False):
    """"
    Get the GloVe tokenizer and embedding"
    
    Args:
        glove_path (str): The directory containing the GloVe tokenizer and embedding
        model_name (str): The name of the GloVe model
        dim_model (int): The dimension of the model
    
    Returns:
        GloVeTokenizer, GloVeEmbedding: The GloVe tokenizer, The GloVe embedding
    """

    glove_dir = Path(glove_path) / model_name
    try:
        assert not force_init
        print("Loading GloVe embeddings...")
        embeddings, special_tokens = load_glove_embs(glove_dir)
    except:
        print("Initializing GloVe vocab and embeddings...")
        embeddings, special_tokens = init_glove_vocab_and_embs(glove_path, model_name)
    
    try:
        assert not force_init
        print("Loading GloVe tokenizer...")
        tokenizer = GloVeTokenizer.from_file(glove_dir)
    except:
        print("Initializing GloVe tokenizer...")
        tokenizer = GloVeTokenizer(glove_dir=glove_dir, special_tokens=special_tokens)
        print("Tokenizer and embeddings files saved at:", glove_dir)
    print("GloVe tokenizer and embeddings loaded successfully!")
    return tokenizer, GloVeEmbedding(embeddings, dim_model)
