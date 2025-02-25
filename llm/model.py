from llm.module import Module

import torch
from torch import nn


import math

DIM_MODEL = 512
MAX_CONTEXT = 512
VOCAB_SIZE = 10000
N_HEAD = 8
N_ENC_LAYERS = 6
N_DEC_LAYERS = 6

class PositionalEncoding(nn.Module):
    def __init__(
            self, 
            d_model: int = DIM_MODEL, 
            max_len: int = MAX_CONTEXT
        ):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe, persistent=True)

    def forward(self, x):
        pe = self.pe[:,:x.size(1), :].to(x.device)
        assert not torch.isnan(x).any(), "NaN detected in x"
        assert not torch.isinf(x).any(), "Inf detected in x"
        assert not torch.isnan(pe).any(), "NaN detected in pe"
        assert not torch.isinf(pe).any(), "Inf detected in pe"

        x = x + self.pe[:,:x.size(1), :].to(x.device)
        return x
    
class TransformerBase(Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, src, tgt):
        assert self.embedding, "Model must have to be attributed in self.embedding"
        assert self.pos_enc, "Model must have to be attributed in self.pos_enc"
        assert self.main, "Model must have to be attributed in self.main"
        assert self.fc, "Model must have to be attributed in self.fc"

        if self.training:
            tgt = self.embedding(tgt)
            tgt = tgt + self.pos_enc(tgt)

            src = self.embedding(src)
            src = src + self.pos_enc(src)

            out = self.main(src, tgt)
            out = self.fc(out)
            out = self.softmax(out)
            return out
        
        else:
            return self.inference(src, tgt)


class DecoderOnlyLLM(Module):
    def __init__(
            self, 
            vocab_size: int = VOCAB_SIZE, 
            model_size: int = DIM_MODEL,
        ):
        super(LLM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, model_size)
        self.pos_enc = PositionalEncoding(d_model=model_size)

        decoder_layer = nn.TransformerDecoderLayer(d_model=model_size, nhead=8)
        self.main = nn.TransformerDecoder(decoder_layer, num_layers=6)
        
        self.fc = nn.Linear(model_size, vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        self.init_weights()
    
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(self.weight, mean=0.0, std=0.02)
            
    def forward(self, tgt, memory):
        tgt = self.embedding(tgt)
        tgt = tgt + self.pos_enc(tgt)
        
        memory = self.embedding(memory)
        memory = memory + self.pos_enc(memory)

        out = self.main(tgt, memory)
        out = self.fc(out)
        out = self.softmax(out)
        return out
    
    
class LLM(Module):
    def __init__(
            self, 
            vocab_size: int = VOCAB_SIZE, 
            model_size: int = DIM_MODEL,
            max_content: int = MAX_CONTEXT,
            nhead: int = N_HEAD,
            num_encoder_layers: int = N_ENC_LAYERS,
            num_decoder_layers: int = N_DEC_LAYERS,
        ):
        super(LLM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=model_size)
        self.pos_enc = PositionalEncoding(d_model=model_size, max_len=max_content)

        # decoder_layer = nn.TransformerDecoderLayer(d_model=model_size, nhead=8)
        # self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        
        # encoder_layer = nn.TransformerEncoderLayer(d_model=model_size, nhead=8)
        # self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        self.main = nn.Transformer(
            model_size, 
            nhead=nhead, 
            num_encoder_layers=num_encoder_layers, 
            num_decoder_layers=num_decoder_layers
        )

        self.fc = nn.Linear(model_size, vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        self.init_weights()
        self.save_shapes()
    
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)
            
    def forward(self, src, tgt):
        assert tgt.min() >= 0, "Embedding input contains negative indices!"
        assert tgt.max() < self.embedding.num_embeddings, f"Embedding input exceeds dictionary size! Found size: {tgt.max()} which is >= {self.embedding.num_embeddings} instead of strictly lower"

        tgt = self.embedding(tgt)
        src = self.embedding(src)

        tgt = self.pos_enc(tgt)
        
        src = self.pos_enc(src) # .table[:src.size(1)]

        out = self.main(src, tgt)
        out = self.fc(out)
        out = self.softmax(out)
        return out