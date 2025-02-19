from llm.module import Module

import torch
from torch import nn


import math

DIM_MODEL = 512
MAX_CONTEXT = 512
VOCAB_SIZE = 10000

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
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:,:x.size(1), :].to(x.device)
        return x
    

class DecoderOnlyLLM(Module):
    def __init__(
            self, 
            vocab_size: int = VOCAB_SIZE, 
            model_size: int = DIM_MODEL,
        ):
        super(LLM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, model_size)
        self.pos_enc = PositionalEncoding()

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
            max_content: int = MAX_CONTEXT
        ):
        super(LLM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, model_size)
        self.pos_enc = PositionalEncoding(d_model=model_size, max_len=max_content)

        # decoder_layer = nn.TransformerDecoderLayer(d_model=model_size, nhead=8)
        # self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        
        # encoder_layer = nn.TransformerEncoderLayer(d_model=model_size, nhead=8)
        # self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        self.main = nn.Transformer(model_size, nhead=8, num_encoder_layers=6, num_decoder_layers=6)

        self.fc = nn.Linear(model_size, vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        self.init_weights()
    
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)
            
    def forward(self, src, tgt):
        tgt = self.embedding(tgt)
        tgt = self.pos_enc(tgt)
        
        src = self.embedding(src)
        src = self.pos_enc(src) # .table[:src.size(1)]

        out = self.main(src, tgt)
        out = self.fc(out)
        out = self.softmax(out)
        return out