from llm.module import Module

import torch
from torch import nn

DIM_MODEL = 512
MAX_CONTEXT = 512
VOCAB_SIZE = 10000

class PositionalEncoding(Module):
    def __init__(
            self,
            dim_model: int = DIM_MODEL,
            n_pos: int = MAX_CONTEXT,
        ) -> None:
        super().__init__()
        self.register_buffer('table', self._get_sinusoid_encoding_table(dim_model, n_pos))
    
    def _get_sinusoid_encoding_table(self, d_model: int = DIM_MODEL, n_pos: int = MAX_CONTEXT):
        ''' Sinusoid position encoding table '''
        pos = torch.arange(n_pos, dtype=torch.float32)
        i = torch.arange(d_model)

        pos_enc = torch.ger(pos, 1e4 ** (- 2 * (i//2) / d_model))

        pos_enc[:, 0::2] = torch.sin(pos_enc[:, 0::2])
        pos_enc[:, 1::2] = torch.cos(pos_enc[:, 1::2]) 
        return pos_enc
    

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
        tgt = tgt + self.pos_enc.table[:tgt.size(1)]
        
        memory = self.embedding(memory)
        memory = memory + self.pos_enc.table[:memory.size(1)]

        out = self.main(tgt, memory)
        out = self.fc(out)
        out = self.softmax(out)
        return out
    
class LLM(Module):
    def __init__(
            self, 
            vocab_size: int = VOCAB_SIZE, 
            model_size: int = DIM_MODEL,
        ):
        super(LLM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, model_size)
        self.pos_enc = PositionalEncoding()

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
        tgt = tgt + self.pos_enc.table[:tgt.size(1)]
        
        src = self.embedding(src)
        src = src + self.pos_enc.table[:src.size(1)]

        out = self.main(src, tgt)
        out = self.fc(out)
        out = self.softmax(out)
        return out