from llm.module import Module

import torch
from torch import nn
import torch.nn.functional as F

from transformers import LogitsProcessor

import math

class RepetitionPenaltyProcessor(LogitsProcessor):
    def __init__(self, penalty):
        self.penalty = penalty

    def __call__(self, input_ids, scores):
        """Applies repetition penalty to logits."""
        batch_size, vocab_size = scores.shape
        for i in range(batch_size):
            unique_tokens = set(map(int, input_ids[i].tolist())) 
            for token in unique_tokens:
                if 0 <= token < vocab_size: 
                    if scores[i, token] < 0:
                        scores[i, token] *= self.penalty
                    else:
                        scores[i, token] /= self.penalty
        return scores
    


class PositionalEncoding(nn.Module):
    def __init__(
            self, 
            d_model: int, 
            max_len: int
        ):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe, persistent=True)


class DecoderBlock(nn.Module):
    def __init__(self, model_size, nhead, dim_ffn, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(model_size, nhead, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(model_size)
        self.ff = nn.Sequential(
            nn.Linear(model_size, dim_ffn),
            nn.GELU(),
            nn.Linear(dim_ffn, model_size)
        )
        self.ln2 = nn.LayerNorm(model_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask, is_causal=True)
        x = self.ln1(x + self.dropout(attn_output))

        ff_output = self.ff(x)
        x = self.ln2(x + self.dropout(ff_output))  
        
        return x

class LLM(Module):
    def __init__(
            self, 
            vocab_size: int, 
            model_size: int,
            dim_ffn: int,
            max_seq_len: int,
            nhead: int,
            num_layers: int,
            padding_idx: int = None,
            embedding_: nn.Module = None,
            dropout: float = 0.1
        ):
        super().__init__()
        self.vocab_size = vocab_size
        self.nhead = nhead
        self.tok_embeddings = embedding_ or nn.Embedding(vocab_size, model_size)
        self.position_embedding = PositionalEncoding(d_model=model_size, max_len=max_seq_len)
        self.layers = nn.ModuleList([DecoderBlock(model_size, nhead, dim_ffn, dropout) for _ in range(num_layers)])
        self.ln_final = nn.LayerNorm(model_size)
        self.output_layer = nn.Linear(model_size, vocab_size)

    def forward(self, input_ids: torch.Tensor, starting_pos: int):
        """
        input_ids: (batch_size, seq_len) - tokenized input sequence
        starting_pos: (batch_size,) - position at which generation should start
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        x = self.tok_embeddings(input_ids) + self.position_embedding.pe[:, :seq_len, :].to(input_ids.device) 

        mask = None
        if seq_len > 1:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device), diagonal=starting_pos-1)
            mask[seq_len - starting_pos + 1:,:] = torch.zeros(starting_pos - 1, seq_len, device=device)
            mask = mask.masked_fill(mask == 0, True).masked_fill(mask == 1, False)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.ln_final(x)
        logits = self.output_layer(x) 

        return logits
