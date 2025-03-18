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
            dropout: float = 0.1
        ):
        super().__init__()
        self.vocab_size = vocab_size
        self.nhead = nhead
        self.token_embedding = nn.Embedding(vocab_size, model_size)
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

        x = self.token_embedding(input_ids) + self.position_embedding.pe[:, :seq_len, :].to(input_ids.device) 

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


class LLM_O(Module):
    def __init__(
            self, 
            vocab_size: int, 
            model_size: int,
            dim_ffn: int,
            max_seq_len: int,
            nhead: int,
            num_layers: int,
            padding_idx: int = None,
            dropout: float = 0.1
        ):
        super(LLM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=model_size, padding_idx=padding_idx)
        self.pos_enc = PositionalEncoding(d_model=model_size, max_len=max_seq_len)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_size, 
            dim_feedforward=dim_ffn,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.main = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )
        self.max_seq_len = max_seq_len
        self.fc = nn.Linear(model_size, vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        self.init_weights()
        self.save_shapes()
    
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, start_pos: int):
        assert x.min() >= 0, "Embedding input contains negative indices!"
        assert x.max() < self.embedding.num_embeddings, f"Embedding input exceeds dictionary size! Found size: {x.max()} which is >= {self.embedding.num_embeddings} instead of strictly lower"
    
        _, x_len = x.shape
        x_emb = self.embedding(x) + self.pos_enc.pe[:, :x_len, :].to(x.device) 
        
        mask = torch.triu(torch.ones(x_len, x_len), diagonal=1)
        mask[:, :start_pos] = float('-inf')  
         
        output = self.main(
            tgt=x_emb, 
            memory=x, 
            tgt_mask=mask
        )
        output = self.fc(output) 
        return output

    
    @torch.inference_mode()
    def generate(self, src: torch.Tensor, tokenizer, repetition_penalty = 1.2, beam_width=3, len_answer=16, retrieve_probs=False):
        """Performs beam search decoding."""
        self.eval()
        with torch.no_grad():
            batch_size = src.shape[0]
            beams = [(torch.full((batch_size, 1), tokenizer.bos_token_id, device=src.device), 0, torch.zeros((batch_size, len_answer, self.vocab_size), device=src.device) if retrieve_probs else None)] 
            rep_penalty_processor = RepetitionPenaltyProcessor(repetition_penalty)
            generated = torch.full((batch_size, len_answer), tokenizer.pad_token_id, device=src.device) 
            probabilities = torch.zeros((batch_size, len_answer, self.vocab_size), device=src.device) if retrieve_probs else None

            for t in range(len_answer):
                candidates = []
                for seq, score, probs in beams:
                    logits = self(src, seq, has_mask=True)[:, -1, :] 
                    if logits.isnan().any():
                        print("seq iter", t)
                        print("seq", logits)
                        print("src", src)
                        print("seq", seq)
                        break
                    if retrieve_probs:
                        probs[:,t] = logits.clone()

                    logits = rep_penalty_processor(seq, logits)
                    log_probs = F.log_softmax(logits, dim=-1)
                    top_k_probs, top_k_indices = log_probs.topk(beam_width, dim=-1) 

                    for i in range(beam_width):
                        new_seq = torch.cat([seq, top_k_indices[:, i].unsqueeze(-1)], dim=1)
                        new_score = score + top_k_probs[:, i].sum().item()
                        new_probs = probs.clone() if retrieve_probs else None
                        candidates.append((new_seq, new_score, new_probs))
                
                beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
                best_seq = beams[0][0][:, 1:]

                generated[:, t] = best_seq[:, -1] 

                if (best_seq[:, -1] == tokenizer.eos_token_id).all():
                    break

            if retrieve_probs:
                probabilities = beams[0][2]

        return generated, probabilities if retrieve_probs else generated
