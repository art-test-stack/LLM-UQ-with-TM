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

    def forward(self, x):
        pe = self.pe[:,:x.size(1), :].to(x.device)
        assert not torch.isnan(x).any(), "NaN detected in x"
        assert not torch.isinf(x).any(), "Inf detected in x"
        assert not torch.isnan(pe).any(), "NaN detected in pe"
        assert not torch.isinf(pe).any(), "Inf detected in pe"

        x = x + self.pe[:,:x.size(1), :].to(x.device)
        return x
    


# class DecoderOnlyLLM(Module):
#     def __init__(
#             self, 
#             vocab_size: int = VOCAB_SIZE, 
#             model_size: int = DIM_MODEL,
#         ):
#         super(LLM, self).__init__()

#         self.embedding = nn.Embedding(vocab_size, model_size)
#         self.pos_enc = PositionalEncoding(d_model=model_size)

#         decoder_layer = nn.TransformerDecoderLayer(d_model=model_size, nhead=8)
#         self.main = nn.TransformerDecoder(decoder_layer, num_layers=6)
        
#         self.fc = nn.Linear(model_size, vocab_size)
#         self.softmax = nn.Softmax(dim=-1)
#         self.init_weights()
    
#     def init_weights(self):
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.normal_(self.weight, mean=0.0, std=0.02)
            
#     def forward(self, tgt, memory):
#         tgt = self.embedding(tgt)
#         tgt = tgt + self.pos_enc(tgt)
        
#         memory = self.embedding(memory)
#         memory = memory + self.pos_enc(memory)

#         out = self.main(tgt, memory)
#         out = self.fc(out)
#         out = self.softmax(out)
#         return out
    
        # decoder_layer = nn.TransformerDecoderLayer(d_model=model_size, nhead=8)
        # self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        
        # encoder_layer = nn.TransformerEncoderLayer(d_model=model_size, nhead=8)
        # self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
    
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
        super(LLM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=model_size, padding_idx=padding_idx)
        self.pos_enc = PositionalEncoding(d_model=model_size, max_len=max_seq_len)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_size, 
            dim_ffn=dim_ffn,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.main = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )
        self.max_seq_len = max_seq_len
        self.fc = nn.Linear(model_size, vocab_size, dropout=dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.init_weights()
        self.save_shapes()
    
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(self, seq: torch.Tensor, start_pos: int):
        assert seq.min() >= 0, "Embedding input contains negative indices!"
        assert seq.max() < self.embedding.num_embeddings, f"Embedding input exceeds dictionary size! Found size: {seq.max()} which is >= {self.embedding.num_embeddings} instead of strictly lower"
    
        _, seq_len = seq.shape
        seq_emb = self.embedding(seq) + self.pos_enc(seq)  
        
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask[:, :start_pos] = float('-inf')   

        output = self.main(
            tgt=seq_emb, 
            memory=seq, 
            tgt_mask=mask
        )
        output = self.fc(output) 
        return output

    # def generate(self, src: torch.Tensor, tokenizer, repetition_penalty = 1.2, beam_width=3, len_answer=16, retrieve_probs=False):
    #     """Performs beam search decoding."""
    #     self.eval()
    #     with torch.no_grad():
    #         batch_size = src.shape[0]
    #         beams = [(torch.full((batch_size, 1), tokenizer.bos_token_id, device=src.device), 0)] 
    #         rep_penalty_processor = RepetitionPenaltyProcessor(repetition_penalty)
    #         generated = torch.full((batch_size, len_answer), tokenizer.pad_token_id, device=src.device) 
    #         probabilities = torch.zeros((batch_size, len_answer, self.vocab_size), device=src.device) if retrieve_probs else None

    #         for t in range(len_answer):
    #             candidates = []
    #             for seq, score in beams:
    #                 logits = self(src, seq, has_mask=True)[:, -1, :] 

    #                 logits = rep_penalty_processor(seq, logits)
    #                 probs = F.log_softmax(logits, dim=-1)
    #                 if retrieve_probs:
    #                     probabilities[:, t] = probs
    #                 top_k_probs, top_k_indices = probs.topk(beam_width, dim=-1) 

    #                 for i in range(beam_width):
    #                     new_seq = torch.cat([seq, top_k_indices[:, i].unsqueeze(-1)], dim=1)
    #                     new_score = score + top_k_probs[:, i].sum().item()
    #                     candidates.append((new_seq, new_score))
                
    #             beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
    #             best_seq = beams[0][0][:, 1:]

    #             generated[:, t] = best_seq[:, -1] 

    #             if (best_seq[:, -1] == tokenizer.eos_token_id).all():
    #                 break

    #     return generated
    
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
