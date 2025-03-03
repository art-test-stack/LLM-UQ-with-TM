from llm.module import Module

import torch
from torch import nn
import torch.nn.functional as F

from transformers import LogitsProcessor

import math

DIM_MODEL = 512
MAX_CONTEXT = 512
VOCAB_SIZE = 10000
N_HEAD = 8
N_ENC_LAYERS = 6
N_DEC_LAYERS = 6

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
    
        # decoder_layer = nn.TransformerDecoderLayer(d_model=model_size, nhead=8)
        # self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        
        # encoder_layer = nn.TransformerEncoderLayer(d_model=model_size, nhead=8)
        # self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
    
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


        self.main = nn.Transformer(
            model_size, 
            nhead=nhead, 
            num_encoder_layers=num_encoder_layers, 
            num_decoder_layers=num_decoder_layers,
            batch_first=True
        )
        self.max_content = max_content
        self.fc = nn.Linear(model_size, vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        self.init_weights()
        self.save_shapes()
    
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)
            
    def forward(self, src, tgt, has_mask=True):
        assert tgt.min() >= 0, "Embedding input contains negative indices!"
        assert tgt.max() < self.embedding.num_embeddings, f"Embedding input exceeds dictionary size! Found size: {tgt.max()} which is >= {self.embedding.num_embeddings} instead of strictly lower"
        
        src = self.embedding(src)
        tgt = self.embedding(tgt)

        src = self.pos_enc(src)
        tgt = self.pos_enc(tgt)
        mask = None
        if has_mask:
            device = src.device
            seq_len = tgt.shape[1]
            mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).bool()
        
        out = self.main(src, tgt, tgt_mask=mask)
        out = self.fc(out)
        # out = self.softmax(out)
        return out

    # def generate(self, src: torch.Tensor, tokenizer, repetition_penalty = 1.2, beam_width=3, len_answer=16, retrieve_probs=False):
    #     """Performs beam search decoding."""
    #     self.eval()
    #     with torch.no_grad():
    #         batch_size = src.shape[0]
    #         beams = [(torch.full((batch_size, 1), tokenizer.soa_token_id, device=src.device), 0)] 
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

    #             if (best_seq[:, -1] == tokenizer.eoa_token_id).all():
    #                 break

    #     return generated
    
    def generate(self, src: torch.Tensor, tokenizer, repetition_penalty = 1.2, beam_width=3, len_answer=16, retrieve_probs=False):
        """Performs beam search decoding."""
        self.eval()
        with torch.no_grad():
            batch_size = src.shape[0]
            beams = [(torch.full((batch_size, 1), tokenizer.soa_token_id, device=src.device), 0, torch.zeros((batch_size, len_answer, self.vocab_size), device=src.device) if retrieve_probs else None)] 
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

                if (best_seq[:, -1] == tokenizer.eoa_token_id).all():
                    break

            if retrieve_probs:
                probabilities = beams[0][2]

        return generated, probabilities if retrieve_probs else generated
