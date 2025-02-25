from llm.model import LLM
from tm_data.preprocessing import InputCSV

import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loops.base import Loop

DIM_MODEL = 512
MAX_CONTEXT = 512
VOCAB_SIZE = 10000

class CustomLoop(Loop):
    def advance(self, batch, i):
        loss = pl.lightning_module.training_step(batch, i)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def run(self, dataloader):
        for i, batch in enumerate(dataloader):
            self.advance(batch, i)

class Wrapper(pl.LightningModule):
    def __init__(
            self,
            model: LLM,
            optimizer: torch.optim.Optimizer,
            criterion: nn.Module, 
            csv_object: InputCSV
        ) -> None:
        super().__init__()

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.csv_object = csv_object


    def forward(self, src, tgt):
        if self.training:
            tgt = self.model.embedding(tgt)
            tgt = tgt + self.model.pos_enc(tgt)

            src = self.model.embedding(src)
            src = src + self.model.pos_enc(src)

            out = self.model.main(src, tgt)
            out = self.model.fc(out)
            out = self.model.softmax(out)
            return out
        else:
            return self.model.inference(src, tgt)

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        # self.model.train()
        total_loss = 0

        ddp_loss = torch.zeros(2).to(self.rank)

        assert not torch.isnan(src).any(), "NaN found in sources!"
        assert not torch.isnan(tgt).any(), "NaN found in targets!"

        src, tgt = src.to(self.rank), tgt.to(self.rank)
        self.optimizer.zero_grad()
        output = self(src, tgt)
        loss = self.criterion(output.view(-1, output.size(-1)), tgt.view(-1))
        loss.backward()
        self.optimizer.step()
        total_loss += loss.item()

        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(src)
        self.csv_object.update_model()
        
        # dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
        # if self.rank == 0:
            # print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, ddp_loss[0] / ddp_loss[1]))
        return ddp_loss[0] / ddp_loss[1]
    
    