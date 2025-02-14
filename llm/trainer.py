from llm.model import LLM
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from tqdm import tqdm


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.05):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.save_model = True

    def __call__(self, test_loss):
        self.save_model = False
        if self.best_loss is None:
            self.best_loss = test_loss
            self.save_model = True
        elif test_loss < self.best_loss - self.min_delta:
            self.best_loss = test_loss
            self.counter = 0
            self.save_model = True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class Trainer:
    def __init__(
            self,
            model: LLM,
            optimizer: optim.Optimizer,
            criterion: nn.Module,
            device: torch.device,
            name: str = "model"
        ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model.to(device)
        self.name = name

    def fit(
            self, 
            train_set, 
            val_set, 
            epochs: int = 100,
            batch_size: int = 32,
            patience: int = 5,
            min_delta: float = 0.05,
            verbose: bool = True
        ) -> None:
        
        early_stopping = EarlyStopping()

        history = {"train_loss": [], "test_loss": []}
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, drop_last=True)
        early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

        with tqdm(range(epochs), unit="epoch", disable=not verbose) as tepoch:
            for epoch in tepoch:
                train_loss = self._train_epoch(train_loader)
                test_loss = self._test_epoch(val_loader)

                history["train_loss"].append(train_loss)
                history["test_loss"].append(test_loss)

                tepoch.set_postfix(train_loss=train_loss, test_loss=test_loss)

                early_stopping(test_loss)
                if early_stopping.early_stop:
                    break

                # history["test_loss"].append(test_loss / len(dev_loader.dataset))

            tepoch.set_postfix(
                loss = history["train_loss"][-1],
                test_loss = history["test_loss"][-1], 
                # auc = 100. * auc_value,
                # gmean = 100. * gmean_value,
            )
        self.save_model()

    def _train_epoch(self, train_set) -> float:
        self.model.train()
        total_loss = 0
        for src, tgt in train_set:
            src, tgt = src.to(self.device), tgt.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(src, tgt)
            loss = self.criterion(output.view(-1, output.size(-1)), tgt.view(-1))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            break
        return total_loss / len(train_set)
    
    def _test_epoch(self, val_set) -> float:
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for src, tgt in val_set:
                src, tgt = src.to(self.device), tgt.to(self.device)
                output = self.model(src, tgt)
                loss = self.criterion(output.view(-1, output.size(-1)), tgt.view(-1))
                total_loss += loss.item()
                break
        return total_loss / len(val_set)
    
    def save_model(self):
        path = f"models/{self.name}.pt"
        # torch.save(self.model.state_dict(), path + ".pt")
        # for param in self.model.parameters():
        #     if param.grad is not None:
        #         torch.save(param.grad, path + f"_{param.shape}.grad.pt")
        # torch.save(self.optimizer.state_dict(), path + ".opt.pt")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.criterion
        }, path)
        self.save_grads()
    
    def load_model(self):
        path = f"models/{self.name}.pt"
        checkpoint = torch.load(path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.criterion = checkpoint['loss']
        self.load_grads()
        return self.model
    
    def save_grads(self):
        path = f"models/{self.name}.pt"
        for id, param in enumerate(self.model.parameters()):
            if param.grad is not None:
                # print(param.grad.shape)
                torch.save(param.grad, path + f"_{id}_{param.shape}.grad.pt")
    
    def load_grads(self):
        path = f"models/{self.name}.pt"
        for id, param in enumerate(self.model.parameters()):
            grad = torch.load(path + f"_{id}_{param.shape}.grad.pt")
            param.grad = grad
        return self.model