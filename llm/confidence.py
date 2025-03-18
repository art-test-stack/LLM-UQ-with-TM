import torch
import torch.nn.functional as F


class ConfidenceScore:
    def __init__(self, device):
        self.device = device
        self.log_probs = []

    def update(self, raw_output: torch.Tensor):
        log_probs = F.log_softmax(raw_output, dim=-1)
        self.log_probs.append(float(log_probs.mean().item()))

    def get(self):
        return sum(self.log_probs) / len(self.log_probs)