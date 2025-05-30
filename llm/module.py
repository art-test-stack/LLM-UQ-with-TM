import torch
from torch import nn

import numpy as np

class Module(nn.Module):
    '''class Module'''
    def nb_parameters(self) -> int:
        '''Give the number of parameters of the module'''
        return sum([np.prod(p.size(), dtype = np.int32) for p in self.parameters()])

    def nb_trainable_parameters(self) -> int:
        '''Give the number of trainable parameters of the module'''
        return sum([np.prod(p.size(), dtype = np.int32) for p in self.parameters() if p.requires_grad])

    def nb_non_trainable_parameters(self) -> int:
        '''Give the number of non-trainable parameters of the module'''
        return sum([np.prod(p.size(), dtype = np.int32) for p in self.parameters() if not p.requires_grad])

    def summary(self) -> None:
        '''Summarize the module'''
        print(f'Number of parameters: {self.nb_parameters():,}')
        print(f'Number of trainable parameters: {self.nb_trainable_parameters():,}')
        print(f'Number of non-trainable parameters: {self.nb_non_trainable_parameters():,}')

    def clean_nan(self) -> None:
        '''Remove NaNs from the module gradients'''
        for p in self.parameters():
            if p.grad is not None:
                torch.nan_to_num(p.grad, nan = 0, posinf = 1e5, neginf = -1e5, out = p.grad)

    def memory_storage(self):
        return sum([p.element_size() * p.nelement() for p in self.parameters()])

    def clip_gradient(self, max_norm: float) -> None:
        '''Clip the module gradients'''
        nn.utils.clip_grad_norm_(self.parameters(), max_norm)
    
    def get_grads(self) -> list:
        '''Get the module gradients'''
        # for p, s in zip(self.parameters(), self.shapes):
        #     if p.grad is not None:
        #         print("param", p.cpu().shape)
        #         print("shape", s)
        #         p.grad.cpu().view(s)
        # return [p.grad.cpu().view(s) for p, s in zip(self.parameters(), self.shapes) if p.grad is not None]
        return [ p.grad for p in self.parameters() if p.grad is not None ]
    
    def save_shapes(self) -> None:
        self.shapes = [ p.shape for p in self.parameters() ]
    
def nb_parameters(model: nn.Module) -> int:
        '''Give the number of parameters of the module'''
        return sum([np.prod(p.size(), dtype = np.int32) for p in model.parameters()])

def nb_trainable_parameters(model: nn.Module) -> int:
    '''Give the number of trainable parameters of the module'''
    return sum([np.prod(p.size(), dtype = np.int32) for p in model.parameters() if p.requires_grad])

def nb_non_trainable_parameters(model: nn.Module) -> int:
    '''Give the number of non-trainable parameters of the module'''
    return sum([np.prod(p.size(), dtype = np.int32) for p in model.parameters() if not p.requires_grad])

def summary(model: nn.Module) -> None:
    '''Summarize the module'''
    print(f'Number of parameters: {nb_parameters(model):,}')
    print(f'Number of trainable parameters: {nb_trainable_parameters(model):,}')
    print(f'Number of non-trainable parameters: {nb_non_trainable_parameters(model):,}')
