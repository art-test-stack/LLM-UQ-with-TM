import torch
from typing import List, Union, Tuple


def get_tensor_stats(tensor: Union[torch.Tensor, list[torch.Tensor]]) -> Tuple[float, float, float, float, float]:
    if type(tensor) == list:
        tensor = torch.Tensor(tensor)

    mean = torch.mean(tensor).item()
    median = torch.median(tensor).item()
    std = torch.std(tensor).item()
    max = torch.max(tensor).item()
    min = torch.min(tensor).item()
    return mean, median, std, max, min

def compute_beta_dist_params(mean, std):
    variance = std ** 2
    alpha = mean * ((mean * (1 - mean) / variance) - 1)
    beta_param = (1 - mean) * ((mean * (1 - mean) / variance) - 1)
    return alpha, beta_param

def get_grad_spectrums(grads: List[torch.Tensor]) -> List[torch.Tensor]:
    spectrums = []
    for grad in grads:
        if grad.ndim == 2:
            u, s, v = torch.svd(grad)
            spectrums.append(s)
    return spectrums

def compute_cosine_similarity(grads1: torch.Tensor, grads2: torch.Tensor) -> Union[None, torch.Tensor]:
    cosine_dist = None
    if grads1 is not None and grads2 is not None:
        print(grads1.shape, grads2.shape)
        assert len(grads1) == len(grads2), "The number of layers should be the same"
        cosine_dist = []
        for layer_1, layer_2 in zip(grads1, grads2):
            cos_dist = torch.matmul(layer_1, layer_2)
            cos_dist /= torch.norm(layer_1) * torch.norm(layer_2)
            cosine_dist.append(cos_dist)
        
    
    return cosine_dist
