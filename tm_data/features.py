import torch
from typing import List, Union


def get_tensor_stats(tensor: torch.Tensor) -> Union[float, float, float, float, float]:
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

def get_grad_spectrums(grads, spectra):
    spectrums = []
    for grad in grads:
        if grad.ndim == 2:
            u, s, v = torch.svd(grad)
            spectra.append(s)
    return spectrums