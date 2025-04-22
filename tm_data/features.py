import torch
from typing import List, Union, Tuple


def get_tensor_stats(tensor: Union[torch.Tensor, list[torch.Tensor]]) -> Tuple[float, float, float, float, float, float]:
    """
    Compute statistics of a tensor or a list of tensors.
    Args:
        tensor (Union[torch.Tensor, list[torch.Tensor]]): Input tensor or list of tensors.
    
    Returns:
        Tuple[float, float, float, float, float, float]: Mean, median, standard deviation, max, min, and noise scale.
    """
    if type(tensor) == list:
        tensor = torch.Tensor(tensor)

    mean = torch.mean(tensor).item()
    median = torch.median(tensor).item()
    std = torch.std(tensor).item()
    max = torch.max(tensor).item()
    min = torch.min(tensor).item()
    return { "grad_mean": mean, "grad_median": median, "grad_std": std, "grad_max": max, "grad_min": min, "grad_noise_scale": std / mean**2 }
    # return mean, median, std, max, min, std / mean**2

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
        if isinstance(grads1, list):
            for layer_1, layer_2 in zip(grads1, grads2):
                cos_dist = torch.matmul(layer_1, layer_2)
                cos_dist /= torch.norm(layer_1) * torch.norm(layer_2)
                cosine_dist.append(cos_dist)
        else:
            cos_dist = torch.matmul(grads1, grads2)
            cos_dist /= torch.norm(grads1) * torch.norm(grads2)
            cosine_dist = cos_dist
    else:
        print("One of the gradients is None")
        return 0.
    return cosine_dist

def compute_grad_dir(grads: torch.Tensor, last_grads: torch.Tensor) -> float:
    if grads is not None and last_grads is not None:
        assert len(grads) == len(last_grads), "The number of layers should be the same"
        grad_dist = last_grads - grads
        grad_dist /= torch.norm(grad_dist)
        return grad_dist.mean().item()
    else:
        return 0.
