import torch
from torch import Tensor

def loss_function(tensor_given: Tensor, tensor_decomposition: Tensor) -> int:
    difference = tensor_given - tensor_decomposition
    squared = difference**2
    sum = torch.sum(squared)
    loss = torch.sqrt(sum)
    return loss