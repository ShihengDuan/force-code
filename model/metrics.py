import torch
from torch import nn, Tensor

def r2_score(
    real: Tensor,
    pred: Tensor,) -> Tensor:
    B, C, T, X, Y = real.shape()
    real = real.reshape(B, T, -1)
    pred = pred.reshape(B, T, -1)
    mse = torch.square(real-pred) # B, T
    mean = torch.mean(real, dim=[0, 1]) # X*Y=grid points
    tss = torch.square(real-mean)
    