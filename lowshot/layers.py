import torch
import torch.nn as nn
import torch.nn.functional as F


def flatten(x):
    batch_size = x.shape[0]
    return x.view(batch_size, -1)
