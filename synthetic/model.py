import time
import math
import torch
import random
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, hidden_dims=32):
        super(MLP, self).__init__()

        input_dims  = 2
        output_dims = 5

        self.layers = nn.Sequential(
            nn.Linear(input_dims, hidden_dims, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dims, output_dims, bias=True),
        )

    def forward(self, x):
        return self.layers(x)
        