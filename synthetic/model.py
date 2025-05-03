import time
import torch
import random
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, hidden_dims=32):
        super(MLP, self).__init__()

        input_dims  = 2
        output_dims = 5

        self.w1 = torch.randn(input_dims, hidden_dims)
        self.w1 = nn.Parameter(self.w1 / torch.norm(self.w1))
        self.b1 = nn.Parameter(torch.zeros(hidden_dims))
        self.a1 = nn.Tanh()
        self.w2 = torch.randn(hidden_dims, hidden_dims)
        self.w2 = nn.Parameter(self.w2 / torch.norm(self.w2))
        self.b2 = nn.Parameter(torch.zeros(hidden_dims))
        self.a2 = nn.Tanh()
        self.w3 = torch.randn(hidden_dims, output_dims)
        self.w3 = nn.Parameter(self.w3 / torch.norm(self.w3))
        self.b3 = nn.Parameter(torch.zeros(output_dims))

        self.masks = []
        self.masks.append(torch.ones_like(self.w1))
        self.masks.append(torch.ones_like(self.w2))
        self.masks.append(torch.ones_like(self.w3))
        
    def update_masks(self, prune_layer, prune_neuron, prune_weight):
        self.masks[prune_layer][prune_neuron, prune_weight] = 0.0

    def forward(self, x):
        x = torch.matmul(x, (self.w1 * self.masks[0])) + self.b1
        x = self.a1(x)
        x = torch.matmul(x, (self.w2 * self.masks[1])) + self.b2
        x = self.a2(x)
        x = torch.matmul(x, (self.w3 * self.masks[2])) + self.b3
        return x
        
    def collect_all_activations(self, x):
        activations = []
 
        w1 = self.w1 * self.masks[0]
        activations_1 = x.unsqueeze(1) * torch.transpose(w1.unsqueeze(0).repeat(x.shape[0], 1, 1), 1,2)
        activations.append(torch.transpose(activations_1,0,2))

        x = torch.matmul(x, w1) + self.b1
        x = self.a1(x)

        w2 = self.w2 * self.masks[1]
        activations_2 = x.unsqueeze(1) * torch.transpose(w2.unsqueeze(0).repeat(x.shape[0], 1, 1), 1,2)
        activations.append(torch.transpose(activations_2,0,2))

        x = torch.matmul(x, w2) + self.b2
        x = self.a2(x)

        w3 = self.w3 * self.masks[2]
        activations_3 = x.unsqueeze(1) * torch.transpose(w3.unsqueeze(0).repeat(x.shape[0], 1, 1), 1,2)
        activations.append(torch.transpose(activations_3,0,2))

        x = torch.matmul(x, w3) + self.b3
        
        return activations