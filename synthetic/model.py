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

        self.w1 = torch.randn(input_dims, hidden_dims)
        self.w1 = nn.Parameter(self.w1 / torch.norm(self.w1))
        #self.b1 = nn.Parameter(torch.randn(hidden_dims) / math.sqrt(self.w1.size(1)))
        self.b1 = nn.Parameter(torch.zeros(hidden_dims))
        self.a1 = nn.Tanh()
        self.w2 = torch.randn(hidden_dims, hidden_dims)
        self.w2 = nn.Parameter(self.w2 / torch.norm(self.w2))
        #self.b2 = nn.Parameter(torch.randn(hidden_dims) / math.sqrt(self.w2.size(1)))
        self.b2 = nn.Parameter(torch.zeros(hidden_dims))
        self.a2 = nn.Tanh()
        self.w3 = torch.randn(hidden_dims, output_dims)
        self.w3 = nn.Parameter(self.w3 / torch.norm(self.w3))
        #self.b3 = nn.Parameter(torch.randn(output_dims) / math.sqrt(self.w3.size(1)))
        self.b3 = nn.Parameter(torch.zeros(output_dims))

        self.hidden_dims = hidden_dims
        self.layer_sizes = [input_dims*hidden_dims, hidden_dims*hidden_dims, hidden_dims*output_dims] # Not counting biases
        self.prunned_idx = torch.zeros((sum(self.layer_sizes),))

    def forward(self, x):
        x = torch.matmul(x, self.w1) + self.b1
        x = self.a1(x)
        x = torch.matmul(x, self.w2) + self.b2
        x = self.a2(x)
        x = torch.matmul(x, self.w3) + self.b3
        return x
        
    def collect_all_activations(self, x):
        activations = []
 
        activations_1 = x.unsqueeze(1) * torch.transpose(self.w1.unsqueeze(0).repeat(x.shape[0], 1, 1), 1,2)
        activations.append(torch.transpose(activations_1,0,2))

        x = torch.matmul(x, self.w1) + self.b1
        x = self.a1(x)

        activations_2 = x.unsqueeze(1) * torch.transpose(self.w2.unsqueeze(0).repeat(x.shape[0], 1, 1), 1,2)
        activations.append(torch.transpose(activations_2,0,2))

        x = torch.matmul(x, self.w2) + self.b2
        x = self.a2(x)

        activations_3 = x.unsqueeze(1) * torch.transpose(self.w3.unsqueeze(0).repeat(x.shape[0], 1, 1), 1,2)
        activations.append(torch.transpose(activations_3,0,2))

        x = torch.matmul(x, self.w3) + self.b3
        
        return activations
    
    def prune_weights(self, train_data, num_prune, prune_type):
        #Get all activations
        for data, label in train_data:
            all_activations = self.collect_all_activations(data)

        total_length = 0
        for activation in all_activations:
            total_length += activation.shape[0]*activation.shape[1]

        stds = torch.FloatTensor([])
        for layer, activation in enumerate(all_activations):
            #Scale activations to min 0 and max 1 along the last dimension
            activation = activation - torch.min(activation,-1, keepdim=True)[0]
            activation = activation / (torch.max(activation,-1,keepdim=True)[0] + 1e-8)

            #Find std along the last dim
            std = torch.std(activation, -1, unbiased=False)
            std_flatten = std.view(-1)
            stds = torch.cat((stds, std_flatten), -1)

        #Set the stds to inf if the corresponding parameter is already prunned
        stds[self.prunned_idx.bool()] = float("inf")

        #Sort the stds
        std_sorted, indices = torch.sort(stds)
        
        #Get the index of the lowest std that is not zero
        if prune_type == "random":
            lowest_std_indices = random.sample(range(total_length), num_prune)
        else:
            lowest_std_indices = indices[std_sorted != 0][:num_prune]
        self.prunned_idx[lowest_std_indices[0]] = 1

        prune_layers, prune_neurons, prune_weights = [], [], []

        for lowest_std_indice in lowest_std_indices:
            if lowest_std_indice < self.layer_sizes[0]:
                prune_layer = 0
                prune_neuron = lowest_std_indice // self.hidden_dims
                prune_weight = (lowest_std_indice - (prune_neuron*self.hidden_dims)) % 2
            elif lowest_std_indice < self.layer_sizes[0] + self.layer_sizes[1]:
                prune_layer = 1
                prune_neuron = (lowest_std_indice - self.layer_sizes[0]) // self.hidden_dims
                prune_weight = (lowest_std_indice - self.layer_sizes[0]  - (prune_neuron*self.hidden_dims)) % self.hidden_dims
            else:
                prune_layer = 2
                prune_neuron = (lowest_std_indice - self.layer_sizes[0] - self.layer_sizes[1]) // 5
                prune_weight = (lowest_std_indice - self.layer_sizes[0] - self.layer_sizes[1] - (prune_neuron*5)) % self.hidden_dims

            prune_layers.append(prune_layer)
            prune_neurons.append(prune_neuron)
            prune_weights.append(prune_weight)

        #For the prune layer and parameter index, set the parameter to zero
        for prune_layer, prune_neuron, prune_weight in zip(prune_layers, prune_neurons, prune_weights):
            for parameter_idx, parameter in enumerate(self.parameters()):
                if (parameter_idx // 2) == prune_layer and parameter_idx % 2 == 0:  #To avoid bias
                    parameter.data[prune_neuron][prune_weight] = 0.0