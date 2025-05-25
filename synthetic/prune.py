import os
import time
import torch
import random
import torch.nn as nn
from hydra.utils import get_original_cwd

from model import MLP
from dataset import get_dataloaders

from plot_decision_boundary import plot_decision_boundary
        
class Prune:
    def __init__(self, cfg):
        self.cfg            = cfg
        self.biased_dataset = cfg.dataset.biased
        self.random_prune   = cfg.prune.random_prune

        print()
        print("Pruning type: {}".format("Random" if self.random_prune else "Structured"))
        print("Dataset type: {}".format("Biased" if self.biased_dataset else "Un-biased / Uniform"))
        time.sleep(2)

        #Load datasets like overfit.py
        self.train_data, self.test_data, self.miss_data, self.all_train, _ = get_dataloaders(
                                                                                        cfg.dataset.train_samples,
                                                                                        cfg.dataset.test_samples,
                                                                                        cfg.dataset.miss_samples,
                                                                                        batch_size=cfg.model.batch_size, 
                                                                                        biased=cfg.dataset.biased,
                                                                                                )

        #Load model
        self.cwd      = get_original_cwd()
        self.model    = MLP(hidden_dims=cfg.model.hidden_dims) 
        self.datatype = "biased" if cfg.dataset.biased else "unbiased"
        self.model.load_state_dict(torch.load(f"{self.cwd}/saved_models/overfitted_{self.datatype}.pt"))

        #Get the initial accuracies
        test_accuracy, train_accuracy, miss_accuracy = self.get_accuracies()
        print("Initial accuracies:")
        print("Test accuracy: {} | Train accuracy: {} | Miss accuracy: {}".format(test_accuracy, train_accuracy, miss_accuracy), end="\n\n")

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.model.lr)

        # Create plots directory if it doesn't exist
        if not os.path.exists(f"{self.cwd}/plots"):
            os.makedirs(f"{self.cwd}/plots")
        self.pruning_type = "random" if self.random_prune else "structured"
        if not os.path.exists(f"{self.cwd}/plots/{self.datatype}_{self.pruning_type}"):
            os.makedirs(f"{self.cwd}/plots/{self.datatype}_{self.pruning_type}")

        self.start()

    def freeze_pruned_gradients(self):
        """
        Zeroes out gradients on any parameter entries that are currently exactly zero,
        so the optimizer will not update them.
        Call this after loss.backward() but before optimizer.step().
        """
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            # build a mask: 1.0 where param.data != 0, 0.0 where param.data == 0
            keep_mask = param.data.ne(0.0).float()
            # zero out gradients on pruned entries
            param.grad.data.mul_(keep_mask)

    def count_zero_parameters(self):
        """
        Count zero-valued parameters in the model, separately for weights vs. biases.

        Returns:
            zero_weights (int):   number of zero entries among all weight tensors
            total_weights (int):  total number of entries among all weight tensors
            zero_biases (int):    number of zero entries among all bias tensors
            total_biases (int):   total number of entries among all bias tensors
        """
        zero_weights = 0
        total_weights = 0
        zero_biases = 0
        total_biases = 0

        for name, param in self.model.named_parameters():
            # param.data is a Tensor
            num_elements = param.numel()
            num_zero     = int((param.data == 0).sum().item())

            if "bias" in name:
                zero_biases  += num_zero
                total_biases += num_elements
            else:
                zero_weights += num_zero
                total_weights += num_elements

        return zero_weights, total_weights, zero_biases, total_biases
         
    def get_accuracies(self):
        #Evaluate accuracy on test, miss and train data
        with torch.no_grad():
            test_accuracy = 0
            for data, label in self.test_data:
                label          = label
                pred_probs     = self.model(data)
                pred           = torch.argmax(pred_probs, -1)
                test_accuracy += (pred == label).float().mean().item()

            train_accuracy = 0
            for data, label in self.train_data:
                label           = label
                pred_probs      = self.model(data)
                pred            = torch.argmax(pred_probs, -1)
                train_accuracy += (pred == label).float().mean().item()

            miss_accuracy = 0
            for data, label in self.miss_data:
                label          = label
                pred_probs     = self.model(data)
                pred           = torch.argmax(pred_probs, -1)
                miss_accuracy += (pred == label).float().mean().item()

        #Round accuracies to 4 decimal places
        test_accuracy  = round(test_accuracy, 4)
        miss_accuracy  = round(miss_accuracy, 4)
        train_accuracy = round(train_accuracy, 4)
        
        return test_accuracy, train_accuracy, miss_accuracy
    
    def find_prune_candidates(self):
        #Get all activations
        for data, label in self.all_train:
            data = data     
            all_activations = self.model.collect_all_activations(data)

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
        if self.random_prune:
            lowest_std_indices = random.sample(range(total_length), self.cfg.prune.num_prune)
        else:
            lowest_std_indices = indices[std_sorted != 0][:self.cfg.prune.num_prune]
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

        return prune_layers, prune_neurons, prune_weights

    def start(self):
        self.hidden_dims = self.cfg.model.hidden_dims
        self.layer_sizes = [2*self.hidden_dims, self.hidden_dims*self.hidden_dims, self.hidden_dims*5]
        self.prunned_idx = torch.zeros((sum(self.layer_sizes),))

        for epoch in range(2001):
            test_accuracy, train_accuracy, miss_accuracy = self.get_accuracies()

            if epoch % 10 == 0:
                test_accuracy  = round(test_accuracy, 3)
                miss_accuracy  = round(miss_accuracy, 3)
                train_accuracy = round(train_accuracy, 3)
                zero_weights, total_weights, zero_biases, total_biases = self.count_zero_parameters()
                print(f"\033[F\033[KEpoch: {epoch} \t| Test: {test_accuracy} \t| Train: {train_accuracy} \t| Miss: {miss_accuracy} \t| zw: {zero_weights} \t tw: {total_weights} zb: {zero_biases}, tb: {total_biases}", end="\n")

            #Plot decision boundary
            if epoch % 50 == 0:
                plot_decision_boundary(self.model, f"{self.cwd}/plots/{self.datatype}_{self.pruning_type}/{epoch}.png", test_accuracy, train_accuracy, miss_accuracy, biased=self.biased_dataset)
                
            #Find the prune candidates
            prune_layers, prune_neurons, prune_weights = self.find_prune_candidates()

            #For the prune layer and parameter index, set the parameter to zero
            for prune_layer, prune_neuron, prune_weight in zip(prune_layers, prune_neurons, prune_weights):
                for parameter_idx, parameter in enumerate(self.model.parameters()):
                    if (parameter_idx // 2) == prune_layer and parameter_idx % 2 == 0:  #To avoid bias
                        parameter.data[prune_neuron][prune_weight] = 0.0
                        
            #Retrain the model for 1 epoch on all training data
            for j in range(1):
                for data in iter(self.all_train):
                    inputs, labels = data
                    inputs = inputs
                    labels = labels
                    outputs = self.model(inputs)
                    loss = nn.CrossEntropyLoss()(outputs, labels)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.freeze_pruned_gradients()
                    self.optimizer.step()

if __name__ == "__main__":
    Prune()