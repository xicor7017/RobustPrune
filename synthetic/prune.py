import os
import time
import torch
import random
import torch.nn as nn

from model import MLP
from dataset import get_dataloaders

#from torch.utils.tensorboard import SummaryWriter

from plot_decision_boundary import plot_decision_boundary

'''
class Logger:
    def __init__(self, name):
        self.writer = SummaryWriter(name)

    def log(self, name, value, step):
        self.writer.add_scalar(name, value, step)
'''
        
class Prune:
    def __init__(self, cfg):
        self.cfg            = cfg
        self.biased_dataset = cfg.dataset.biased
        self.random_prune   = cfg.prune.random_prune

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
        self.model   = MLP(hidden_dims=cfg.model.hidden_dims) 
        self.datatype     = "biased" if cfg.dataset.biased else "unbiased"
        self.model.load_state_dict(torch.load(f"saved_models/overfitted_{self.datatype}.pt"))

        #Get the initial accuracies
        test_accuracy, train_accuracy, miss_accuracy = self.get_accuracies()
        print("Initial accuracies:")
        print("Test accuracy: {} | Train accuracy: {} | Miss accuracy: {}".format(test_accuracy, train_accuracy, miss_accuracy), end="\n\n")

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.model.lr)

        # Create plots directory if it doesn't exist
        if not os.path.exists("plots"):
            os.makedirs("plots")
        self.pruning_type = "random" if self.random_prune else "structured"
        if not os.path.exists(f"plots/{self.datatype}_{self.pruning_type}"):
            os.makedirs(f"plots/{self.datatype}_{self.pruning_type}")

        self.start()
         
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
            lowest_std_indices = random.sample(range(total_length), 10)
        else:
            lowest_std_indices = indices[std_sorted != 0][:10]
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
                print(f"Epoch: {epoch} \t| Test: {test_accuracy} \t| Train: {train_accuracy} \t| Miss: {miss_accuracy}", end="\n")

            #Plot decision boundary
            if epoch % 50 == 0:
                plot_decision_boundary(self.model, f"plots/{self.datatype}_{self.pruning_type}/{epoch}.png", test_accuracy, train_accuracy, miss_accuracy, biased=self.biased_dataset)
                
            #Get the number of model parameters that are not zero
            num_parameters = 0
            for layer, parameter in enumerate(self.model.parameters()):
                if layer % 2 == 0:  #To avoid bias
                    num_parameters += (parameter == 0).sum().item()
                
            prune_layers, prune_neurons, prune_weights = self.find_prune_candidates()

            #Manual:
            #For the prune layer and parameter index, set the parameter to zero
            for prune_layer, prune_neuron, prune_weight in zip(prune_layers, prune_neurons, prune_weights):
                for parameter_idx, parameter in enumerate(self.model.parameters()):
                    #print(parameter_idx, parameter.shape)
                    if (parameter_idx // 2) == prune_layer and parameter_idx % 2 == 0:  #To avoid bias
                        parameter.data[prune_neuron][prune_weight] = 0.0
        

            #Automatic:
            #Set the corresponding masks to zero
            prune_layer, prune_neuron, prune_weight = prune_layers[0], prune_neurons[0], prune_weights[0]
            if random.random() < 0.4:
                self.model.update_masks(prune_layer, prune_neuron, prune_weight)     
            
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
                    self.optimizer.step()
        
        

if __name__ == "__main__":
    Prune()