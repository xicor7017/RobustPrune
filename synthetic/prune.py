import os
import time
import torch
import random
import torch.nn as nn
from hydra.utils import get_original_cwd

from prune_utils import Pruner
from dataset import get_dataloaders

from model import MLP
from transformer_model import TransformerPrunableEncoder

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
        if cfg.model.type.lower() == "mlp":
            self.model   = MLP(hidden_dims=cfg.model.hidden_dims)
            print("Remember to set num_prune = 10 for the transformer model in the config file.")
            time.sleep(1) 
        else:
            self.model    = TransformerPrunableEncoder(
                            input_dim=2,
                            d_model=32,
                            nhead=2,
                            num_layers=2,
                            dim_feedforward=32,
                            num_classes=5,
                            max_seq_len=1
                        )
            print("Remember to set num_prune = 100 for the transformer model in the config file.")
            time.sleep(1)

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


        self.pruner = Pruner(self.model)

        self.start()

    def print_model_architecture(self, model: nn.Module):
        """
        Recursively prints all submodules in the given model.
        """
        for x in model.named_children():
            print(x)
            '''
            # skip the top‐level module if you only want its children:
            if name == "":
                pass
                #print(f"{model.__class__.__name__}:")
            else:
                print(f"{name}: {layer}")
            '''

        time.sleep(1000)

    '''
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
    '''

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

    def start(self):
        self.hidden_dims = self.cfg.model.hidden_dims
        self.layer_sizes = [2*self.hidden_dims, self.hidden_dims*self.hidden_dims, self.hidden_dims*5]
        self.prunned_idx = torch.zeros((sum(self.layer_sizes),))

        for epoch in range(401):
            test_accuracy, train_accuracy, miss_accuracy = self.get_accuracies()

            if epoch % 10 == 0:
                test_accuracy  = round(test_accuracy, 3)
                miss_accuracy  = round(miss_accuracy, 3)
                train_accuracy = round(train_accuracy, 3)
                zero_weights, total_weights, zero_biases, total_biases = self.count_zero_parameters()
                
                total_masks = self.pruner.mask.float().sum().item()

                print(f"\033[F\033[KEpoch: {epoch} \t| Test: {test_accuracy} \t| Train: {train_accuracy} \t| Miss: {miss_accuracy} \t| zw: {zero_weights} \t tw: {total_weights} zb: {zero_biases}, tb: {total_masks}", end="\n")
                #print(f"Epoch: {epoch} \t| Test: {test_accuracy} \t| Train: {train_accuracy} \t| Miss: {miss_accuracy} \t| zw: {zero_weights} \t tw: {total_weights} zb: {zero_biases}, tb: {total_masks}", end="\n")


            #Plot decision boundary
            if epoch % 50 == 0:
                plot_decision_boundary(self.model, f"{self.cwd}/plots/{self.datatype}_{self.pruning_type}/{epoch}.png", test_accuracy, train_accuracy, miss_accuracy, biased=self.biased_dataset)

            with torch.no_grad():
                self.pruner.prune_weights(self.all_train, self.cfg.prune.num_prune, prune_type="random" if self.random_prune else "structured")
                
            #Retrain the model for 1 epoch on all training data
            for j in range(10):
                for data in iter(self.all_train):
                    inputs, labels = data
                    inputs = inputs
                    labels = labels
                    outputs = self.model(inputs)
                    loss = nn.CrossEntropyLoss()(outputs, labels)

                    self.optimizer.zero_grad()
                    loss.backward()
                    #self.pruner.freeze_pruned_gradients()
                    self.optimizer.step()
                    #self.pruner.force_zero_weights()

if __name__ == "__main__":
    Prune()