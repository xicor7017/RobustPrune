import torch
import torch.nn as nn
import random
import time
from collections import defaultdict

class Pruner:
    """
    Generic pruner that selects weights based on weight-wise activation variability.
    This version uses weight-wise normalization (Method 1) instead of sample-wise.
    """
    def __init__(self, model: nn.Module):
        """
        Args:
            model: the neural network to prune (contains nn.Linear submodules)
        """
        self.model = model
        self.prunable = []
        offset = 0
        for module in model.modules():
            if isinstance(module, nn.Linear):
                weight = module.weight
                numel = weight.numel()
                self.prunable.append({
                    'module': module,
                    'shape': weight.shape,
                    'numel': numel,
                    'offset': offset,
                })
                offset += numel
        print(f"Found a total of {offset} weights.")

        self.mask = torch.zeros(offset, dtype=torch.bool)

    def _register_hooks(self):
        """
        Attach a forward-hook to every nn.Linear module.
        The hook captures, for each batch, the per-weight contributions:
        contribution[b,i,j] = x[b,j] * w[i,j]
        """
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                handle = module.register_forward_hook(self._make_hook(name))
                self.handles.append(handle)

    def _remove_hooks(self):
        """
        Clean up: remove all forward hooks.
        """
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def _make_hook(self, layer_name):
        def hook(module, inputs, outputs):
            # inputs[0]: shape [batch_size, in_features]
            x = inputs[0].detach().to('cpu')
            w = module.weight.detach().to('cpu')  # [out_features, in_features]

            # Compute contributions: [batch, out, in]
            # Each weight w[i,j] multiplies each input x[b,j]
            contrib = x.unsqueeze(1) * w.unsqueeze(0)

            # Flatten per-sample: [batch, out*in]
            batch_flat = contrib.view(contrib.size(0), -1)

            # Store for later
            self.activations[layer_name].append(batch_flat)
        return hook
    
    def compute_scores(self):
        """
        For each layer, concatenate all collected batches into [N_samples, num_weights],
        then perform weight-wise normalization (min-max per column) and compute std deviation.

        Returns:
            dict[layer_name] = torch.Tensor of shape [num_weights] with each weight's score
        """
        scores = {}
        eps = 1e-8
        for layer, batched_activations in self.activations.items():

            # Concatenate across all batches: shape [N, W]
            data = torch.cat(batched_activations, dim=0)  # N_samples x num_weights

            # Weight-wise min/max across samples (dim=0)
            min_vals = data.min(dim=0, keepdim=True)[0]  # 1 x W
            max_vals = data.max(dim=0, keepdim=True)[0]  # 1 x W

            # Normalize each weight's vector to [0,1]
            normed = (data - min_vals) / (max_vals - min_vals + eps)

            # Compute std across samples (dim=0): low std = low variability
            stds = normed.std(dim=0, unbiased=False)  # W
            scores[layer] = stds

        return scores
    
    def collect_activations(self, data_loader):
        """
        Collect activations for each batch in the data_loader.
        This will fill self.activations with per-layer contributions.
        """
        self.activations = defaultdict(list)
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in data_loader:
                self.model(inputs)

    def get_weight_indices(self, scores, num_prune, prune_type='random'):
        if prune_type == "structured":
            all_scores = torch.cat(list(scores.values()))
            all_scores[self.mask] = float('inf')
            sorted_idx = torch.argsort(all_scores)
            nonzero = sorted_idx[all_scores[sorted_idx] != 0]
            candidates = nonzero[:num_prune].tolist()

        else:
            candidates = random.sample(range(len(self.mask)), num_prune)

        return candidates
    
    def prune(self, indices):
        """
        Zero out the weights at the given global flat indices.
        
        Args:
            indices: list of global flat indices.
        """

        #Create a mask for the current indices to prune
        prune_mask = torch.zeros_like(self.mask, dtype=torch.bool)
        prune_mask[indices] = True

        # Apply prune mask
        for entry in self.prunable:
            module = entry['module']
            weight = module.weight.data
            weight[prune_mask[entry['offset']:entry['offset'] + entry['numel']].view_as(weight)] = 0.0

    def prune_weights(self, data_loader, num_prune, prune_type='random'):
        """
        Convenience method: find candidates and prune them in one call.
        """

        # Will store per-layer mini-batched lists of per-sample flattened contributions
        self.activations = defaultdict(list)

        # Register hooks to capture activation contributions
        self.handles = []
        self._register_hooks()

        # Collect activations from the data from the data_loader
        self.collect_activations(data_loader)
        self._remove_hooks()

        # Compute the variability score for each weight
        scores = self.compute_scores()
        self.prune_indices = self.get_weight_indices(scores, num_prune, prune_type)
        
        self.mask[self.prune_indices] = True
        self.prune(self.prune_indices)

    def freeze_pruned_gradients(self):
        """
        After backward(), zero gradients for pruned weights so they stay zero.
        """
        for entry in self.prunable:
            weight = entry['module'].weight
            if weight.grad is not None:
                grad_mask = weight.data.ne(0).float().to(weight.grad.device)
                weight.grad.mul_(grad_mask)