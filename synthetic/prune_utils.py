import time
import random
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear


class Pruner:
    """
    Pruner: A class to perform activation variability-based pruning of model weights.

    This class tracks and prunes the following weights based on their relative activation variance:
      1. Standalone Linear layers (nn.Linear, NonDynamicallyQuantizableLinear) outside MultiheadAttention (MHA).
      2. MultiheadAttention internal weights:
         - in_proj_weight (combined Query, Key, Value projection)
         - out_proj.weight (final attention output projection)

    Pruning Workflow:
      a. Register forward hooks to capture per-weight contributions (excluding biases).
      b. Run a forward pass over a data_loader to collect activations.
      c. Remove hooks to avoid side effects.
      d. Compute relative variance scores: variance / (mean + epsilon).
      e. Select lowest-scoring weights (structured) or random weights.
      f. Update a global mask and zero out selected weights in place.
      g. Optionally freeze gradients of pruned weights during backprop.

    Attributes:
        model (nn.Module): The PyTorch model under pruning.
        prunable (list of dict): Metadata entries for each prunable weight tensor.
        mask (torch.BoolTensor): Global flat mask indicating pruned weights.
        activations (defaultdict(list)): Collected activation tensors per layer key.
        prune_indices (list[int]): Indices chosen for pruning in the last operation.
        handles (list): Registered forward hook handles.
    """

    def __init__(self, model: nn.Module):
        """
        Initialize the Pruner by discovering all prunable weights and setting up masks.

        Args:
            model (nn.Module): The neural network model to prune.

        Side-effects:
            - Populates self.prunable with entries for each weight tensor and its flat offset.
            - Initializes self.mask to all False (no weights pruned initially).
        """
        self.model = model

        # Detect all MultiheadAttention modules by name for exclusion/inclusion logic
        self._mha_names = {
            name for name, m in model.named_modules()
            if isinstance(m, nn.MultiheadAttention)
        }

        # Build prunable list, tracking offsets into a global flat parameter vector
        self.prunable = []
        offset = 0

        def is_under_mha(name: str) -> bool:
            """Return True if a module name is a child of any MHA block."""
            return any(name.startswith(mha + ".") for mha in self._mha_names)

        # 1) Standalone Linear / NDQL layers outside MHA
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, NonDynamicallyQuantizableLinear)):
                if is_under_mha(name):
                    # Skip linears that are part of an MHA block
                    continue
                w = module.weight
                n = w.numel()
                self.prunable.append({
                    'name': name,
                    'module': module,
                    'param_name': 'weight',
                    'shape': tuple(w.shape),
                    'numel': n,
                    'offset': offset,
                })
                offset += n

        # 2) MHA internal weights: in_proj_weight and out_proj.weight
        for name, module in model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                # Combined QKV projection weight
                w_qkv = module.in_proj_weight
                n_qkv = w_qkv.numel()
                self.prunable.append({
                    'name': f"{name}:in_proj_weight",
                    'module': module,
                    'param_name': 'in_proj_weight',
                    'shape': tuple(w_qkv.shape),
                    'numel': n_qkv,
                    'offset': offset,
                })
                offset += n_qkv

                # Output projection weight
                w_out = module.out_proj.weight
                n_out = w_out.numel()
                self.prunable.append({
                    'name': f"{name}:out_proj",
                    'module': module,
                    'param_name': 'out_proj.weight',
                    'shape': tuple(w_out.shape),
                    'numel': n_out,
                    'offset': offset,
                })
                offset += n_out

        # Initialize pruning mask: False => keep, True => pruned
        print(f"Found a total of {offset} prunable weights.", end='\n\n')
        self.mask = torch.zeros(offset, dtype=torch.bool)

        # Runtime containers for hooks and collected activations
        self.handles = []
        self.activations = defaultdict(list)
        self.prune_indices = []

    def _register_hooks(self):
        """
        Attach forward hooks to collect per-weight contributions.

        - Hooks on MHA parents to capture both in_proj and out_proj contributions.
        - Hooks on standalone Linear modules outside MHA.
        """
        # Hook MHA modules
        for name, module in self.model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                h = module.register_forward_hook(
                    self._make_mha_hook(name, module)
                )
                self.handles.append(h)

        # Hook standalone linears
        for entry in self.prunable:
            if entry['param_name'] == 'weight':
                m = entry['module']
                h = m.register_forward_hook(
                    self._make_linear_hook(entry['name'])
                )
                self.handles.append(h)

    def _remove_hooks(self):
        """Remove all registered forward hooks to restore original model behavior."""
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def _make_linear_hook(self, layer_name: str):
        """
        Create a forward hook for a standalone Linear module.

        Args:
            layer_name (str): Key under which to store collected activations.

        Returns:
            hook (callable): Forward hook capturing x*w contributions.
        """
        def hook(module, inputs, outputs):
            # inputs[0]: the input tensor x of shape [..., in_features]
            x = inputs[0].detach()
            x_flat = x.reshape(-1, x.shape[-1])  # [N, in]
            w = module.weight.detach()          # [out, in]
            # contribution[b,i,j] = x[b, j] * w[i, j]
            contrib = x_flat.unsqueeze(1) * w.unsqueeze(0)  # [N, out, in]
            batch_flat = contrib.view(contrib.size(0), -1)  # [N, out*in]
            self.activations[layer_name].append(batch_flat)

        return hook

    def _make_mha_hook(self, mha_name: str, module: nn.MultiheadAttention):
        """
        Create a forward hook for nn.MultiheadAttention to capture:
          1) in_proj_weight contributions from Q, K, V projections.
          2) out_proj.weight contributions from the post-attention output.

        Args:
            mha_name (str): Unique identifier for this MHA module.
            module (nn.MultiheadAttention): The module instance.

        Returns:
            hook (callable): Forward hook for MHA.
        """
        def hook(mod, inputs, outputs):
            # 1) Q/K/V projection contributions
            q, k, v = inputs[0].detach(), inputs[1].detach(), inputs[2].detach()
            E = module.embed_dim
            # Flatten batch/spatial dims to a single N for each
            qf = q.reshape(-1, E)
            kf = k.reshape(-1, E)
            vf = v.reshape(-1, E)

            # in_proj_weight shape: [3*E, E]
            w_qkv = module.in_proj_weight.detach()
            w_q, w_k, w_v = w_qkv.chunk(3, dim=0)  # each [E, E]

            # Compute per-weight contributions and concatenate
            cq = qf.unsqueeze(1) * w_q.unsqueeze(0)
            ck = kf.unsqueeze(1) * w_k.unsqueeze(0)
            cv = vf.unsqueeze(1) * w_v.unsqueeze(0)
            c_qkv = torch.cat([cq, ck, cv], dim=1)  # [N, 3E, E]
            flat_qkv = c_qkv.view(c_qkv.size(0), -1)  # [N, 3E*E]
            self.activations[f"{mha_name}:in_proj_weight"].append(flat_qkv)

            # 2) out_proj contributions
            attn_out = outputs[0].detach()
            batch_first = getattr(module, "batch_first", False)
            if batch_first:
                oflat = attn_out.reshape(-1, E)  # [N*L, E]
            else:
                oflat = attn_out.permute(1, 0, 2).reshape(-1, E)

            w_out = module.out_proj.weight.detach()  # [out, E]
            contrib_out = oflat.unsqueeze(1) * w_out.unsqueeze(0)  # [N*L, out, E]
            flat_out = contrib_out.view(contrib_out.size(0), -1)   # [N*L, out*E]
            self.activations[f"{mha_name}:out_proj"].append(flat_out)

        return hook

    def collect_activations(self, data_loader):
        """
        Run a single forward sweep over data_loader in eval mode to populate self.activations.

        Args:
            data_loader (Iterable): Yields (inputs, labels) tuples; labels are ignored.
        """
        self.activations.clear()
        self.model.eval()
        with torch.no_grad():
            for inputs, _ in data_loader:
                self.model(inputs)

    def compute_scores(self):
        """
        Compute relative variance scores for each prunable weight.

        Returns:
            Dict[str, torch.Tensor]: layer_name -> 1D tensor of relative variances.
        """
        eps = 1e-8
        scores = {}
        for layer, batches in self.activations.items():
            data = torch.cat(batches, dim=0)           # [N, W]
            means = data.mean(dim=0).abs() + eps       # [W]
            vars_ = data.var(dim=0, unbiased=False)    # [W]
            scores[layer] = vars_ / means              # [W]
        return scores

    def get_weight_indices(self, scores, num_prune, prune_type='random'):
        """
        Select flat weight indices to prune from the global mask space.

        Args:
            scores (dict): Mapping layer_name -> 1D relative-variance tensor.
            num_prune (int): Number of weights to prune this round.
            prune_type (str): 'structured' uses lowest scores; otherwise random.

        Returns:
            List[int]: Flat indices into the global parameter vector.
        """
        if prune_type == 'structured':
            # Concatenate in the same order as self.prunable
            all_scores = torch.cat([scores[e['name']] for e in self.prunable])
            all_scores[self.mask] = float('inf')
            all_scores[all_scores <= 0] = float('inf')
            idx = torch.argsort(all_scores)
            return idx[:num_prune].tolist()
        else:
            return random.sample(range(self.mask.numel()), num_prune)

    def prune(self, indices):
        """
        Zero out the weights at specified global indices.

        Args:
            indices (List[int]): Flat indices to prune (zero-in).
        """
        pmask = torch.zeros_like(self.mask)
        pmask[indices] = True

        # Apply mask to each parameter tensor using its offset
        for e in self.prunable:
            # Resolve tensor by traversing param_name
            names = e['param_name'].split('.')
            tensor = e['module']
            for n in names:
                tensor = getattr(tensor, n)
            w_flat = tensor.data.view(-1)
            segment = pmask[e['offset']: e['offset'] + e['numel']]
            w_flat[segment] = 0.0

    def prune_weights(self, data_loader, num_prune, prune_type='random'):
        """
        High-level pruning API: collect activations, score, select, and prune.

        Args:
            data_loader (Iterable): Used to collect activations.
            num_prune (int): How many weights to prune.
            prune_type (str): 'structured' or 'random'.
        """
        self._register_hooks()
        self.collect_activations(data_loader)
        self._remove_hooks()

        scores = self.compute_scores()
        self.prune_indices = self.get_weight_indices(scores, num_prune, prune_type)
        self.mask[self.prune_indices] = True
        self.prune(self.prune_indices)

    def freeze_pruned_gradients(self):
        """
        Zero out gradients for pruned weights after backprop.
        Should be called after loss.backward() in training loop.
        """
        for e in self.prunable:
            names = e['param_name'].split('.')
            tensor = e['module']
            for n in names:
                tensor = getattr(tensor, n)
            if tensor.grad is not None:
                keep = (~self.mask[e['offset']: e['offset'] + e['numel']]).view_as(tensor)
                tensor.grad.mul_(keep.float().to(tensor.grad.device))
