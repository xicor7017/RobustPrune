import time
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear


class Pruner:
    """
    Prunes weights based on weight-wise activation variability.

    Changes from the original file:
    - We DO NOT hook `out_proj` (or any other Linear) inside `nn.MultiheadAttention` directly.
      Instead, we hook the *parent* `nn.MultiheadAttention` module and manually compute the
      per-weight contributions for its `out_proj` using the tensors available in the hook.
    - All public method signatures and return values are preserved.
    - Linear layers that belong to an MHA block are still prunable, but they are not double-hooked.
    """

    def __init__(self, model: nn.Module):
        self.model = model

        # ------------------------------------------------------------------
        # Identify modules
        # ------------------------------------------------------------------
        self.prunable = []               # list of dicts with module info
        self._mha_names = set()          # names of all nn.MultiheadAttention modules
        self._linears_in_mha = set()     # names of Linear/NDQL that are children of an MHA

        # First pass: record all MHA module names
        for name, module in model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                self._mha_names.add(name)

        # Helper to check if a module name is under an MHA by prefix match
        def _is_child_of_mha(mod_name: str) -> bool:
            return any(mod_name.startswith(mha_name + '.') for mha_name in self._mha_names)

        offset = 0
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, NonDynamicallyQuantizableLinear)):
                if _is_child_of_mha(name):
                    self._linears_in_mha.add(name)
                weight = module.weight
                numel = weight.numel()
                self.prunable.append({
                    'name': name,
                    'module': module,
                    'shape': weight.shape,
                    'numel': numel,
                    'offset': offset,
                })
                offset += numel

        print(f"Found a total of {offset} prunable weights.")

        # Global flat mask
        self.mask = torch.zeros(offset, dtype=torch.bool)

        # Runtime containers
        self.handles = []
        self.activations = defaultdict(list)   # layer_name -> [batch_flat tensors]
        self.prune_indices = []

    # ----------------------------------------------------------------------
    # Hook registration / removal
    # ----------------------------------------------------------------------
    def _register_hooks(self):
        """Attach hooks to (1) parent MHA blocks, (2) Linear layers NOT inside an MHA."""
        # 1) Hook all MHA parents
        for name, module in self.model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                h = module.register_forward_hook(self._make_mha_hook(name))
                self.handles.append(h)

        # 2) Hook all other linears (that are not children of an MHA)
        allowed_linears = (nn.Linear, NonDynamicallyQuantizableLinear)
        for name, module in self.model.named_modules():
            if isinstance(module, allowed_linears) and name not in self._linears_in_mha:
                h = module.register_forward_hook(self._make_linear_hook(name))
                self.handles.append(h)

    def _remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    # ----------------------------------------------------------------------
    # Hook makers
    # ----------------------------------------------------------------------
    def _make_linear_hook(self, layer_name):
        """Standard Linear hook: capture x * w contributions."""
        def hook(module, inputs, outputs):
            x = inputs[0].detach()
            x_flat = x.reshape(-1, x.shape[-1])
            w = module.weight.detach()
            contrib = x_flat.unsqueeze(1) * w.unsqueeze(0)  # [N, out, in]
            batch_flat = contrib.view(contrib.size(0), -1)
            self.activations[layer_name].append(batch_flat)
        return hook

    def _make_mha_hook(self, layer_name):
        """
        Hook for nn.MultiheadAttention parent module.

        We reconstruct per-weight contributions for the *out_proj* weight using the inputs/outputs
        that are available to the parent module.

        PyTorch's internal implementation calls F.linear(attn_output, out_proj.weight...), bypassing
        the Linear module forward. In the parent hook we get:
            inputs:  (query, key, value, ...)  (tuple)
            outputs: (attn_output, attn_weights) or (attn_output,) depending on need_weights

        We cannot directly access the pre-projection tensor (the one fed into F.linear), but we *can*
        recompute the per-weight contributions using the same formula if we also get that tensor.

        Strategy:
        - Re-run a minimal piece of code to get the pre-projection tensor by calling
          torch.nn.functional.multi_head_attention_forward with `need_weights=False` and capturing
          the returned attn_output BEFORE applying out_proj.
        - BUT F.multi_head_attention_forward itself already applies the projection, so we can't grab
          it mid-way. Instead, we abuse the fact that `out_proj.weight` is used at the very end and
          reconstruct x_pre via: x_pre = F.linear^{-1}(attn_output, W, b). This inverse is not exact
          unless W is square & invertible. To avoid instability, we approximate contributions using
          the *attn_output* (post-projection) as the "input" proxy. This keeps relative variance info
          useful for pruning, although it's not mathematically identical.

        If you later want the exact pre-proj tensor, patching or FX is required. For now we follow the
        user's preference to avoid patching and keep forward signatures intact.
        """
        def hook(module: nn.MultiheadAttention, inputs, outputs):
            # outputs[0]: attn_output AFTER projection
            attn_out = outputs[0].detach()  # shape: [L, N, E] or [N, L, E] (depends on batch_first)
            # Flatten last dim is embed_dim (= in_features for out_proj)
            x_flat = attn_out.reshape(-1, attn_out.shape[-1])

            w = module.out_proj.weight.detach()  # [out_features, in_features]
            contrib = x_flat.unsqueeze(1) * w.unsqueeze(0)  # [samples, out, in]
            batch_flat = contrib.view(contrib.size(0), -1)

            # Store under the out_proj's logical name so downstream code is unchanged
            proj_name = f"{layer_name}.out_proj"
            self.activations[proj_name].append(batch_flat)
        return hook

    # ----------------------------------------------------------------------
    # Data collection and scoring
    # ----------------------------------------------------------------------
    def collect_activations(self, data_loader):
        self.activations.clear()
        self.model.eval()
        with torch.no_grad():
            for inputs, _ in data_loader:
                self.model(inputs)

    def compute_scores(self):
        scores = {}
        eps = 1e-8
        for layer, batched in self.activations.items():
            data = torch.cat(batched, dim=0)          # [N, W]
            min_vals = data.min(dim=0, keepdim=True)[0]
            max_vals = data.max(dim=0, keepdim=True)[0]
            normed = (data - min_vals) / (max_vals - min_vals + eps)
            stds = normed.std(dim=0, unbiased=False)  # [W]
            scores[layer] = stds
        return scores

    # ----------------------------------------------------------------------
    # Selecting and applying pruning
    # ----------------------------------------------------------------------
    def get_weight_indices(self, scores, num_prune, prune_type='random'):
        if prune_type == 'structured':
            all_scores = torch.cat(list(scores.values()))
            all_scores[self.mask] = float('inf')
            all_scores[all_scores<=0.0] = float('inf')
            sorted_idx = torch.argsort(all_scores)
            candidates = sorted_idx[:num_prune].tolist()
        else:
            candidates = random.sample(range(len(self.mask)), num_prune)
        return candidates

    def prune(self, indices):
        prune_mask = torch.zeros_like(self.mask)
        prune_mask[indices] = True
        for entry in self.prunable:
            module = entry['module']
            w = module.weight.data
            local = prune_mask[entry['offset']: entry['offset'] + entry['numel']].view_as(w)
            w[local] = 0.0

    def prune_weights(self, data_loader, num_prune, prune_type='random'):
        # Capture activations
        self._register_hooks()
        self.collect_activations(data_loader)
        self._remove_hooks()

        # Score and select
        scores = self.compute_scores()
        self.prune_indices = self.get_weight_indices(scores, num_prune, prune_type)
        self.mask[self.prune_indices] = True
        self.prune(self.prune_indices)

    # ----------------------------------------------------------------------
    # Training-time gradient freezing
    # ----------------------------------------------------------------------
    def freeze_pruned_gradients(self):
        for entry in self.prunable:
            weight = entry['module'].weight
            if weight.grad is not None:
                keep = (~self.mask[entry['offset']: entry['offset'] + entry['numel']]).view_as(weight)
                weight.grad.mul_(keep.float().to(weight.grad.device))
