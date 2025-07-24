import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerPrunableEncoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, num_classes, max_seq_len):
        super().__init__()
        # Input projection + positional embeddings
        
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.00,
            #batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)
        

        '''
        # Input projection + positional embeddings
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model))

        # build one encoder layer…
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            #batch_first=True
        )

        # ← override the .out_proj to a “plain” Linear
        proj = encoder_layer.self_attn.out_proj
        new_proj = nn.Linear(proj.in_features, proj.out_features, bias=(proj.bias is not None))
        # (optional) copy existing weights/bias if you care:
        new_proj.weight.data.copy_(proj.weight.data)
        if proj.bias is not None:
            new_proj.bias.data.copy_(proj.bias.data)
        encoder_layer.self_attn.out_proj = new_proj

        # …then stack them
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Linear(d_model, num_classes)
        '''

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        if len(x.shape) != 3:
            x = x.unsqueeze(1)  # Ensure input is (batch, seq_len, input_dim)

        batch_size, seq_len, _ = x.size()
        # Embed + add positional encoding
        embed = self.embedding(x) + self.pos_embedding[:, :seq_len, :]
        # Transformer expects (seq_len, batch, d_model)
        out = self.transformer(embed.transpose(0,1))
        out = out.transpose(0,1)  # (batch, seq_len, d_model)
        # Mean-pool over sequence
        pooled = out.mean(dim=1)
        logits = self.classifier(pooled)
        return logits

    def _get_module_by_name(self, name):
        for n, m in self.named_modules():
            if n == name:
                return m
        raise ValueError(f"No module named '{name}'")

    def collect_all_activations(self, x):
        """
        Runs a forward pass on x and collects, for each Linear layer, the batch of inputs
        to that layer. Returns a dict mapping module names to input tensors.
        """
        layer_inputs = {}
        handles = []
        # Hook to capture inputs to each Linear
        def hook_fn(name):
            def fn(module, inp, out):  # inp is a tuple
                layer_inputs[name] = inp[0].detach()
            return fn
        # Register hooks
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                handles.append(module.register_forward_hook(hook_fn(name)))
        # Forward pass
        _ = self.forward(x)
        # Remove hooks
        for h in handles:
            h.remove()
        return layer_inputs

    def find_prune_candidates(self, train_loader, num_prune, prune_type="structured"):  # prune_type: "structured" or "random"
        # Accumulate activation stds for each weight-row across the dataset
        stats = []  # list of (layer_name, neuron_idx, std_value)
        device = next(self.parameters()).device
        for x, _ in train_loader:
            x = x.to(device)
            inputs = self.collect_all_activations(x)
            for name, inp in inputs.items():
                # inp shape: (batch, seq_len, in_dim) or (batch, in_dim)
                flat = inp.reshape(-1, inp.size(-1)).to(device)  # (N, in_dim)
                W = self._get_module_by_name(name).weight  # (out_dim, in_dim)
                # Activation matrix: (N, out_dim)
                acts = flat @ W.t()
                # Compute std per output neuron
                stds = acts.std(dim=0)  # (out_dim,)
                for idx in range(stds.size(0)):
                    stats.append((name, idx, stds[idx].item()))
        # Select candidates
        if prune_type == "structured":
            # Sort ascending by std and take lowest
            stats_sorted = sorted(stats, key=lambda x: x[2])
            selected = stats_sorted[:num_prune]
        else:
            # Random selection
            selected = [stats[i] for i in torch.randperm(len(stats))[:num_prune]]
        # Return list of (layer_name, neuron_idx)
        return [(name, idx) for name, idx, _ in selected]

    def prune_weights(self, train_loader, num_prune, prune_type="structured"):  # zeros out selected weights
        candidates = self.find_prune_candidates(train_loader, num_prune, prune_type)
        for name, idx in candidates:
            module = self._get_module_by_name(name)
            # Zero out the entire weight-row corresponding to the neuron
            module.weight.data[idx] = 0.0
