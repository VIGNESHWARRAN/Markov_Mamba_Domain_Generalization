import torch
import torch.nn as nn
from .mamba2 import Mamba2

class DGMamba2(Mamba2):
    def __init__(self, config, id, **factory_kwargs):
        super().__init__(config, id, **factory_kwargs)
        self.dg_enabled = getattr(config, 'dg', False)
        self.suppress_lambda = getattr(config, 'suppress_lambda', 0.01)
        self.suppress_threshold = getattr(config, 'suppress_threshold', 0.1)
        self.min_active_states = getattr(config, 'min_active_states', 4)
        self.current_epoch = 0
        
    def configure_dg(self, enabled=None, suppress_lambda=None, 
                   suppress_threshold=None, min_active_states=None,
                   current_epoch=None):
        if enabled is not None:
            self.dg_enabled = enabled
        if suppress_lambda is not None:
            self.suppress_lambda = suppress_lambda
        if suppress_threshold is not None:
            self.suppress_threshold = suppress_threshold
        if min_active_states is not None:
            self.min_active_states = min_active_states
        if current_epoch is not None:
            self.current_epoch = current_epoch
        
    def apply_state_suppression(self, hidden_states):
        if not self.dg_enabled or self.current_epoch < getattr(self.config, 'dg_start_epoch', 0):
            return hidden_states
            
        # Original DG-Mamba suppression logic
        state_activations = torch.norm(hidden_states, dim=1)
        max_vals = state_activations.max(dim=-1, keepdim=True)[0].clamp(min=1e-6)
        batch_activations = state_activations / max_vals
        
        suppress_mask = (batch_activations < self.suppress_threshold).float()
        
        if self.min_active_states > 0:
            topk = torch.topk(batch_activations, 
                            k=min(self.min_active_states, batch_activations.size(-1)), 
                            dim=-1)
            min_threshold = topk.values[:, -1].unsqueeze(-1)
            suppress_mask = suppress_mask * (batch_activations < min_threshold).float()
        
        suppression_factor = 1.0 - self.suppress_lambda * suppress_mask.unsqueeze(1)
        return hidden_states * suppression_factor
        
    def forward(self, hidden_states, save_weights=False, check_conditions=False):
        out = super().forward(hidden_states, save_weights, check_conditions)
        if self.training and self.dg_enabled:
            out = self.apply_state_suppression(out)
        return out