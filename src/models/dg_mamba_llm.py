import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import wandb

from modules.dg_mamba2 import DGMamba2
from modules.mlp import GatedMLP, MLP

class DGBlock(nn.Module):
    def __init__(self, config, id):
        super().__init__()
        self.id = id
        self.config = config
        self.d_model = config.d_model
        self.mlp_factor = config.mlp_factor
        self.device = config.device
        self.dtype = config.dtype
        factory_kwargs = {"device": self.device, "dtype": self.dtype}

        self.mixer = DGMamba2(config, id, **factory_kwargs)
        
        if self.config.layernorm:
            self.norm = nn.LayerNorm(self.d_model, bias=False, **factory_kwargs)
            self.norm2 = nn.LayerNorm(self.d_model, bias=False, **factory_kwargs)
            
        if not self.config.no_mlp:
            if self.config.gate:
                self.mlp = GatedMLP(config, id, self.d_model, factor=self.mlp_factor, **factory_kwargs)
            else:
                self.mlp = MLP(config, id, self.d_model, factor=self.mlp_factor, **factory_kwargs)

    def configure_dg(self, **kwargs):
        self.mixer.configure_dg(**kwargs)

    def forward(self, hidden_states, save_weights=False, check_conditions=False):
        # Keep original forward pass exactly the same
        residual = hidden_states
        if self.config.layernorm:
            hidden_states = self.norm(hidden_states).to(self.dtype)
        hidden_states = self.mixer(hidden_states, save_weights=save_weights, check_conditions=check_conditions)

        if save_weights and self.config.wandb:
            print("hidden states 1:")
            print(hidden_states[0,:30])

        residual = hidden_states + residual

        if save_weights and self.config.wandb:
            print("residual 1:")
            print(residual[0,:30])

        if self.config.layernorm:
            hidden_states = self.norm2(residual).to(self.dtype)
        else:
            hidden_states = residual
            
        if not self.config.no_mlp:
            hidden_states = self.mlp(hidden_states, save_weights=save_weights)

            if save_weights and self.config.wandb:
                print("hidden states 2:")
                print(hidden_states[0,:30])
            
            hidden_states = hidden_states + residual

        if save_weights and self.config.wandb:
            print("hidden states 3:")
            print(hidden_states[0,:30])
        
        return hidden_states

class DGMambaLMHeadModel(nn.Module):
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.dtype = config.dtype
        d_model = config.d_model
        n_layer = config.n_layer
        vocab_size = config.vocab_size
        factory_kwargs = {"device": self.device, "dtype": self.dtype}

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)
        self.layers = nn.ModuleList([DGBlock(config, i) for i in range(n_layer)])
        if self.config.layernorm:
            self.norm_f = nn.LayerNorm(d_model, bias=False, **factory_kwargs)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)

        # Keep original initialization
        self.apply(self._init_weights)

        if self.config.fix_conv:
            for mn, m in self.named_modules():
                for pn, p in m.named_parameters():
                    if pn.endswith('conv1d.weight'):
                        with torch.no_grad():
                            ker = torch.Tensor([2**i for i in range(p.size(2))])
                            ker = ker.repeat(p.size(0),1,1)
                            p.copy_(ker)
                    elif pn.endswith('conv1d.bias'):
                        with torch.no_grad():
                            p.fill_(1.0)
    
    def _init_weights(self, module, initializer_range=0.02, n_residuals_per_layer=2):
        # Keep original weight initialization
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=initializer_range)
        
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * self.config.n_layer)
    
    def configure_dg(self, **kwargs):
        """Configure DG parameters for all blocks"""
        for layer in self.layers:
            layer.configure_dg(**kwargs)

    # Keep all other original methods exactly the same
    def tie_weights(self):
        self.lm_head.weight = self.embedding.weight

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def get_parameter_group_specs(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear)
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding, nn.Conv1d)
        
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
                if hasattr(p, '_no_weight_decay'):
                    if p._no_weight_decay:
                        no_decay.add(fpn)
                elif "ker" in pn:
                    no_decay.add(fpn)
                elif pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        return [
            {"params": sorted(list(decay))},
            {"params": sorted(list(no_decay)), "weight_decay": 0.0},
        ]
    
    def forward(self, idx, targets, save_weights=False, check_conditions=False):
        # Keep original forward pass
        if save_weights:
            print("Input sequence:")
            print(idx[0,:30])
        hidden_states = self.embedding(idx)
        if save_weights:
            print("Embedded input:")
            print(hidden_states[0,:30])
        for layer in self.layers:
            hidden_states = layer(hidden_states, save_weights=save_weights, check_conditions=check_conditions)
        if self.config.layernorm:
            hidden_states = self.norm_f(hidden_states).to(self.dtype)
        logits = self.lm_head(hidden_states)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        if save_weights and self.config.wandb:
            print("logits:")
            print(logits[0,:30])
            print("probs:")
            print(F.softmax(logits[0,:30], dim=-1))
            print("emb-final:")
            print(self.embedding.weight)
            wandb.log({"emb-final": wandb.Image(self.embedding.weight.numpy(force=True))})
            print("dot product:")
            print(self.embedding.weight[0] @ self.embedding.weight[1])

        return {'logits': logits, 'loss': loss}