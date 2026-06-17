import torch
import torch.nn as nn
from model.transformer_block import TransformerBlock

class TinyGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token Embedding
        self.token_embedding = nn.Embedding(config["vocab_size"], config["emb_dim"])
        
        # Positional Embedding
        self.position_embedding = nn.Embedding(config["context_length"], config["emb_dim"])
        
        # Embedding Dropout
        # Why: Regularizes initial representations.
        self.emb_drop = nn.Dropout(config.get("drop_rate", 0.0))
        
        # Stack of Transformer Blocks
        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config["n_layers"])])
        
        # Final Layer Normalization
        self.ln_f = nn.LayerNorm(config["emb_dim"])
        
        # Language Modeling Head
        self.lm_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)
        
        # Weight tying
        self.token_embedding.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config["context_length"], f"Cannot forward sequence of length {T}, block size is only {self.config['context_length']}"
        
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        
        # 1. Embed tokens
        tok_emb = self.token_embedding(idx) # (B, T, emb_dim)
        
        # 2. Embed positions
        pos_emb = self.position_embedding(pos) # (T, emb_dim)
        
        # 3. Add token and positional embeddings
        x = tok_emb + pos_emb # (B, T, emb_dim)
        
        # Apply dropout to embeddings
        x = self.emb_drop(x)
        
        # 4. Pass through transformer blocks
        x = self.blocks(x) # (B, T, emb_dim)
        
        # 5. Apply final layer norm
        x = self.ln_f(x) # (B, T, emb_dim)
        
        # 6. Project to vocabulary logits
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        # 7. Compute loss if targets are provided
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B*T, C)
            targets_flat = targets.view(B*T)
            
            import torch.nn.functional as F
            loss = F.cross_entropy(logits_flat, targets_flat)
            
        return logits, loss
