import torch
import torch.nn as nn
from model.transformer_block import TransformerBlock

class TinyGPT(nn.Module):
    """
    The full Decoder-Only Transformer Language Model.
    
    1. What it does:
       Takes token IDs, converts them to embeddings, adds positional information,
       processes them through multiple transformer blocks, and predicts the next token
       probabilities (logits) for every position.
       
    2. Why it is needed:
       This is the complete system that coordinates all the sub-components to perform
       autoregressive language modeling.
       
    3. Input shape:
       idx: (batch_size, context_length) - Tensor of integer token IDs
       
    4. Output shape:
       logits: (batch_size, context_length, vocab_size) - Unnormalized predictions
       
    5. Relation to real GPT models:
       This mirrors the exact architecture of GPT-2, scaled down. It includes
       token embeddings, learned positional embeddings, a stack of transformer blocks,
       a final layer norm, and a linear language modeling head.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token Embedding: Maps token IDs to d_model dimensional vectors
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Positional Embedding: Maps positions (0 to context_length-1) to d_model dimensional vectors
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        
        # Stack of Transformer Blocks
        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layers)])
        
        # Final Layer Normalization
        self.ln_f = nn.LayerNorm(config.d_model)
        
        # Language Modeling Head: Maps from d_model back to vocab_size to get logits
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying: In many GPT models, the token embedding weights and LM head weights are shared
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
        assert T <= self.config.context_length, f"Cannot forward sequence of length {T}, block size is only {self.config.context_length}"
        
        # Generate position indices [0, 1, ..., T-1]
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        
        # 1. Embed tokens
        tok_emb = self.token_embedding(idx) # (B, T, d_model)
        
        # 2. Embed positions
        pos_emb = self.position_embedding(pos) # (T, d_model)
        
        # 3. Add token and positional embeddings
        x = tok_emb + pos_emb # (B, T, d_model)
        
        # 4. Pass through transformer blocks
        x = self.blocks(x) # (B, T, d_model)
        
        # 5. Apply final layer norm
        x = self.ln_f(x) # (B, T, d_model)
        
        # 6. Project to vocabulary logits
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        # 7. Compute loss if targets are provided
        loss = None
        if targets is not None:
            # Flatten predictions and targets to compute cross entropy
            # logits: (B*T, vocab_size), targets: (B*T)
            B, T, C = logits.shape
            logits_flat = logits.view(B*T, C)
            targets_flat = targets.view(B*T)
            
            import torch.nn.functional as F
            loss = F.cross_entropy(logits_flat, targets_flat)
            
        return logits, loss
