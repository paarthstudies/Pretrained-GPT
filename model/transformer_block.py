import torch
import torch.nn as nn
from model.attention import MultiHeadAttention

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config["emb_dim"], 4 * config["emb_dim"]),
            nn.GELU(),
            nn.Linear(4 * config["emb_dim"], config["emb_dim"]),
            nn.Dropout(config.get("drop_rate", 0.0)) # Add dropout inside MLP
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config["emb_dim"])
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config["emb_dim"])
        self.mlp = FeedForward(config)

    def forward(self, x):
        # 1. Attention part with residual connection
        x = x + self.attn(self.ln_1(x))
        
        # 2. Feed-forward part with residual connection
        x = x + self.mlp(self.ln_2(x))
        
        return x
