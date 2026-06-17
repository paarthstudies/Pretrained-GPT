import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config["emb_dim"] % config["n_heads"] == 0, "emb_dim must be divisible by n_heads"
        
        self.emb_dim = config["emb_dim"]
        self.n_heads = config["n_heads"]
        self.head_dim = config["emb_dim"] // config["n_heads"]
        self.context_length = config["context_length"]
        self.drop_rate = config.get("drop_rate", 0.0)
        self.qkv_bias = config.get("qkv_bias", False)
        
        # Linear projections for Query, Key, and Value
        self.qkv_proj = nn.Linear(self.emb_dim, 3 * self.emb_dim, bias=self.qkv_bias)
        
        # Output projection
        self.out_proj = nn.Linear(self.emb_dim, self.emb_dim, bias=self.qkv_bias)
        
        # Dropout
        self.attn_drop = nn.Dropout(self.drop_rate)
        self.resid_drop = nn.Dropout(self.drop_rate)
        
        # Causal mask (lower triangular matrix of ones)
        self.register_buffer(
            "bias", 
            torch.tril(torch.ones(self.context_length, self.context_length))
            .view(1, 1, self.context_length, self.context_length)
        )
        
    def forward(self, x):
        B, T, C = x.size() # Batch, Time (context_length), Channels (emb_dim)
        
        # 1. Project input to Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.emb_dim, dim=2)
        
        # 2. Reshape for multi-head attention
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 3. Compute Attention Scores (Scaled Dot-Product)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # 4. Apply Causal Mask
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        
        # 5. Softmax to get probabilities
        att = F.softmax(att, dim=-1)
        
        # Apply attention dropout
        # Why: Prevents the model from heavily focusing on the same exact tokens across all layers.
        att = self.attn_drop(att)
        
        # 6. Weighted sum of values
        y = att @ v
        
        # 7. Concatenate heads back to emb_dim
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # 8. Final linear projection
        out = self.out_proj(y)
        
        # Apply residual dropout
        # Why: Regularizes the output before being added back into the residual stream.
        out = self.resid_drop(out)
        
        return out
