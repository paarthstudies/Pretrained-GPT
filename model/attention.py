import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism with causal masking.
    
    1. What it does:
       It allows the model to attend to different parts of the sequence simultaneously.
       It uses multiple "heads" to learn different types of relationships between tokens.
       
    2. Why it is needed:
       Different words have different relationships. For example, a word might relate
       to its adjective in one way, and its verb in another. Multi-head attention
       captures these diverse relationships.
       
    3. Input shape:
       x: (batch_size, context_length, d_model)
       
    4. Output shape:
       (batch_size, context_length, d_model)
       
    5. Relation to real GPT models:
       This is the exact same self-attention mechanism used in GPT models (like GPT-2 and GPT-3),
       including the causal mask that prevents the model from "looking ahead" at future tokens.
    """
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.context_length = config.context_length
        
        # Linear projections for Query, Key, and Value
        # We combine them into one matrix for efficiency
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model)
        
        # Output projection
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        
        # Causal mask (lower triangular matrix of ones)
        # Using register_buffer so it's moved to the correct device but not treated as a trainable parameter
        self.register_buffer(
            "bias", 
            torch.tril(torch.ones(config.context_length, config.context_length))
            .view(1, 1, config.context_length, config.context_length)
        )
        
    def forward(self, x):
        B, T, C = x.size() # Batch, Time (context_length), Channels (d_model)
        
        # 1. Project input to Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        
        # 2. Reshape for multi-head attention: (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 3. Compute Attention Scores (Scaled Dot-Product)
        # (B, n_heads, T, head_dim) @ (B, n_heads, head_dim, T) -> (B, n_heads, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # 4. Apply Causal Mask
        # We fill the upper triangular part (where bias is 0) with -inf
        # so that softmax will output 0 probability for future tokens.
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        
        # 5. Softmax to get probabilities
        att = F.softmax(att, dim=-1)
        
        # 6. Weighted sum of values
        # (B, n_heads, T, T) @ (B, n_heads, T, head_dim) -> (B, n_heads, T, head_dim)
        y = att @ v
        
        # 7. Concatenate heads back to d_model: (B, T, n_heads, head_dim) -> (B, T, d_model)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # 8. Final linear projection
        out = self.out_proj(y)
        
        return out
