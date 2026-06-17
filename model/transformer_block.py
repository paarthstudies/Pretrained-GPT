import torch
import torch.nn as nn
from model.attention import MultiHeadAttention

class FeedForward(nn.Module):
    """
    Standard Transformer Multi-Layer Perceptron (MLP) block.
    
    1. What it does:
       It applies two linear transformations with a GELU activation in between.
       The hidden dimension is typically 4x the model dimension.
       
    2. Why it is needed:
       While attention handles the routing of information between different tokens,
       the feed-forward network processes each token individually, allowing the model
       to compute complex non-linear features from the attended information.
    """
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    """
    A single block of the Transformer Decoder architecture.
    
    1. What it does:
       Combines Layer Normalization, Multi-Head Attention, and a Feed-Forward Network
       with Residual (skip) connections.
       
    2. Why it is needed:
       Stacking multiple blocks allows the network to learn hierarchical representations.
       The residual connections help gradients flow during backpropagation, mitigating
       the vanishing gradient problem in deep networks.
       Layer norm stabilizes the activations.
       
    3. Input shape:
       x: (batch_size, context_length, d_model)
       
    4. Output shape:
       (batch_size, context_length, d_model)
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = FeedForward(config)

    def forward(self, x):
        # Pre-LayerNorm architecture (used in GPT-2 and GPT-3)
        # x + Sublayer(LayerNorm(x))
        
        # 1. Attention part with residual connection
        x = x + self.attn(self.ln_1(x))
        
        # 2. Feed-forward part with residual connection
        x = x + self.mlp(self.ln_2(x))
        
        return x
