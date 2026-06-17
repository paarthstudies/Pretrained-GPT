import torch
from dataclasses import dataclass

@dataclass
class Config:
    # Model configuration
    vocab_size: int = 281  # Will be updated based on tokenizer
    context_length: int = 64
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    
    # Training configuration
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_iters: int = 1000
    eval_interval: int = 100
    
    # Paths
    corpus_path: str = "data/corpus.txt"
    vocab_path: str = "tokenizer/vocab.json"
    merges_path: str = "tokenizer/merges.json"
    checkpoint_path: str = "checkpoints/model.pt"
    plot_path: str = "plots/loss_curve.png"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

config = Config()
