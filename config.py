import torch

# Configuration Dictionary as requested
GPT_CONFIG = {
    "vocab_size": 742,  # Will be updated based on tokenizer
    "context_length": 256,
    "emb_dim": 256,
    "n_heads": 8,
    "n_layers": 6,
    "drop_rate": 0.1,
    "qkv_bias": False,
    
    # Training configuration added to the same dict for simplicity
    "batch_size": 16,     # Reduced batch size to accommodate larger context/emb_dim
    "learning_rate": 3e-4,
    "max_iters": 500,     # Reduced iterations for the demonstration
    "eval_interval": 100,
    
    # Paths
    "corpus_path": "data/corpus.txt",
    "vocab_path": "tokenizer/vocab.json",
    "merges_path": "tokenizer/merges.json",
    "checkpoint_path": "checkpoints/model.pt",
    "plot_path": "plots/loss_curve.png",
    
    # Device
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}
