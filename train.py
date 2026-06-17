import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import GPT_CONFIG
from dataset import get_dataloader
from model.tiny_gpt import TinyGPT

def print_model_statistics(config, model_params):
    print("\n--- Model Statistics ---")
    print(f"Vocabulary Size: {config['vocab_size']}")
    print(f"Context Length: {config['context_length']}")
    print(f"Embedding Dimension: {config['emb_dim']}")
    print(f"Heads: {config['n_heads']}")
    print(f"Layers: {config['n_layers']}")
    print(f"Parameters: {model_params / 1e6:.2f} Million")
    print(f"Device: {config['device'].upper()}")
    print("------------------------\n")

def main():
    config = GPT_CONFIG
    
    # Ensure directories exist
    os.makedirs("plots", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Prepare Data
    dataloader = get_dataloader(config)
    
    # Initialize Model
    model = TinyGPT(config)
    model.to(config["device"])
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Print the required statistics before training
    print_model_statistics(config, num_params)
    
    print(f"Dataset has {len(dataloader.dataset)} sequences.")
    print(f"Steps per epoch: {len(dataloader)}")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])

    # Training Loop
    model.train()
    losses = []
    steps_completed = 0
    pbar = tqdm(total=config["max_iters"], desc="Training")
    
    while steps_completed < config["max_iters"]:
        for x, y in dataloader:
            if steps_completed >= config["max_iters"]:
                break
                
            x, y = x.to(config["device"]), y.to(config["device"])
            
            logits, loss = model(x, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            steps_completed += 1
            
            pbar.update(1)
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
    pbar.close()
    
    print(f"Final training loss: {losses[-1]:.4f}")

    torch.save(model.state_dict(), config["checkpoint_path"])
    print(f"Saved model checkpoint to {config['checkpoint_path']}")

    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Training Loss", alpha=0.8)
    plt.xlabel("Training Steps")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Tiny GPT Training Dynamics")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(config["plot_path"])
    print(f"Saved loss curve plot to {config['plot_path']}")

if __name__ == "__main__":
    main()
