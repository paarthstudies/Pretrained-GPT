import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import config
from dataset import get_dataloader
from model.tiny_gpt import TinyGPT

def main():
    print(f"Using device: {config.device}")
    
    # Ensure directories exist
    os.makedirs("plots", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Prepare Data
    dataloader = get_dataloader(config)
    print(f"Dataset has {len(dataloader.dataset)} sequences.")
    print(f"Steps per epoch: {len(dataloader)}")
    
    # Initialize Model
    model = TinyGPT(config)
    model.to(config.device)
    
    # We want to trace the number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Optimizer (AdamW is standard for Transformers)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

    # Training Loop
    model.train()
    
    losses = []
    
    # To keep it quick and demonstrable, we train for max_iters or a few epochs
    steps_completed = 0
    pbar = tqdm(total=config.max_iters, desc="Training")
    
    while steps_completed < config.max_iters:
        for x, y in dataloader:
            if steps_completed >= config.max_iters:
                break
                
            x, y = x.to(config.device), y.to(config.device)
            
            # Forward pass: compute logits and cross entropy loss
            logits, loss = model(x, y)
            
            # Backward pass: compute gradients
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient Update: adjust weights
            optimizer.step()
            
            losses.append(loss.item())
            steps_completed += 1
            
            pbar.update(1)
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # (Optional) We could evaluate on a val set here if we had one
            
    pbar.close()
    
    print(f"Final training loss: {losses[-1]:.4f}")

    # Save checkpoint
    torch.save(model.state_dict(), config.checkpoint_path)
    print(f"Saved model checkpoint to {config.checkpoint_path}")

    # Plot and save loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Training Loss", alpha=0.8)
    plt.xlabel("Training Steps")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Tiny GPT Training Dynamics")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(config.plot_path)
    print(f"Saved loss curve plot to {config.plot_path}")

if __name__ == "__main__":
    main()
