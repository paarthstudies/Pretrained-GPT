import torch
import torch.nn.functional as F

from config import config
from model.tiny_gpt import TinyGPT
from tokenizer.bpe import BPETokenizer

def generate(model, tokenizer, prompt, max_new_tokens=50, temperature=1.0, top_k=None):
    """
    Autoregressive text generation.
    """
    model.eval()
    
    # Encode prompt
    tokens = tokenizer.encode_to_ids(prompt, add_special_tokens=False)
    
    # We maintain the sequence as a tensor
    idx = torch.tensor([tokens], dtype=torch.long, device=config.device)
    
    for _ in range(max_new_tokens):
        # 1. Truncate context if it exceeds model's context length
        idx_cond = idx if idx.size(1) <= config.context_length else idx[:, -config.context_length:]
        
        # 2. Forward pass to get logits
        with torch.no_grad():
            logits, _ = model(idx_cond)
            
        # 3. Focus on the very last step in the sequence
        # logits shape is (B, T, vocab_size). We want (B, 1, vocab_size)
        logits = logits[:, -1, :] # (B, vocab_size)
        
        # 4. Temperature Scaling
        # Dividing logits by temperature. 
        # t < 1.0 makes distribution sharper (more greedy/deterministic)
        # t > 1.0 makes distribution flatter (more random/diverse)
        logits = logits / temperature
        
        # 5. Top-k Sampling
        if top_k is not None:
            # Find the top-k values and their indices
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            # The lowest value among the top-k becomes our threshold
            # Anything below this threshold is set to -infinity
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        # 6. Convert to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # 7. Sample the next token
        next_token = torch.multinomial(probs, num_samples=1) # (B, 1)
        
        # 8. Append to the sequence
        idx = torch.cat((idx, next_token), dim=1) # (B, T+1)
        
    # Decode back to text
    out_tokens = idx[0].tolist()
    return tokenizer.decode_from_ids(out_tokens)

def run_comparative_analysis():
    print(f"Loading model from {config.checkpoint_path}")
    
    # Initialize tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load_vocab(config.vocab_path)
    tokenizer.load_merges(config.merges_path)
    
    # Load model
    model = TinyGPT(config)
    model.load_state_dict(torch.load(config.checkpoint_path, weights_only=True))
    model.to(config.device)
    
    prompt = "The transformer architecture"
    print(f"\n--- Comparative Analysis ---")
    print(f"Prompt: '{prompt}'\n")
    
    configs = [
        {"temp": 0.5, "top_k": 1, "desc": "Low Temp + Greedy (Deterministic, repetitive)"},
        {"temp": 0.8, "top_k": 5, "desc": "Med Temp + Tight Top-k (Balanced, focused)"},
        {"temp": 1.0, "top_k": 10, "desc": "Standard Temp + Med Top-k (Natural diversity)"},
        {"temp": 1.5, "top_k": 20, "desc": "High Temp + Loose Top-k (Creative, potentially incoherent)"},
        {"temp": 2.0, "top_k": 20, "desc": "Very High Temp (Random, likely nonsense)"}
    ]
    
    for c in configs:
        print(f"\nExperiment: Temperature={c['temp']}, Top-k={c['top_k']}")
        print(f"Expected behavior: {c['desc']}")
        generated_text = generate(
            model=model, 
            tokenizer=tokenizer, 
            prompt=prompt, 
            max_new_tokens=30, 
            temperature=c['temp'], 
            top_k=c['top_k']
        )
        print(f"Output:\n{generated_text}\n")
        print("-" * 50)

if __name__ == "__main__":
    run_comparative_analysis()
