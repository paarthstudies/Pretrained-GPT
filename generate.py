import torch
import torch.nn.functional as F

from config import GPT_CONFIG
from model.tiny_gpt import TinyGPT
from tokenizer.bpe import BPETokenizer

def generate(model, tokenizer, prompt, max_new_tokens=50, temperature=1.0, top_k=None):
    model.eval()
    
    tokens = tokenizer.encode_to_ids(prompt, add_special_tokens=False)
    idx = torch.tensor([tokens], dtype=torch.long, device=GPT_CONFIG["device"])
    
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= GPT_CONFIG["context_length"] else idx[:, -GPT_CONFIG["context_length"]:]
        
        with torch.no_grad():
            logits, _ = model(idx_cond)
            
        logits = logits[:, -1, :]
        logits = logits / temperature
        
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)
        
    out_tokens = idx[0].tolist()
    return tokenizer.decode_from_ids(out_tokens)

def run_comparative_analysis():
    print(f"Loading model from {GPT_CONFIG['checkpoint_path']}")
    
    tokenizer = BPETokenizer()
    tokenizer.load_vocab(GPT_CONFIG["vocab_path"])
    tokenizer.load_merges(GPT_CONFIG["merges_path"])
    
    model = TinyGPT(GPT_CONFIG)
    model.load_state_dict(torch.load(GPT_CONFIG["checkpoint_path"], weights_only=True))
    model.to(GPT_CONFIG["device"])
    
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
