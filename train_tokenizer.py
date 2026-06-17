import os
import json
from tokenizer.bpe import BPETokenizer

def main():
    corpus_path = "data/corpus.txt"
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()

    print("Training BPE Tokenizer...")
    tokenizer = BPETokenizer()
    merges = tokenizer.train_bpe(text, num_merges=300)
    tokenizer.merges = merges
    
    # Encode text to build unique tokens for vocabulary
    # We will use words to build corpus tokens
    import re
    words = re.findall(r"\w+|[^\w\s]", text)
    corpus_tokens = []
    for word in words:
        corpus_tokens.extend(tokenizer.encode_word(word))
    
    tokenizer.build_vocab(corpus_tokens)
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    os.makedirs("tokenizer", exist_ok=True)
    
    # Save the vocabulary and merges
    tokenizer.save_vocab("tokenizer/vocab.json")
    tokenizer.save_merges("tokenizer/merges.json")
    print("Saved vocab.json and merges.json")

    # Update config.py with the new vocab size
    config_path = "config.py"
    with open(config_path, "r", encoding="utf-8") as f:
        config_content = f.read()
    
    import re as regex
    new_content = regex.sub(r"vocab_size: int = \d+", f"vocab_size: int = {tokenizer.vocab_size}", config_content)
    
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("Updated config.py with the correct vocab_size.")

if __name__ == "__main__":
    main()
