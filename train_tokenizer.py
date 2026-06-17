import os
from tokenizer.bpe import BPETokenizer

def main():
    corpus_path = "data/corpus.txt"
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()

    print("Training BPE Tokenizer...")
    tokenizer = BPETokenizer()
    num_merges = 1000
    merges = tokenizer.train_bpe(text, num_merges=num_merges)
    tokenizer.merges = merges
    
    import re
    words = re.findall(r"\w+|[^\w\s]", text)
    corpus_tokens = []
    for word in words:
        corpus_tokens.extend(tokenizer.encode_word(word))
    
    tokenizer.build_vocab(corpus_tokens)
    
    os.makedirs("tokenizer", exist_ok=True)
    tokenizer.save_vocab("tokenizer/vocab.json")
    tokenizer.save_merges("tokenizer/merges.json")
    
    print("\n--- Tokenizer Statistics ---")
    print(f"Vocabulary Size: {tokenizer.vocab_size}")
    print(f"Number of Merges Learned: {len(merges)}")
    print("Top 10 Most Common Learned Merges:")
    for i, merge in enumerate(merges[:10]):
        print(f"{i+1}. {merge}")
    print("----------------------------\n")

    # Update config.py dynamically with new vocab size
    config_path = "config.py"
    with open(config_path, "r", encoding="utf-8") as f:
        config_content = f.read()
    
    import re as regex
    new_content = regex.sub(r'"vocab_size": \d+,', f'"vocab_size": {tokenizer.vocab_size},', config_content)
    
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("Updated config.py with the correct vocab_size.")

if __name__ == "__main__":
    main()
