import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer.bpe import BPETokenizer

class NextTokenDataset(Dataset):
    def __init__(self, data_path, vocab_path, merges_path, context_length=64):
        self.context_length = context_length
        
        # Initialize and load tokenizer
        self.tokenizer = BPETokenizer()
        self.tokenizer.load_vocab(vocab_path)
        self.tokenizer.load_merges(merges_path)
        
        # Load and encode data
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        self.token_ids = self.tokenizer.encode_to_ids(text, add_special_tokens=False)
        
    def __len__(self):
        # We need context_length + 1 tokens for each sample (x and y)
        return len(self.token_ids) - self.context_length
        
    def __getitem__(self, idx):
        # Input sequence
        x = torch.tensor(self.token_ids[idx : idx + self.context_length], dtype=torch.long)
        # Target sequence (shifted by 1)
        y = torch.tensor(self.token_ids[idx + 1 : idx + self.context_length + 1], dtype=torch.long)
        
        return x, y

def get_dataloader(config):
    dataset = NextTokenDataset(
        data_path=config.corpus_path,
        vocab_path=config.vocab_path,
        merges_path=config.merges_path,
        context_length=config.context_length
    )
    
    return DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        drop_last=True
    )
