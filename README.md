# Tiny GPT From Scratch 🚀

A comprehensive, from-scratch implementation of a GPT-style Decoder-Only Transformer. This project was built to demonstrate a deep, practical understanding of modern Large Language Model (LLM) pretraining concepts, architecture, and text generation mechanics.

Rather than relying on pre-built libraries like HuggingFace Transformers, this repository builds the core components from the ground up using raw PyTorch.

## Project Structure

```text
tiny-gpt-llm/
├── data/
│   └── corpus.txt              # AI/LLM domain-specific training corpus
├── tokenizer/
│   ├── bpe.py                  # Simplified BPE Tokenizer implementation
│   ├── vocab.json              # Trained vocabulary
│   └── merges.json             # BPE merge rules
├── model/
│   ├── attention.py            # Multi-Head Causal Self-Attention
│   ├── transformer_block.py    # Transformer Block (MLP + Attention + LayerNorm)
│   └── tiny_gpt.py             # Complete Decoder-Only Transformer module
├── train.py                    # Complete PyTorch training loop
├── generate.py                 # Autoregressive generation & sampling strategies
├── dataset.py                  # PyTorch Dataset for context-window generation
├── config.py                   # Centralized configuration hyperparameters
├── plots/
│   └── loss_curve.png          # Visualized training dynamics
└── checkpoints/
    └── model.pt                # Saved model weights
```

## Educational Deep Dive

This project breaks down the "magic" of LLMs into understandable, implementable components.

### 1. Byte Pair Encoding (BPE) Tokenization
- **What it does:** Converts raw text into a sequence of integer IDs by iteratively merging the most frequent character pairs into subwords.
- **Why it is needed:** Strikes a balance between character-level (too long, little semantic meaning) and word-level (infinite vocabulary, out-of-vocabulary issues) representations.
- **Implementation:** Found in `tokenizer/bpe.py`.

### 2. Token & Positional Embeddings
- **What they do:** `Token Embeddings` map discrete IDs to dense continuous vectors. `Positional Embeddings` add sequence order information to these vectors.
- **Why they are needed:** Neural networks cannot process raw text. They need dense vectors. Furthermore, self-attention has no inherent notion of sequence order (it treats input as a "bag of words"), making positional embeddings critical.
- **Shapes:** Input `(Batch, Sequence_Length)` → Output `(Batch, Sequence_Length, Embedding_Dim)`.

### 3. Multi-Head Self-Attention & Causal Masking
- **What it does:** Computes relationships between all tokens in a sequence using Queries, Keys, and Values. "Multi-head" means it does this in parallel across different learned linear projections.
- **Causal Masking:** A lower triangular matrix of ones (the rest negative infinity) applied before softmax.
- **Why it is needed:** Ensures that when predicting the token at position $t$, the model can only attend to tokens at positions $1$ through $t-1$. This prevents "looking into the future" during training.

### 4. Transformer Decoder Architecture
- **What it does:** A stack of Transformer Blocks, each containing Multi-Head Attention, a Feed Forward Network (MLP), Layer Normalization, and Residual Connections.
- **Why it is needed:** Stacking blocks allows the network to learn deep hierarchical features. Residual connections are crucial to allow gradients to flow through deep networks without vanishing.

### 5. Training Dynamics (Next Token Prediction)
- **What it does:** Uses `CrossEntropyLoss` to measure the difference between the predicted probability distribution of the next token and the actual next token.
- **Workflow:** The model processes a sequence $X = [x_1, x_2, ..., x_t]$ and attempts to predict $Y = [x_2, x_3, ..., x_{t+1}]$.
- **Optimization:** We use the `AdamW` optimizer, a standard choice for Transformers due to its decoupled weight decay.

### 6. Autoregressive Generation & Sampling
- **What it does:** Generates text one token at a time, appending the newly generated token back into the sequence and feeding it back into the model.
- **Temperature:** Scales the logits before softmax. `T < 1.0` makes the model more confident/greedy, while `T > 1.0` flattens the distribution, adding diversity.
- **Top-k Sampling:** Restricts the model to sample only from the top `k` most likely next tokens, preventing the generation of completely nonsensical words.

## Running the Project

1. **Install Requirements:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the BPE Tokenizer:**
   ```bash
   python train_tokenizer.py
   ```
   *(This trains the BPE tokenizer on `data/corpus.txt` and saves the vocabulary)*

3. **Train the Model:**
   ```bash
   python train.py
   ```
   *(This will train the Tiny GPT model and output a loss curve to `plots/loss_curve.png`)*

4. **Generate Text:**
   ```bash
   python generate.py
   ```
   *(Runs the comparative analysis across different Temperature and Top-k values)*

## Training Dynamics Visualization

During training, we track the Cross Entropy Loss. You can find the plot in `plots/loss_curve.png`. A healthy training run exhibits a steep initial drop followed by a gradual convergence.

*Project built to demonstrate engineering fundamentals in modern AI.*
