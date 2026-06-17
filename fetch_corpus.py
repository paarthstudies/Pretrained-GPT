import json
import urllib.request
import urllib.parse
import re

def fetch_wikipedia_text(page_title):
    url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&titles={urllib.parse.quote(page_title)}&format=json"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
    pages = data['query']['pages']
    for page_id in pages:
        extract = pages[page_id].get('extract', '')
        return extract
    return ""

def main():
    topics = [
        "Large language model",
        "Transformer (machine learning model)",
        "Attention (machine learning)",
        "Deep learning",
        "Machine learning",
        "Artificial neural network",
        "Generative pre-trained transformer",
        "Natural language processing",
        "Word embedding"
    ]
    
    print("Fetching Wikipedia articles for corpus...")
    all_text = ""
    for topic in topics:
        print(f"Fetching: {topic}")
        text = fetch_wikipedia_text(topic)
        # Clean up some wiki formatting (like == Headings ==)
        text = re.sub(r'==+ .*? ==+', '', text)
        all_text += text + "\n\n"
        
    # Append the previous synthetic text as well
    synth_text = """
The transformer architecture utilizes self-attention mechanisms to capture long-range dependencies across tokens.
By leveraging self-attention, transformers can process sequences in parallel, unlike recurrent neural networks.
A key component of the transformer is the multi-head attention mechanism.
Multi-head attention allows the model to jointly attend to information from different representation subspaces.
In natural language processing, tokens are often generated using byte pair encoding or BPE.
Subword tokenization strikes a balance between character-level and word-level representations.
Language models are trained using cross entropy loss to predict the next token in a sequence.
Autoregressive text generation involves predicting one token at a time based on previous tokens.
To prevent the model from looking ahead during training, causal masking is applied.
Causal masking ensures that the prediction for a token only depends on previous tokens.
Token embeddings map discrete token IDs to continuous dense vectors.
Positional embeddings are added to token embeddings to provide information about the sequence order.
Without positional embeddings, the self-attention mechanism would treat the sequence as a bag of words.
The feed forward network in a transformer block consists of two linear transformations with a ReLU or GELU activation in between.
Layer normalization is used to stabilize the training of deep neural networks.
Residual connections help in mitigating the vanishing gradient problem by providing shortcuts around layers.
The language modeling head projects the final hidden states into a vocabulary sized distribution.
Sampling strategies like temperature scaling and top-k are used to control the randomness of generated text.
Lower temperature values make the model output more deterministic and focused.
Higher temperature values increase the diversity and creativity of the generated text.
Top-k sampling restricts the next token selection to the k most likely tokens.
Greedy decoding is a special case where we always select the most likely next token.
Large language models have shown remarkable capabilities in zero-shot and few-shot learning.
Pretraining involves training the model on a massive corpus of text data.
Fine-tuning adapts a pretrained model to specific downstream tasks.
During backpropagation, the gradients of the loss with respect to the model parameters are computed.
The AdamW optimizer is widely used for training transformer models due to its effectiveness.
Learning rate schedules, such as warmup and decay, are crucial for successful training.
Overfitting occurs when the model memorizes the training data but fails to generalize.
Tracking the training loss curve provides insights into the learning dynamics.
Next token prediction is the fundamental objective for training causal language models.
The context length determines the maximum sequence of tokens the model can process at once.
Byte pair encoding learns merge rules by iteratively combining frequent character pairs.
The vocabulary size is a hyperparameter that affects both model size and tokenization granularity.
Transformers rely entirely on attention mechanisms to draw global dependencies between input and output.
Self-attention computes a weighted sum of values, where the weights are derived from queries and keys.
The scaled dot-product attention scales the dot products by the inverse square root of the dimension.
This scaling prevents the gradients from becoming too small during training.
Model architecture decisions, such as the number of layers and heads, impact performance and capacity.
Deep learning frameworks like PyTorch provide efficient implementations of tensor operations.
A tiny GPT model serves as an excellent educational tool for understanding language modeling.
Text generation pipelines must handle prompt encoding, forward passes, and token decoding.
By implementing a transformer from scratch, one gains a deep understanding of its inner workings.
The matrix multiplication in the linear layers accounts for a significant portion of the computational cost.
Understanding the shape of tensors at each step is essential for debugging neural networks.
The causal mask is typically implemented as an upper triangular matrix of negative infinities.
When passed through the softmax function, these negative infinities become zero probabilities.
This ensures that the attention weights for future tokens are strictly zero.
The hidden state dimension is also known as the model dimension or d_model.
Multi-head attention concatenates the outputs of multiple attention heads and projects them.
Each attention head can potentially learn to focus on different linguistic features.
The loss function quantifies the difference between the predicted probabilities and the actual next tokens.
Minimizing the cross entropy loss maximizes the likelihood of the training data.
The context window limits the amount of historical information the model can use for prediction.
Byte pair encoding starts with a vocabulary of individual characters and merges them into subwords.
The decode function converts a sequence of token IDs back into readable text.
Special tokens like pad, unk, bos, and eos are often added to the vocabulary.
The pad token is used to make sequences of varying lengths uniform within a batch.
The eos token signifies the end of a generated sequence.
Training dynamics can be analyzed by observing how the loss decreases over epochs.
Spikes in the loss curve might indicate issues with the learning rate or data quality.
A smooth, decreasing loss curve generally suggests that the model is learning effectively.
Temperature scaling divides the logits by a temperature parameter before applying softmax.
Top-k sampling sorts the logits and keeps only the top k values, masking the rest.
This project demonstrates the core concepts of building and training a generative language model.
"""
    all_text += synth_text
    
    with open("data/corpus.txt", "w", encoding="utf-8") as f:
        f.write(all_text)
        
    words = all_text.split()
    print(f"Corpus generated successfully. Total words: {len(words)}")

if __name__ == "__main__":
    main()
