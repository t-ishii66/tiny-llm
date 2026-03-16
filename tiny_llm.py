"""tiny-LLM: A minimal Transformer implementation for learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def param(*shape):
    """Create a learnable parameter with small random init."""
    return nn.Parameter(torch.randn(*shape) * 0.02)

def param_ones(*shape):
    return nn.Parameter(torch.ones(*shape))

def param_zeros(*shape):
    return nn.Parameter(torch.zeros(*shape))

# ---- Hyperparameters ----
D_MODEL = 64       # embedding dimension
N_HEADS = 4        # number of attention heads
D_FF = 128         # feed-forward hidden dimension
N_LAYERS = 2       # number of transformer blocks
SEQ_LEN = 16       # max sequence length
EPOCHS = 200
LR = 0.001

# ---- Tokenizer (word-level, whitespace split) ----

def build_vocab(text):
    words = text.split()
    vocab = {"<pad>": 0}
    for w in words:
        if w not in vocab:
            vocab[w] = len(vocab)
    return vocab, {i: w for w, i in vocab.items()}

def tokenize(text, vocab):
    return [vocab[w] for w in text.split()]

# ---- Model components (hand-written forward pass) ----

class TinyTransformer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        # Embedding + output projection
        self.tok_emb = param(vocab_size, D_MODEL)
        self.pos_emb = param(SEQ_LEN, D_MODEL)

        # Transformer layers
        self.layers = []
        for _ in range(N_LAYERS):
            layer = {
                # Multi-head attention Q, K, V projections
                "Wq": param(D_MODEL, D_MODEL),
                "Wk": param(D_MODEL, D_MODEL),
                "Wv": param(D_MODEL, D_MODEL),
                "Wo": param(D_MODEL, D_MODEL),
                # Layer norm 1
                "ln1_g": param_ones(D_MODEL),
                "ln1_b": param_zeros(D_MODEL),
                # Feed-forward network
                "W1": param(D_MODEL, D_FF),
                "b1": param_zeros(D_FF),
                "W2": param(D_FF, D_MODEL),
                "b2": param_zeros(D_MODEL),
                # Layer norm 2
                "ln2_g": param_ones(D_MODEL),
                "ln2_b": param_zeros(D_MODEL),
            }
            self.layers.append(layer)

        # Final layer norm
        self.ln_f_g = param_ones(D_MODEL)
        self.ln_f_b = param_zeros(D_MODEL)

    def parameters(self):
        params = [self.tok_emb, self.pos_emb, self.ln_f_g, self.ln_f_b]
        for layer in self.layers:
            params.extend(layer.values())
        return params

    def forward(self, token_ids):
        """
        token_ids: (batch, seq_len) integer tensor
        returns:   (batch, seq_len, vocab_size) logits
        """
        B, T = token_ids.shape

        # ---- Step 1: Token Embedding + Positional Embedding ----
        x = self.tok_emb[token_ids] + self.pos_emb[:T]  # (B, T, D_MODEL)

        # ---- Step 2: Transformer Blocks ----
        for layer in self.layers:
            x = transformer_block(x, layer)

        # ---- Step 3: Final Layer Norm + Project to Vocab ----
        x = layer_norm(x, self.ln_f_g, self.ln_f_b)
        logits = x @ self.tok_emb.T  # weight tying: reuse tok_emb
        return logits


def layer_norm(x, g, b, eps=1e-5):
    """Layer normalization."""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return g * (x - mean) / torch.sqrt(var + eps) + b


def self_attention(x, Wq, Wk, Wv, Wo):
    """
    Multi-head self-attention with causal mask.

    x:  (B, T, D_MODEL)
    Each of Wq, Wk, Wv: (D_MODEL, D_MODEL)
    Wo: (D_MODEL, D_MODEL) output projection
    """
    B, T, D = x.shape
    head_dim = D // N_HEADS

    # Project to Q, K, V
    Q = x @ Wq  # (B, T, D)
    K = x @ Wk
    V = x @ Wv

    # Reshape into multiple heads: (B, N_HEADS, T, head_dim)
    Q = Q.view(B, T, N_HEADS, head_dim).transpose(1, 2)
    K = K.view(B, T, N_HEADS, head_dim).transpose(1, 2)
    V = V.view(B, T, N_HEADS, head_dim).transpose(1, 2)

    # Scaled dot-product attention
    #   scores(i,j) = how much token i attends to token j
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(head_dim)  # (B, H, T, T)

    # Causal mask: token i can only see tokens 0..i (not future tokens)
    mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
    scores = scores.masked_fill(mask, float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)  # (B, H, T, T)

    # Weighted sum of values
    out = attn_weights @ V  # (B, H, T, head_dim)

    # Concatenate heads and project
    out = out.transpose(1, 2).contiguous().view(B, T, D)  # (B, T, D)
    out = out @ Wo
    return out


def feed_forward(x, W1, b1, W2, b2):
    """Position-wise feed-forward network with ReLU."""
    return F.relu(x @ W1 + b1) @ W2 + b2


def transformer_block(x, layer):
    """One transformer block: attention + FFN, both with residual + layer norm."""
    # Multi-head self-attention with residual connection
    attn_out = self_attention(x, layer["Wq"], layer["Wk"], layer["Wv"], layer["Wo"])
    x = layer_norm(x + attn_out, layer["ln1_g"], layer["ln1_b"])

    # Feed-forward with residual connection
    ff_out = feed_forward(x, layer["W1"], layer["b1"], layer["W2"], layer["b2"])
    x = layer_norm(x + ff_out, layer["ln2_g"], layer["ln2_b"])
    return x


# ---- Training ----

def make_training_data(text, vocab):
    """Create input/target pairs for next-word prediction."""
    tokens = tokenize(text, vocab)
    # Slide a window of SEQ_LEN+1 over the text
    inputs, targets = [], []
    for i in range(len(tokens) - SEQ_LEN):
        inputs.append(tokens[i : i + SEQ_LEN])
        targets.append(tokens[i + 1 : i + SEQ_LEN + 1])
    return torch.tensor(inputs), torch.tensor(targets)


def train(model, inputs, targets):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        logits = model.forward(inputs)               # (B, T, vocab_size)
        loss = F.cross_entropy(
            logits.view(-1, model.vocab_size),        # flatten to (B*T, vocab_size)
            targets.view(-1),                         # flatten to (B*T,)
        )

        optimizer.zero_grad()
        loss.backward()                               # autograd handles backprop
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"epoch {epoch+1:4d}  loss={loss.item():.4f}")


# ---- Text Generation ----

def generate(model, prompt, vocab, id2word, max_tokens=20):
    tokens = tokenize(prompt, vocab)

    for _ in range(max_tokens):
        # Use last SEQ_LEN tokens as context
        context = tokens[-SEQ_LEN:]
        x = torch.tensor([context])
        logits = model.forward(x)                     # (1, T, vocab_size)
        next_logit = logits[0, -1, :]                 # last position
        next_id = torch.argmax(next_logit).item()     # greedy decode
        tokens.append(next_id)

    return " ".join(id2word[t] for t in tokens)


# ---- Main ----

if __name__ == "__main__":
    corpus = (
        "the cat sat on the mat . the dog sat on the log . "
        "the cat saw the dog . the dog saw the cat . "
        "the cat sat on the log . the dog sat on the mat ."
    )

    vocab, id2word = build_vocab(corpus)
    print(f"vocab size: {len(vocab)}")
    print(f"vocab: {vocab}\n")

    inputs, targets = make_training_data(corpus, vocab)
    print(f"training samples: {inputs.shape[0]}, seq_len: {inputs.shape[1]}\n")

    model = TinyTransformer(len(vocab))
    train(model, inputs, targets)

    print("\n--- Generation ---")
    print(generate(model, "the cat sat on", vocab, id2word))
    print(generate(model, "the dog saw", vocab, id2word))
