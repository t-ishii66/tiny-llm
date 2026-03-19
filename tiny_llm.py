"""tiny-LLM: A minimal Transformer implementation for learning.

This single file implements a complete Transformer language model from scratch:
  - Word-level tokenizer (whitespace split, no subword)
  - Token & positional embeddings
  - Multi-head self-attention with causal mask
  - Position-wise feed-forward network
  - Layer normalization and residual connections
  - Training loop with cross-entropy loss (backprop via torch.autograd)
  - Greedy text generation

Forward pass is hand-written; backward pass uses PyTorch autograd.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==============================================================================
# Parameter helpers
# ==============================================================================
# nn.Parameter tells PyTorch "this tensor is learnable — track its gradient."
# We use small random values (× 0.02) so initial outputs are near zero,
# which keeps early training stable.

def param(*shape):
    """Create a learnable parameter initialized with small random values."""
    return nn.Parameter(torch.randn(*shape) * 0.02)

def param_ones(*shape):
    """Create a learnable parameter initialized to ones (for LayerNorm scale)."""
    return nn.Parameter(torch.ones(*shape))

def param_zeros(*shape):
    """Create a learnable parameter initialized to zeros (for biases)."""
    return nn.Parameter(torch.zeros(*shape))


# ==============================================================================
# Hyperparameters
# ==============================================================================

D_MODEL = 64       # Embedding dimension — size of each word vector
N_HEADS = 4        # Number of attention heads (D_MODEL must be divisible by this)
D_FF = 128         # Hidden dimension of the feed-forward network
N_LAYERS = 2       # Number of stacked transformer blocks
SEQ_LEN = 12       # Maximum sequence length (context window size)
EPOCHS = 200       # Number of training iterations over the full dataset
LR = 0.001         # Learning rate for Adam optimizer


# ==============================================================================
# Tokenizer — word-level, whitespace split
# ==============================================================================
# Real LLMs use subword tokenization (BPE, etc.).
# We simplify: one word = one token.

def build_vocab(text):
    """Build word-to-id and id-to-word mappings from a text corpus.

    Args:
        text: Raw text string (e.g. "the cat sat on the mat")

    Returns:
        vocab:   dict mapping word -> integer id  (e.g. {"the": 1, "cat": 2, ...})
        id2word: dict mapping integer id -> word   (e.g. {1: "the", 2: "cat", ...})

    The special token <pad> is always assigned id 0.
    Words are assigned ids in order of first appearance.
    """
    words = text.split()
    vocab = {"<pad>": 0}  # Reserve id 0 for padding
    for w in words:
        if w not in vocab:
            vocab[w] = len(vocab)  # Assign next available id
    return vocab, {i: w for w, i in vocab.items()}


def tokenize(text, vocab):
    """Convert a text string into a list of token ids.

    Args:
        text:  Raw text string (e.g. "the cat sat on")
        vocab: Word-to-id mapping from build_vocab()

    Returns:
        List of integers (e.g. [1, 2, 3, 4])
    """
    return [vocab[w] for w in text.split()]


# ==============================================================================
# Model components — hand-written forward pass
# ==============================================================================

class TinyTransformer:
    """A minimal GPT-style Transformer language model.

    Architecture:
        token_ids → tok_emb + pos_emb → N × TransformerBlock → LayerNorm → logits

    All parameters are raw nn.Parameter tensors (no nn.Module).
    The forward pass is written explicitly to show every computation step.
    """

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

        # --- Embeddings ---
        # tok_emb: lookup table, each row is a learnable vector for one word
        #   shape: (vocab_size, D_MODEL) = (10, 64)
        self.tok_emb = param(vocab_size, D_MODEL)

        # pos_emb: one learnable vector per position in the sequence
        #   shape: (SEQ_LEN, D_MODEL) = (12, 64)
        self.pos_emb = param(SEQ_LEN, D_MODEL)

        # --- Transformer layers ---
        # Each layer contains: multi-head attention + FFN, with LayerNorm for each
        self.layers = []
        for _ in range(N_LAYERS):
            layer = {
                # --- Multi-head self-attention weights ---
                # Wq: projects input to Query  ("what am I looking for?")
                # Wk: projects input to Key    ("what information do I have?")
                # Wv: projects input to Value  ("what content do I provide?")
                # Wo: projects concatenated heads back to D_MODEL
                "Wq": param(D_MODEL, D_MODEL),  # (64, 64)
                "Wk": param(D_MODEL, D_MODEL),
                "Wv": param(D_MODEL, D_MODEL),
                "Wo": param(D_MODEL, D_MODEL),

                # --- Layer norm before attention ---
                # g = scale (gamma), b = shift (beta)
                "ln1_g": param_ones(D_MODEL),   # (64,) initialized to 1
                "ln1_b": param_zeros(D_MODEL),  # (64,) initialized to 0

                # --- Feed-forward network weights ---
                # Two linear layers: D_MODEL → D_FF → D_MODEL
                # W1, b1: first layer  (64 → 128, expand)
                # W2, b2: second layer (128 → 64, compress)
                "W1": param(D_MODEL, D_FF),     # (64, 128)
                "b1": param_zeros(D_FF),         # (128,)
                "W2": param(D_FF, D_MODEL),     # (128, 64)
                "b2": param_zeros(D_MODEL),      # (64,)

                # --- Layer norm before FFN ---
                "ln2_g": param_ones(D_MODEL),
                "ln2_b": param_zeros(D_MODEL),
            }
            self.layers.append(layer)

        # --- Final layer norm (applied before output projection) ---
        self.ln_f_g = param_ones(D_MODEL)
        self.ln_f_b = param_zeros(D_MODEL)

    def parameters(self):
        """Return a flat list of all learnable parameters for the optimizer."""
        params = [self.tok_emb, self.pos_emb, self.ln_f_g, self.ln_f_b]
        for layer in self.layers:
            params.extend(layer.values())
        return params

    def forward(self, token_ids):
        """Run the full forward pass: token ids → logits.

        Args:
            token_ids: Integer tensor of shape (batch, seq_len)
                       e.g. tensor([[1, 2, 3, 4, ...]])

        Returns:
            logits: Float tensor of shape (batch, seq_len, vocab_size)
                    Each position contains scores for every word in the vocabulary.
                    Higher score = model thinks that word is more likely to come next.
        """
        B, T = token_ids.shape  # B = batch size, T = sequence length

        # ---- Step 1: Embedding ----
        # Look up token vectors and add positional vectors.
        # tok_emb[token_ids]: (B, T) → (B, T, 64)  — each id becomes a 64-dim vector
        # pos_emb[:T]:        (T, 64)                — broadcast-added to every sample
        x = self.tok_emb[token_ids] + self.pos_emb[:T]  # (B, T, D_MODEL)

        # ---- Step 2: Transformer Blocks ----
        # Pass through N_LAYERS transformer blocks sequentially.
        # Each block refines the representation using attention and FFN.
        for layer in self.layers:
            x = transformer_block(x, layer)

        # ---- Step 3: Output projection ----
        # Apply final layer norm, then project back to vocabulary size.
        # Weight tying: reuse tok_emb as the output projection matrix.
        # Intuition: "if the output vector is close to the embedding of word X,
        #             then word X is likely the next word."
        x = layer_norm(x, self.ln_f_g, self.ln_f_b)
        logits = x @ self.tok_emb.T  # (B, T, 64) @ (64, 10) → (B, T, 10)
        return logits


def layer_norm(x, g, b, eps=1e-5):
    """Layer normalization: normalize each vector to mean=0, var=1, then scale+shift.

    Formula: g * (x - mean) / sqrt(var + eps) + b

    This stabilizes training by preventing vectors from growing too large or small
    as they pass through many layers.

    Args:
        x: Input tensor of shape (..., D_MODEL)
        g: Scale parameter (gamma), shape (D_MODEL,)
        b: Shift parameter (beta),  shape (D_MODEL,)
        eps: Small constant to avoid division by zero

    Returns:
        Normalized tensor, same shape as x
    """
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return g * (x - mean) / torch.sqrt(var + eps) + b


def self_attention(x, Wq, Wk, Wv, Wo):
    """Multi-head self-attention with causal mask.

    This is the core mechanism of the Transformer. It allows each token to
    "look at" other tokens in the sequence and gather relevant information.

    The process:
        1. Project input to Query, Key, Value
        2. Split into multiple heads (each head learns different attention patterns)
        3. Compute attention scores: how much should token i attend to token j?
        4. Apply causal mask: prevent attending to future tokens
        5. Softmax: convert scores to probabilities
        6. Weighted sum of Values: gather information from attended tokens
        7. Concatenate heads and project output

    Args:
        x:  Input tensor, shape (B, T, D_MODEL) — B=batch, T=seq_len
        Wq: Query projection weight,  shape (D_MODEL, D_MODEL)
        Wk: Key projection weight,    shape (D_MODEL, D_MODEL)
        Wv: Value projection weight,  shape (D_MODEL, D_MODEL)
        Wo: Output projection weight, shape (D_MODEL, D_MODEL)

    Returns:
        Output tensor, shape (B, T, D_MODEL)
    """
    B, T, D = x.shape
    head_dim = D // N_HEADS  # 64 // 4 = 16 dimensions per head

    # --- Step 1: Compute Query, Key, Value ---
    # Each token gets three different projections:
    #   Query = "what am I looking for?"
    #   Key   = "what do I contain?"
    #   Value = "what information do I pass along?"
    Q = x @ Wq  # (B, T, D) @ (D, D) → (B, T, D)
    K = x @ Wk
    V = x @ Wv

    # --- Step 2: Split into multiple heads ---
    # Reshape D_MODEL into (N_HEADS, head_dim), then move head dim to axis 1.
    # (B, T, D) → (B, T, 4, 16) → (B, 4, T, 16)
    # Each head independently computes attention with 16-dim vectors.
    Q = Q.view(B, T, N_HEADS, head_dim).transpose(1, 2)
    K = K.view(B, T, N_HEADS, head_dim).transpose(1, 2)
    V = V.view(B, T, N_HEADS, head_dim).transpose(1, 2)

    # --- Step 3: Compute attention scores ---
    # score(i,j) = dot product of Q_i and K_j, measuring similarity.
    # Divide by sqrt(head_dim) to keep values in a reasonable range
    # (large dot products would push softmax to extreme values).
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(head_dim)  # (B, H, T, T)

    # --- Step 4: Apply causal mask ---
    # In language modeling, token at position i must NOT see future tokens (i+1, i+2, ...).
    # We set future positions to -inf so softmax gives them probability 0.
    # torch.triu with diagonal=1 creates an upper triangular matrix of ones:
    #   [[0, 1, 1],
    #    [0, 0, 1],   ← 1 = "this position is in the future, mask it out"
    #    [0, 0, 0]]
    mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
    scores = scores.masked_fill(mask, float("-inf"))

    # --- Step 5: Softmax → attention weights ---
    # Convert scores to probabilities (each row sums to 1.0).
    # -inf values become 0 after softmax, effectively hiding future tokens.
    attn_weights = F.softmax(scores, dim=-1)  # (B, H, T, T)

    # --- Step 6: Weighted sum of Values ---
    # Each token's output is a weighted combination of all (visible) Value vectors.
    # If token i strongly attends to token j, it receives more of V_j.
    out = attn_weights @ V  # (B, H, T, head_dim)

    # --- Step 7: Concatenate heads and apply output projection ---
    # Merge all heads back: (B, 4, T, 16) → (B, T, 4, 16) → (B, T, 64)
    # Then project with Wo to mix information across heads.
    out = out.transpose(1, 2).contiguous().view(B, T, D)  # (B, T, D)
    out = out @ Wo
    return out


def feed_forward(x, W1, b1, W2, b2):
    """Position-wise feed-forward network with ReLU activation.

    Each token is independently transformed through two linear layers:
        x → expand to D_FF → ReLU → compress back to D_MODEL

    While self-attention mixes information BETWEEN tokens,
    the FFN transforms each token's representation INDIVIDUALLY.

    Formula: ReLU(x @ W1 + b1) @ W2 + b2

    Args:
        x:  Input tensor, shape (B, T, D_MODEL)
        W1: First layer weight,  shape (D_MODEL, D_FF) — expand: 64 → 128
        b1: First layer bias,    shape (D_FF,)
        W2: Second layer weight, shape (D_FF, D_MODEL) — compress: 128 → 64
        b2: Second layer bias,   shape (D_MODEL,)

    Returns:
        Output tensor, shape (B, T, D_MODEL)
    """
    # ReLU(x) = max(0, x) — introduces non-linearity so the network can
    # learn patterns more complex than simple linear transformations.
    return F.relu(x @ W1 + b1) @ W2 + b2


def transformer_block(x, layer):
    """One transformer block: self-attention + FFN, each with residual + layer norm.

    Structure (Pre-LN, GPT-2+ style):
        x → layer norm → self_attention → add residual → layer norm → FFN → add residual

    Residual connections (x + sublayer_output) let gradients flow directly through
    the network, making deep models trainable. Intuitively: "keep the original
    information and add new information on top."

    Args:
        x:     Input tensor, shape (B, T, D_MODEL)
        layer: Dict of all parameters for this block

    Returns:
        Output tensor, shape (B, T, D_MODEL)
    """
    # Sub-block 1: Layer norm → multi-head self-attention → residual
    normed = layer_norm(x, layer["ln1_g"], layer["ln1_b"])
    attn_out = self_attention(normed, layer["Wq"], layer["Wk"], layer["Wv"], layer["Wo"])
    x = x + attn_out

    # Sub-block 2: Layer norm → feed-forward → residual
    normed = layer_norm(x, layer["ln2_g"], layer["ln2_b"])
    ff_out = feed_forward(normed, layer["W1"], layer["b1"], layer["W2"], layer["b2"])
    x = x + ff_out
    return x


# ==============================================================================
# Training
# ==============================================================================

def make_training_data(text, vocab):
    """Create input/target pairs for next-word prediction.

    Slides a window of size SEQ_LEN across the token sequence.
    For each window, the target is the input shifted by one position:
        input:  [the, cat, sat, on,  the, mat, ...]
        target: [cat, sat, on,  the, mat, .,   ...]
    So at every position, the model must predict the NEXT word.

    Args:
        text:  Raw text corpus string
        vocab: Word-to-id mapping

    Returns:
        inputs:  Integer tensor, shape (num_samples, SEQ_LEN) e.g. (28, 12)
        targets: Integer tensor, shape (num_samples, SEQ_LEN) e.g. (28, 12)
    """
    tokens = tokenize(text, vocab)  # Convert entire corpus to list of ids

    # Slide a window across the token sequence
    inputs, targets = [], []
    for i in range(len(tokens) - SEQ_LEN):
        inputs.append(tokens[i : i + SEQ_LEN])       # 12 tokens as input
        targets.append(tokens[i + 1 : i + SEQ_LEN + 1])  # shifted by 1 as target
    return torch.tensor(inputs), torch.tensor(targets)


def train(model, inputs, targets):
    """Train the model to predict the next word at every position.

    Training loop:
        1. Forward:  inputs → transformer → logits (predicted scores)
        2. Loss:     compare logits with targets using cross-entropy
        3. Backward: compute gradients of loss w.r.t. all parameters (autograd)
        4. Update:   adjust parameters in the direction that reduces loss (Adam)

    Args:
        model:   TinyTransformer instance
        inputs:  Integer tensor, shape (num_samples, SEQ_LEN)
        targets: Integer tensor, shape (num_samples, SEQ_LEN)
    """
    # Adam optimizer: an improved gradient descent that adapts the learning rate
    # per-parameter based on the history of gradients.
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        # --- 1. Forward pass: compute predictions ---
        logits = model.forward(inputs)  # (B, T, vocab_size) e.g. (28, 12, 10)

        # --- 2. Compute loss: how wrong are the predictions? ---
        # Cross-entropy loss = -log(probability assigned to the correct word).
        # Low loss = model is confident about the right answer.
        # We flatten (B, T) into a single dimension for the loss function.
        loss = F.cross_entropy(
            logits.view(-1, model.vocab_size),  # (B*T, vocab_size) = (336, 10)
            targets.view(-1),                    # (B*T,) = (336,)
        )

        # --- 3. Backward pass: compute gradients ---
        optimizer.zero_grad()  # Reset gradients from previous iteration
        loss.backward()        # Backpropagate: compute d(loss)/d(param) for all params

        # --- 4. Update parameters ---
        # Each parameter moves a small step in the direction that reduces loss:
        #   param = param - learning_rate * gradient
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"epoch {epoch+1:4d}  loss={loss.item():.4f}")


# ==============================================================================
# Text Generation
# ==============================================================================

def generate(model, prompt, vocab, id2word, max_tokens=20):
    """Generate text by repeatedly predicting the next word.

    This is the core idea behind all LLMs: given a sequence of words,
    predict the most likely next word, append it, and repeat.

    Uses greedy decoding: always picks the highest-scoring word.
    Real LLMs use sampling with temperature, top-k, top-p, etc.
    for more diverse and natural outputs.

    Args:
        model:      Trained TinyTransformer instance
        prompt:     Starting text string (e.g. "the cat sat on")
        vocab:      Word-to-id mapping
        id2word:    Id-to-word mapping
        max_tokens: Number of new tokens to generate

    Returns:
        Generated text string (prompt + generated tokens)
    """
    tokens = tokenize(prompt, vocab)  # Convert prompt to token ids

    with torch.no_grad():  # No gradient tracking needed during generation
        for _ in range(max_tokens):
            # Take the last SEQ_LEN tokens as context (model's window size)
            context = tokens[-SEQ_LEN:]
            x = torch.tensor([context])           # (1, T) — batch of 1

            # Run forward pass to get scores for all words at each position
            logits = model.forward(x)             # (1, T, vocab_size)

            # We only care about the LAST position's prediction:
            # it has seen all previous tokens and predicts what comes next
            next_logit = logits[0, -1, :]         # (vocab_size,) — scores for next word

            # Pick the word with the highest score (greedy decoding)
            next_id = torch.argmax(next_logit).item()
            tokens.append(next_id)

    # Convert all token ids back to words
    return " ".join(id2word[t] for t in tokens)


# ==============================================================================
# Main — put it all together
# ==============================================================================

if __name__ == "__main__":
    # A tiny corpus — just enough to demonstrate the Transformer
    corpus = (
        "the cat sat on the mat . the dog sat on the log . "
        "the cat saw the dog . the dog saw the cat . "
        "the cat sat on the log . the dog sat on the mat ."
    )

    # Step 1: Build vocabulary (word <-> id mappings)
    vocab, id2word = build_vocab(corpus)
    print(f"vocab size: {len(vocab)}")
    print(f"vocab: {vocab}\n")

    # Step 2: Create training data (input/target pairs)
    inputs, targets = make_training_data(corpus, vocab)
    print(f"training samples: {inputs.shape[0]}, seq_len: {inputs.shape[1]}\n")

    # Step 3: Create and train the model
    model = TinyTransformer(len(vocab))
    train(model, inputs, targets)

    # Step 4: Generate text from prompts
    print("\n--- Generation ---")
    prompts = ["the cat sat on", "the dog saw"]
    for prompt in prompts:
        print(f'prompt: "{prompt}"')
        print(f"output: {generate(model, prompt, vocab, id2word)}")
        print()
