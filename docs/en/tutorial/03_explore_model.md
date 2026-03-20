# Step 3: Peeking Inside the Transformer

Let's observe the internals of the trained model.
We'll check what values the embedding vectors and attention weights actually take.

Continue working in interactive mode with `python -i tiny_llm.py`.

---

## 3.1 Checking the Number of Parameters

```python
>>> total = sum(p.numel() for p in model.parameters())
>>> print(f"Total parameters: {total}")
Total parameters: 67968
```

Approximately 68,000 parameters (mainly weight matrices) were tuned over 200 training iterations.

---

## 3.2 Observing Embedding Vectors

Each word is represented as a 64-dimensional vector:

```python
>>> model.tok_emb.shape
torch.Size([10, 64])
```

Let's look at the first 10 elements of the vector for "cat" (number 2):

```python
>>> model.tok_emb[2][:10]
tensor([...], requires_grad=True)
```

Through training, words with similar roles should have similar vectors.
Let's verify using cosine similarity:

```python
>>> import torch.nn.functional as F
>>>
>>> def similarity(word1, word2):
...     v1 = model.tok_emb[vocab[word1]]
...     v2 = model.tok_emb[vocab[word2]]
...     return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
...
>>> similarity("cat", "dog")    # Used in similar contexts
>>> similarity("cat", ".")      # Completely different roles
>>> similarity("mat", "log")    # Both come after "sat on the ___"
```

If the similarity between "cat" and "dog" is high, and the similarity between "cat" and "." is low,
it shows that the model has learned (to a small extent) the semantic relationships between words.

---

## 3.3 Visualizing Attention Weights

Let's see which words the Transformer is paying attention to.

```python
>>> # Tokenize a test sentence and run Forward pass
>>> test_tokens = tokenize("the cat sat on the mat", vocab)
>>> x = torch.tensor([test_tokens])   # (1, 6)
>>> x.shape
torch.Size([1, 6])
```

To extract attention scores, we manually compute up to the intermediate step:

```python
>>> import math
>>>
>>> # Embedding
>>> emb = model.tok_emb[x] + model.pos_emb[:6]   # (1, 6, 64)
>>>
>>> # Get first layer's parameters
>>> layer = model.layers[0]
>>>
>>> # Pre-LN: Apply Layer Norm before Attention
>>> normed = layer_norm(emb, layer["ln1_g"], layer["ln1_b"])  # (1, 6, 64)
>>>
>>> # Compute Q, K (using the Layer Norm output)
>>> Q = normed @ layer["Wq"]    # (1, 6, 64)
>>> K = normed @ layer["Wk"]    # (1, 6, 64)
>>>
>>> # Attention scores (before mask)
>>> # Note: This is a simplified single-head version (64-dim).
>>> #    The actual model splits into 4 heads and divides by sqrt(16) (see section 3.4).
>>> scores = Q @ K.transpose(-2, -1) / math.sqrt(64)   # (1, 6, 6)
>>>
>>> # Apply causal mask + Softmax
>>> mask = torch.triu(torch.ones(6, 6), diagonal=1).bool()
>>> scores = scores.masked_fill(mask, float("-inf"))
>>> attn = torch.softmax(scores, dim=-1)
>>>
>>> print(attn[0].detach())
```

A 6x6 Attention weight matrix (after softmax) is displayed. Each row shows "which positions that position is attending to":

```
Row 0 (the):  [1.00, 0.00, 0.00, 0.00, 0.00, 0.00]  ← Can only see itself
Row 1 (cat):  [0.??, 0.??, 0.00, 0.00, 0.00, 0.00]  ← Can see "the" and "cat"
Row 2 (sat):  [0.??, 0.??, 0.??, 0.00, 0.00, 0.00]
...
```

The upper right being 0.00 is the effect of the **causal mask**.
Future words cannot be seen.

---

## 3.4 Comparing All Heads' Attention

With multi-head attention (4 heads), each head attends with a different pattern:

```python
>>> # Split Q, K into 4 heads (head_dim = 64 / 4 = 16)
>>> B, T, D = 1, 6, 64
>>> head_dim = D // 4
>>>
>>> q = Q.view(B, T, 4, head_dim).transpose(1, 2)  # (1, 4, 6, 16)
>>> k = K.view(B, T, 4, head_dim).transpose(1, 2)  # (1, 4, 6, 16)
>>>
>>> scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
>>> mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
>>> scores = scores.masked_fill(mask, float("-inf"))
>>> attn_heads = torch.softmax(scores, dim=-1)
>>>
>>> # Display each head's attention
>>> for h in range(4):
...     print(f"\n--- Head {h} ---")
...     print(attn_heads[0, h].detach().round(decimals=2))
```

You can observe each head attending with different patterns.
One head might focus on the immediately preceding word, while another focuses on the word at the beginning of the sentence.

---

## 3.5 Tracing the Generation Process Step by Step

> **Note**: Use only words that appear in the training corpus for the prompt.
> The current `tokenize()` does not handle out-of-vocabulary words, so unknown words will cause a `KeyError`.

```python
>>> # Predict the next word from "the cat sat on"
>>> prompt = "the cat sat on"
>>> tokens = tokenize(prompt, vocab)
>>> print(tokens)
[1, 2, 3, 4]

>>> # Forward pass
>>> x = torch.tensor([tokens])
>>> logits = model.forward(x)          # (1, 4, 10)
>>> next_logit = logits[0, -1, :]      # Scores at the last position

>>> # Display scores for each word
>>> for i, score in enumerate(next_logit.tolist()):
...     print(f"  {id2word[i]:>5s}: {score:.3f}")
```

The word with the highest score is selected by `argmax`:

```python
>>> next_id = torch.argmax(next_logit).item()
>>> print(f"predicted: {id2word[next_id]}")
```

"the" should be predicted as the next word after "the cat sat on"
(because the corpus contains "the cat sat on the mat").

---

## 3.6 Key Takeaways So Far

- **Embedding**: Through training, words used in similar contexts get similar vectors
- **Attention weight matrix**: Becomes triangular due to the causal mask. Each head shows different patterns
- **Generation**: Scores are produced for all words, and the word with the highest score becomes the next prediction
- Everything can be inspected as **numerical tensors** — it's not a black box

---

Next: [Step 4: Experiments and Modifications](04_experiments.md)
