# Step 3: Peeking Inside the Transformer

Let's actually observe the internals of the trained model.
We'll check what values the embedding vectors and attention weights actually take.

We'll continue working in interactive mode with `uv run --with torch python -i tiny_llm.py`.

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
tensor([-0.05,  0.13,  0.27, ...], requires_grad=True)
```

The numbers are learned from random initialization with each training run, so they vary every time you run it.

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

<details>
<summary>Copy-paste version (no prompt markers)</summary>

Raw code with `>>>` / `...` stripped. **Paste directly** into interactive mode.

```python
import torch.nn.functional as F

def similarity(word1, word2):
    v1 = model.tok_emb[vocab[word1]]
    v2 = model.tok_emb[vocab[word2]]
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()

similarity("cat", "dog")    # Used in similar contexts
similarity("cat", ".")      # Completely different roles
similarity("mat", "log")    # Both come after "sat on the ___"
```

</details>

If the similarity between "cat" and "dog" is high and the similarity between "cat" and "." is low,
it shows that the model has learned (to a small extent) the semantic relationships between words.

---

## 3.3 Peeking at Attention Weights

The core of the Transformer is the **Attention weight matrix** that expresses "which tokens are attending to which positions." Let's pull out the attention weights of the first layer and take a look.

The details of the computation are covered thoroughly in [Chapter 2: Self-Attention](../02_transformer.md) of the main text, so here let's focus on **seeing the results**. Paste the following helper into interactive mode:

```python
>>> import math
>>> def attn_layer0(text):
...     x = torch.tensor([tokenize(text, vocab)])
...     T = x.shape[1]
...     emb = model.tok_emb[x] + model.pos_emb[:T]
...     L = model.layers[0]
...     n = layer_norm(emb, L["ln1_g"], L["ln1_b"])
...     Q, K = n @ L["Wq"], n @ L["Wk"]
...     scores = (Q @ K.transpose(-2, -1)) / math.sqrt(64)
...     mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
...     return torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)[0]
...
```

<details>
<summary>Copy-paste version (no prompt markers)</summary>

Raw code with `>>>` / `...` stripped. **Paste directly** into interactive mode.

```python
import math

def attn_layer0(text):
    x = torch.tensor([tokenize(text, vocab)])
    T = x.shape[1]
    emb = model.tok_emb[x] + model.pos_emb[:T]
    L = model.layers[0]
    n = layer_norm(emb, L["ln1_g"], L["ln1_b"])
    Q, K = n @ L["Wq"], n @ L["Wk"]
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(64)
    mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
    return torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)[0]
```

</details>

Let's try running it:

```python
>>> attn = attn_layer0("the cat sat on the mat")
>>> print(attn.detach().round(decimals=2))
```

A 6×6 attention weight matrix is displayed. Each row shows "which positions that position is attending to":

```
Row 0 (the): [1.00, 0.00, 0.00, 0.00, 0.00, 0.00]   ← Can only see itself
Row 1 (cat): [0.??, 0.??, 0.00, 0.00, 0.00, 0.00]   ← Can see "the" and "cat"
Row 2 (sat): [0.??, 0.??, 0.??, 0.00, 0.00, 0.00]
...
```

**The upper-right triangle being 0 is the effect of the causal mask** — the mechanism that "forbids looking at future tokens" can be confirmed here as concrete numbers.

> If you want to see the differences across multi-head (4 heads), split the helper's `Q`, `K` into 4 heads and run the same computation. The procedure is in the main text [§2 Multi-Head Attention](../02_transformer.md).

---

## 3.4 (Optional) Drawing Attention as a Heatmap

Visualizing is easier to grasp than a table of numbers, so if you have matplotlib, let's visualize it.

```bash
# First start interactive mode with matplotlib installed
uv run --with torch --with matplotlib python -i tiny_llm.py
```

After re-pasting the helper `attn_layer0` from 3.3:

```python
>>> import matplotlib.pyplot as plt
>>> tokens = "the cat sat on the mat".split()
>>> attn = attn_layer0("the cat sat on the mat")
>>>
>>> fig, ax = plt.subplots(figsize=(5, 4))
>>> im = ax.imshow(attn.detach().numpy(), cmap="Blues")
>>> ax.set_xticks(range(6)); ax.set_yticks(range(6))
>>> ax.set_xticklabels(tokens); ax.set_yticklabels(tokens)
>>> ax.set_xlabel("attended to"); ax.set_ylabel("from position")
>>> plt.colorbar(im)
>>> plt.tight_layout(); plt.show()
```

<details>
<summary>Copy-paste version (no prompt markers)</summary>

Raw code with `>>>` stripped. **Paste directly** into interactive mode.

```python
import matplotlib.pyplot as plt
tokens = "the cat sat on the mat".split()
attn = attn_layer0("the cat sat on the mat")

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(attn.detach().numpy(), cmap="Blues")
ax.set_xticks(range(6)); ax.set_yticks(range(6))
ax.set_xticklabels(tokens); ax.set_yticklabels(tokens)
ax.set_xlabel("attended to"); ax.set_ylabel("from position")
plt.colorbar(im)
plt.tight_layout(); plt.show()
```

</details>

You should see a pattern where only the lower triangle is colored (the upper-right causal mask region is white) and the area near the diagonal is dark.
Drawing and comparing the different attention patterns for each head gives an intuitive sense of why Multi-Head Attention matters.

---

## 3.5 Tracing the Generation Process Step by Step

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

<details>
<summary>Copy-paste version (no prompt markers)</summary>

Raw code with `>>>` / `...` stripped. **Paste directly** into interactive mode.

```python
# Predict the next word from "the cat sat on"
prompt = "the cat sat on"
tokens = tokenize(prompt, vocab)
print(tokens)

# Forward pass
x = torch.tensor([tokens])
logits = model.forward(x)          # (1, 4, 10)
next_logit = logits[0, -1, :]      # Scores at the last position

# Display scores for each word
for i, score in enumerate(next_logit.tolist()):
    print(f"  {id2word[i]:>5s}: {score:.3f}")
```

</details>

The word with the highest score is selected by `argmax`:

```python
>>> next_id = torch.argmax(next_logit).item()
>>> print(f"predicted: {id2word[next_id]}")
```

"the" should be predicted as the next word after "the cat sat on"
(because the corpus contains "the cat sat on the mat").

---

## 3.6 Key Takeaways So Far

- **Embedding**: Through training, words in similar contexts get similar vectors
- **Attention weight matrix**: Becomes triangular due to the causal mask. Each head shows different patterns
- **Generation**: Scores are produced for all words, and the word with the highest score becomes the next prediction
- Everything can be inspected as **numerical tensors** — it's not a black box

---

Next: [Step 4: Modifying the Code](04_experiments.md)
