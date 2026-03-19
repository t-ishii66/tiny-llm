# Step 4: Experiments and Modifications

By now, you've gained insight into how the Transformer works.
Finally, let's modify the code and experiment.

---

## 4.1 Changing the Corpus

Try changing the corpus in the `if __name__ == "__main__":` block of `tiny_llm.py`:

```python
# Original corpus
corpus = (
    "the cat sat on the mat . the dog sat on the log . "
    "the cat saw the dog . the dog saw the cat . "
    "the cat sat on the log . the dog sat on the mat ."
)
```

For example, add a new pattern:

```python
corpus = (
    "the cat sat on the mat . the dog sat on the log . "
    "the cat saw the dog . the dog saw the cat . "
    "the cat sat on the log . the dog sat on the mat . "
    "the bird sat on the log . the bird saw the cat ."
)
```

Run it and see if "bird" is learned correctly:

```bash
python tiny_llm.py
```

> **Note**: Adding new words changes the vocabulary size (10 → 11).
> The code automatically detects the vocabulary size, so it will work as-is.

---

## 4.2 Changing Hyperparameters

Change the hyperparameters at the top of `tiny_llm.py` and observe how the training results change.

### Experiment 1: Change the Number of Attention Heads

```python
N_HEADS = 1    # Single head only (no multi-head)
```

It can still learn with just 1 head, but attention patterns are limited to a single type.

### Experiment 2: Change the Number of Layers

```python
N_LAYERS = 1   # Single layer only
```

Even 1 layer can learn this small corpus.
However, more complex patterns require deeper layers.

### Experiment 3: Change the Embedding Dimension

```python
D_MODEL = 16   # Reduced from 64 → 16
D_FF = 32      # Usually 2–4× D_MODEL
```

With smaller dimensions, the model may not be able to represent word meanings adequately.
Observe whether loss convergence slows down or the final loss remains higher.

### Experiment 4: Change the Learning Rate

```python
LR = 0.01     # 10× larger (faster learning but potentially unstable)
LR = 0.0001   # 1/10 (stable but slower learning)
```

### Experiment 5: Change the Number of Epochs

```python
EPOCHS = 50    # Too few leads to underfitting
EPOCHS = 1000  # Too many just causes overfitting on this small corpus
```

---

## 4.3 Changing the Generation Method

### Temperature Sampling

The `generate()` function selects the next word using `argmax` (always the highest score).
Let's change this to probabilistic sampling:

> **Note**: Here too, the prompt must consist of words from the training corpus.
> Including out-of-vocabulary words will cause an exception in the current implementation.

```python
def generate(model, prompt, vocab, id2word, max_tokens=20, temperature=1.0):
    tokens = tokenize(prompt, vocab)

    with torch.no_grad():
        for _ in range(max_tokens):
            context = tokens[-SEQ_LEN:]
            x = torch.tensor([context])
            logits = model.forward(x)
            next_logit = logits[0, -1, :] / temperature    # ← Divide by temperature

            probs = torch.softmax(next_logit, dim=-1)      # Convert to probabilities
            next_id = torch.multinomial(probs, 1).item()    # Sample according to probabilities
            tokens.append(next_id)

    return " ".join(id2word[t] for t in tokens)
```

- `temperature = 0.1`: Nearly the same as argmax (picks the most confident word)
- `temperature = 1.0`: Samples faithfully from the model's probability distribution
- `temperature = 2.0`: More random (unexpected words become more likely)

> This corpus is very small, so the differences are hard to see,
> but in real LLMs, temperature significantly affects the diversity of generated text.

---

## 4.4 Removing Weight Tying

At the end of the forward pass in `tiny_llm.py`, `tok_emb` is reused as the output projection:

```python
logits = x @ self.tok_emb.T    # Weight Tying: Reuse Embedding
```

Let's change this to an independent weight matrix. Add to `__init__`:

```python
self.out_proj = param(D_MODEL, vocab_size)   # (64, 10)
```

Change the end of forward:

```python
logits = x @ self.out_proj     # Independent output projection
```

Don't forget to also add `self.out_proj` to `parameters()`.

Compare how much the parameter count increases and whether there's a difference in loss convergence.

---

## 4.5 Further Challenges

If these experiments have given you a feel for how the Transformer works, try the following:

- **Remove Layer Norm**: Can the model learn with only residual connections?
- **Remove the causal mask**: What happens when the model can see future words during training?
- **Remove residual connections**: Change `x = x + attention(x)` to `x = attention(x)`?
- **Larger corpus**: Add more short English sentences, expanding the vocabulary to 30–50 words

Through these experiments, you should be able to feel firsthand
**why each element** of the Transformer is needed.

---

## Summary

In this tutorial:

1. **Ran the code** and verified training and generation
2. **Examined the data** at the token level
3. **Inspected the model's internals** (Embedding, Attention) as numerical values
4. **Modified the code** and experimented with the effects of hyperparameters and algorithms

tiny-LLM is a small toy, but it operates with
**exactly the same mechanism** as real LLMs like GPT.
The understanding gained here directly carries over to studying production-scale LLMs.

For a deeper understanding of the mechanisms, proceed to the [documentation](../01_data.md).
