# Chapter 4: Text Generation — Predicting the Next Word

Training is complete. All of the model's parameters have been
tuned to "correctly predict the next word."

In this chapter, we look at how the trained model **generates new text**.
This is where the essence of LLMs lies.

---

## 4.1 The Essence of LLMs: Next-Word Prediction

ChatGPT and GPT-4 are fundamentally doing the same thing:

> **"Given a sequence of words so far, predict the word that comes next."**

Just by repeating this, text is generated.

```
Input:  "the cat"
Prediction:  "sat"        ← Predict the next single word

Input:  "the cat sat"
Prediction:  "on"         ← Predict the next single word again

Input:  "the cat sat on"
Prediction:  "the"        ← Predict yet another next word

...repeat this process
```

---

## 4.2 Code — The `generate` Function

```python
def generate(model, prompt, vocab, id2word, max_tokens=20):
    tokens = tokenize(prompt, vocab)

    with torch.no_grad():                                # No gradient computation during inference
        for _ in range(max_tokens):
            context = tokens[-SEQ_LEN:]                  # Get the last 12 tokens
            x = torch.tensor([context])                  # (1, T)
            logits = model.forward(x)                    # (1, T, 10)
            next_logit = logits[0, -1, :]                # Scores at the last position
            next_id = torch.argmax(next_logit).item()    # Word with the highest score
            tokens.append(next_id)

    return " ".join(id2word[t] for t in tokens)
```

>
> **Python Tips: `" ".join(...)` — Joining a list into a string**
>
> `" ".join(list)` joins list elements with spaces into a single string:
> ```python
> words = ["the", "cat", "sat"]
> " ".join(words)     # → "the cat sat"
> "-".join(words)     # → "the-cat-sat"
> ```
> `id2word[t] for t in tokens` is a **generator expression** that
> converts each token number to a word while passing them to `join`.

### Step-by-Step Explanation

**Step 1: Tokenize the prompt**

```python
tokens = tokenize("the cat sat on", vocab)
# → [1, 2, 3, 4]
```

**Step 2: Pass through the Transformer**

```python
context = tokens[-SEQ_LEN:]    # [1, 2, 3, 4]  ← Last 12 tokens (only 4 for now)
x = torch.tensor([context])    # shape: (1, 4)
logits = model.forward(x)      # shape: (1, 4, 10)
```

Predictions are produced at all 4 positions, but we only need the **last position**.
(The last position = "the prediction after seeing all preceding context")

**Step 3: Choose the next word**

```python
next_logit = logits[0, -1, :]   # (10,) ← Scores at the last position
# Example: [0.1, 2.8, -0.2, 0.1, 0.8, 0.3, -0.3, -0.1, 0.4, 0.0]
#           pad   the   cat  sat   on   mat    .   dog   log  saw

next_id = torch.argmax(next_logit).item()   # → 1 (= "the")
```

>
> **Python Tips: Multi-dimensional tensor indexing `logits[0, -1, :]`**
>
> Specify positions along each axis with commas. `-1` means "last", `:` means "all":
> ```python
> x = torch.zeros(3, 4, 10)   # 3 samples × 4 positions × 10 words
>
> x[0]         # → shape: (4, 10)   Entire first sample
> x[0, -1]     # → shape: (10,)     Last position of first sample
> x[0, -1, :]  # → shape: (10,)     Same (: means "all", so it can be omitted)
> x[0, -1, 3]  # → scalar           A specific single element
> ```

>
> **Python Tips: `torch.argmax()` — Index of the maximum value**
>
> Returns the **position (index)** of the largest value in a tensor:
> ```python
> scores = torch.tensor([0.1, 0.3, 2.1, -0.5, 0.8])
> torch.argmax(scores)          # → tensor(2)   ← 2.1 is the max, at position 2
> torch.argmax(scores).item()   # → 2            ← .item() to get a Python int
> ```

`argmax` returns the index with the highest score.
→ The next word after "the cat sat on" is predicted as "the" (in the corpus, "the" always follows "on").

**Step 4: Append to token sequence and repeat**

```python
tokens.append(1)
# tokens = [1, 2, 3, 4, 1]  ← "the cat sat on the"
# → Next loop predicts the next word from "the cat sat on the"
```

### Generation Flow (Concrete Example)

```
"the cat sat on"
                  → Predict: "the"  → "the cat sat on the"
                  → Predict: "mat"  → "the cat sat on the mat"
                  → Predict: "."    → "the cat sat on the mat ."
                  → Predict: "the"  → "the cat sat on the mat . the"
                  → Predict: "dog"  → "the cat sat on the mat . the dog"
                  ...
```

---

## 4.3 Limitations of Greedy Decoding

This program uses `argmax` (selecting the word with the highest score).
This is called **Greedy Decoding**.

```
Scores: [0.1, 0.3, -0.2, 0.1, 0.8, 2.1, -0.3, -0.1, 0.4, 0.0]
                                          ↑
                                     Always picks this one
```

It's simple, but because it always picks only "the most probable word,"
it tends to loop the same patterns.

> **Real LLMs** use techniques like sampling from the probability distribution (temperature),
> or choosing from the top k candidates (top-k) to produce more diverse outputs.

---

## 4.4 Examining the Output

```
prompt: "the cat sat on"
output: the cat sat on the mat . the dog sat on the log .
        the cat saw the dog . the dog saw the

prompt: "the dog saw"
output: the dog saw the cat . the cat sat on the log .
        the dog sat on the mat . the dog sat
```

Natural sentences consistent with the training corpus are generated.
It may look like the model is just memorizing the corpus —
and that's actually correct. With only 40 words and 10 vocabulary items, memorization is the optimal solution.

---

## 4.5 From Here to Real LLMs

The difference between tiny-LLM and real LLMs like GPT-4 is
**not a difference in fundamental mechanism, but a difference in scale**.

| | tiny-LLM | GPT-4 class |
|---|---|---|
| Vocabulary size | 10 | 100,000+ |
| Embedding dimension | 64 | 12,288+ |
| Attention heads | 4 | 96+ |
| Transformer layers | 2 | 96+ |
| Parameters | ~68,000 | Hundreds of billions to trillions |
| Training data | 40 words | Trillions of tokens |
| Training time | Seconds | Months (thousands of GPUs) |

However, the core algorithms are **exactly the same**:

1. Convert words to vectors (Embedding)
2. Understand context with Self-Attention (Q, K, V)
3. Transform representations with Feed-Forward
4. Train with a "next word" prediction loss
5. Generate one word at a time with the trained model

When you scale up, instead of memorization, **generalization** begins to emerge.
Being able to predict appropriate next words for "sentences never seen before,"
based on learned patterns — that is the power of large language models.

---

## Summary

```
"the cat sat on"
       ↓
   Transformer (trained)
       ↓
   Predict "the"
       ↓
"the cat sat on the"
       ↓
   Transformer
       ↓
   Predict "."
       ↓
   ...repeat
```

**Everything about LLMs comes down to "predicting the next word."**

- Embedding gives words meaning
- Self-Attention understands context
- Training improves prediction accuracy
- Generation repeats prediction

tiny-LLM is a small toy, but
the essence of the Transformer — Self-Attention, Q/K/V, residual connections,
Layer Norm, and "next-word prediction" — is exactly the same as the real thing.
