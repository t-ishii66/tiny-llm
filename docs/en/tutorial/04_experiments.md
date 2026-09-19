# Step 4: Experiments and Modifications

![Experiments and Modifications](../../images/tutorial-04-experiments.png)

By now the workings of the Transformer have come into view. Let's modify the code and experiment.

> **From here on, the working style changes.**
> Up through Steps 2–3, the flow was to enter interactive mode with `uv run --with torch python -i tiny_llm.py` and observe the trained model on the spot.
> In Step 4 we run the cycle of **editing `tiny_llm.py` directly in an editor → saving → retraining with `uv run --with torch tiny_llm.py` → comparing the output** (we don't use interactive mode).
> After trying each section, reverting your change before moving on to the next section makes the effect of each individual element easier to see.

---

## 4.1 Changing the Corpus

Let's change the corpus in the `if __name__ == "__main__":` block of `tiny_llm.py`:

```python
# the original corpus
corpus = (
    "the cat sat on the mat . the dog sat on the log . "
    "the cat saw the dog . the dog saw the cat . "
    "the cat sat on the log . the dog sat on the mat ."
)
```

For example, let's add a new pattern:

```python
corpus = (
    "the cat sat on the mat . the dog sat on the log . "
    "the cat saw the dog . the dog saw the cat . "
    "the cat sat on the log . the dog sat on the mat . "
    "the bird sat on the log . the bird saw the cat ."
)
```

Let's run it and check whether "bird" gets learned correctly:

```bash
uv run --with torch tiny_llm.py
```

> **Note**: adding a new word changes the vocabulary size (10 → 11).
> The code itself detects the vocabulary size automatically, so it works as-is.

---

## 4.2 Changing the Hyperparameters

Let's change the hyperparameters at the top of `tiny_llm.py` and observe how the training result changes.
**Writing down your "prediction" first, then running it → and lining it up against the measured value** increases how much you learn.

| Setting | Your prediction of the final loss at 200 epochs | Measured |
|---|---|---|
| default (N_HEADS=4, N_LAYERS=2, D_MODEL=64, LR=0.001) | around 0.1 | ? |
| N_HEADS=1 | a bit higher (0.2–0.5)? | ? |
| N_LAYERS=1 | close to the default? | ? |
| D_MODEL=16, D_FF=32 | bottoms out due to lack of expressive power? | ? |
| LR=0.01 | fast but unstable? | ? |
| LR=0.0001 | too slow to come all the way down? | ? |
| EPOCHS=50 | undertrained, stops at a higher value? | ? |

The places to change each setting are:

```python
N_HEADS = 1     # only 1 head (no multi-head)
N_LAYERS = 1    # only 1 layer
D_MODEL = 16    # shrink from 64 to 16; set D_FF = 32 as well
LR = 0.01       # 10× larger
LR = 0.0001     # 1/10
EPOCHS = 50     # too few
EPOCHS = 1000   # too many (overfitting)
```

> **Tip**: change just one thing and run once → record the value → revert to the default and go to the next, repeating this,
> and you can see the effect of a single element. If you change several at once, you won't know what made the difference.

---

## 4.3 Changing the Generation Method

### Temperature Sampling

In the `generate()` function, the next word is chosen with `argmax` (always the highest score).
Let's change that into probabilistic sampling:

```python
def generate(model, prompt, vocab, id2word, max_tokens=20, temperature=1.0):
    tokens = tokenize(prompt, vocab)

    with torch.no_grad():
        for _ in range(max_tokens):
            context = tokens[-SEQ_LEN:]
            x = torch.tensor([context])
            logits = model.forward(x)
            next_logit = logits[0, -1, :] / temperature    # ← divide by temperature

            probs = torch.softmax(next_logit, dim=-1)      # convert into probabilities
            next_id = torch.multinomial(probs, 1).item()    # sample according to the probabilities
            tokens.append(next_id)

    return " ".join(id2word[t] for t in tokens)
```

> **What `torch.multinomial(probs, 1)` is**: a function that draws 1 sample according to the probability distribution `probs`.
> For example, with `probs = [0.7, 0.2, 0.1]` it returns 0/1/2 with probability 70%/20%/10%.
> With `argmax` only 0 would ever come out, but using `multinomial`, other candidates can also be chosen depending on the distribution at that moment, so fluctuation appears in the generation.

- `temperature = 0.1`: almost the same as argmax (picks the high-confidence word)
- `temperature = 1.0`: samples faithfully from the model's probability distribution
- `temperature = 2.0`: more random (surprising words come out more easily)

> This corpus is so small that the difference is hard to see, but
> in real LLMs, temperature greatly influences the diversity of the generated text.

---

## 4.4 Removing Weight Tying

At the end of the Forward Pass in `tiny_llm.py`, `tok_emb` is reused for the output projection:

```python
logits = x @ self.tok_emb.T    # Weight Tying: reuse the Embedding
```

Let's change this into an independent weight matrix. Add a new `out_proj` right after `# --- Embeddings ---` in `TinyTransformer.__init__`:

```python
# --- Embeddings ---
self.tok_emb = param(vocab_size, D_MODEL)
self.pos_emb = param(SEQ_LEN, D_MODEL)
self.out_proj = param(D_MODEL, vocab_size)   # ← added (64, 10)
```

Change the end of `forward` to use `out_proj`:

```python
logits = x @ self.out_proj     # ← remove Weight Tying, use an independent output projection
```

Finally, include `out_proj` among the trainable parameters in `parameters()`:

```python
def parameters(self):
    params = [self.tok_emb, self.pos_emb, self.out_proj,   # ← added out_proj
              self.ln_f_g, self.ln_f_b]
    for layer in self.layers:
        params.extend(layer.values())
    return params
```

Run it and compare

- how much the parameter count (`sum(p.numel() for p in model.parameters())`, which we saw in §3.1) increases
- how the loss convergence curve differs from the default

By the numbers, it should increase by `D_MODEL * vocab_size = 64 * 10 = 640` parameters.

---

## 4.5 Further Challenges

Once these experiments have given you a feel for how the Transformer works, try the following too:

- **Remove Layer Norm**: can it still learn with only the residual connections?
- **Remove the causal mask**: what happens if you train with the future words visible?
- **Remove the residual connections**: what if you change `x = x + attention(x)` into `x = attention(x)`?
- **A larger corpus**: add more short English sentences and grow the vocabulary to 30–50

Through these experiments, you should get a real sense of
**why** each element of the Transformer is necessary.

---

With that, one full round of training and running a "bare language model" is complete.
Next, let's try instruction tuning and see how the "ChatGPT-like instruction-response" behavior is created.

Next: [Step 5: Try Instruction Tuning](05_instruction.md)
