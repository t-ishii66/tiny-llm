# Step 5: Try Instruction Tuning

Run the Alpaca-style instruction tuning from Chapter 5 by hand and see how a pretrained model picks up the "follow an instruction" behavior from just 4 instruction/response pairs.
You'll also see firsthand that the result is **rote memorization** — there is no real generalization like a production LLM has.

---

## 5.1 Run It

```bash
uv run --with torch tiny_llm_instruct.py
```

Three stages run in sequence:

```
--- Stage 1: Pretraining ---        ← Plain language modeling (200 epochs)
epoch   20  loss=...
...
epoch  200  loss=...

--- Stage 2: Instruction tuning --- ← Fine-tune with response masking (300 epochs)
epoch   30  loss=...
...
epoch  300  loss=...

--- Stage 3: Responding ---         ← Generate responses after tuning
### Instruction: who sat on the mat
### Response: the cat

### Instruction: who saw the dog
### Response: the cat

### Instruction: who sat on the log
### Response: the dog
```

Confirm that both Stage 1 and Stage 2 losses are clearly dropping.
The two loss values aren't directly comparable (different loss functions — full sequence vs. response-only) — just check **each one converges** on its own.

---

## 5.2 Predict First, Then Verify

The `examples` list in `tiny_llm_instruct.py` shows the entire training set — just 4 pairs:

```python
examples = [
    ("who sat on the mat", "", "the cat"),
    ("who sat on the log", "", "the dog"),
    ("who saw the dog",    "", "the cat"),
    ("who saw the cat",    "", "the dog"),
]
```

Stage 3 prompts the model with 3 instructions. **Predict the responses yourself first**, then check against the actual output:

| Instruction | Your prediction | Actual output |
|---|---|---|
| `who sat on the mat` | ? | ? |
| `who saw the dog` | ? | ? |
| `who sat on the log` | ? | ? |

Since the model has fully memorized the 4 training pairs, all 3 responses should match the training data exactly.

---

## 5.3 Add a New Instruction Example

Add one more example to the `examples` list in `tiny_llm_instruct.py`:

```python
examples = [
    ("who sat on the mat", "", "the cat"),
    ("who sat on the log", "", "the dog"),
    ("who saw the dog",    "", "the cat"),
    ("who saw the cat",    "", "the dog"),
    ("who saw the mat",    "", "the dog"),   # ← added
]
```

And add a matching prompt to the Stage 3 test list:

```python
for ins in [
    "who sat on the mat",
    "who saw the dog",
    "who sat on the log",
    "who saw the mat",    # ← added
]:
```

Re-run:

```bash
uv run --with torch tiny_llm_instruct.py
```

If the 5th example is memorized and `who saw the mat` produces `the dog`, you've succeeded.
Just one extra example teaches the model a brand-new instruction/response pattern.

> **Note**: If you put a word not in the corpus (e.g. `bird`) into the instruction or response, you'll get a `KeyError`.
> The tokenizer in this project does not handle unknown words, so stick to the existing vocabulary (`the`, `cat`, `sat`, `on`, `mat`, `.`, `dog`, `log`, `saw`, `who`).

---

## 5.4 Try Out-of-Distribution Instructions — See the Limits of Memorization

Now try instructions the model was never trained on.

First **revert** the changes from 5.3 so `examples` is back to the original 4. Then change the Stage 3 test instructions to ones not in the training set:

```python
for ins in [
    "who sat on the cat",    # ← not in training (vocab is fine though)
    "who saw the log",       # ← not in training
    "who sat on the dog",    # ← not in training
]:
```

Re-run and observe the responses. Common patterns:

- The model **parrots** a training response (e.g. always answers `the cat`)
- The output is strongly biased toward the most recently trained example
- The output is gibberish

The model has only seen 4 training examples — there is no real generalization to instructions it hasn't seen. This is where you **feel the "rote memorization" reality** of this toy model.

> **Real LLMs look "smart"** only because they have training data in the millions to tens of millions of examples, and orders-of-magnitude more model capacity.
> The mechanism is **identical** to what you just ran.

---

## 5.5 Reduce the Instruction-Tuning Epochs

Finally, see how much memorization is needed before the model produces correct responses.

Change the `train_instruct` call in the `if __name__ == "__main__":` block:

```python
# --- Stage 2: Instruction tuning ---
print("\n--- Stage 2: Instruction tuning ---")
examples = [...]
train_instruct(model, examples, vocab, epochs=30)   # ← try 30, 50, 100, ...
```

| epochs | Expected behavior |
|---|---|
| 300 (default) | All 3 prompts produce the correct response (memorization complete) |
| 100 | Mostly correct, occasionally breaks down |
| 50 | Only about half are remembered |
| 10 | Almost no learning; output is broken |

With low epoch counts, you'll see intermediate states like "format is correct but the response content is wrong."
Watching the loss curve at each setting gives you a feel for how much compute instruction tuning actually requires.

---

## Summary

In this tutorial you saw:

1. The full **pretrain → instruction tune → respond** pipeline in action
2. That adding **just one new example** teaches the model an entirely new instruction/response pair
3. That **out-of-distribution instructions** trigger parroting or breakdown — the "rote memorization" limit
4. How the model behaves at **partial training** — instruction tuning's learning curve under your eyes

> **The result is rote memorization, but the structure is the real thing.**
> ChatGPT, Alpaca, Vicuna — they're all built with the same "format-delimit + response-only loss" procedure you ran.
> The only difference is **scale** (data volume and model capacity).

That's the end of the tiny-LLM tutorial.
Going back to the main documentation now, you'll find the math and architecture behind everything you just ran:

- [Back to the documentation](../01_data.md)
- [Chapter 5: Instruction Tuning (main)](../05_instruction_tuning.md)
