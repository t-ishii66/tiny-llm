# Step 5: Try Instruction Tuning

![Try Instruction Tuning](../../images/tutorial-05-instruction.png)

Let's actually run the Alpaca-style Instruction Tuning we saw in Chapter 5 (the main text) and get a feel for it.
We observe a pretrained model learning 4 instruction × response pairs until it acquires "instruction-following" behavior.
At the same time, we confirm with our own eyes that the result is "rote memorization" and that there is no generalization like that of a real LLM.

> From Step 5 the code file we work with changes. We run `tiny_llm_instruct.py` rather than `tiny_llm.py`
> (the contents are a separate file that imports `tiny_llm.py` and adds the processing for instruction tuning).

---

## 5.1 Running It

```bash
uv run --with torch tiny_llm_instruct.py
```

Three Stages run in order.

```
--- Stage 1: Pretraining ---        ← ordinary language model training (200 epochs)
epoch   20  loss=...
...
epoch  200  loss=...

--- Stage 2: Instruction tuning --- ← fine-tuning with a Response mask (300 epochs)
epoch   30  loss=...
...
epoch  300  loss=...

--- Stage 3: Responding ---         ← response generation after training
### Instruction: who sat on the mat
### Response: (we'll check this in §5.2 — don't look yet)

### Instruction: who saw the dog
### Response: (we'll check this in §5.2 — don't look yet)

### Instruction: who sat on the log
### Response: (we'll check this in §5.2 — don't look yet)
```

Please confirm that the loss comes down solidly in both Stage 1 and Stage 2.
The losses of Stage 1 and Stage 2 use different loss functions (all positions / response positions only), so comparing their absolute values is meaningless. It's enough to look only at **whether each of them has come all the way down**.

(For Stage 3's actual responses, we'll first predict them in §5.2 and then check the answers.)

---

## 5.2 Predict the Output, Then Check It

Looking at the `examples` list in `tiny_llm_instruct.py`, the training data is only these 4 examples:

```python
examples = [
    ("who sat on the mat", "", "the cat"),
    ("who sat on the log", "", "the dog"),
    ("who saw the dog",    "", "the cat"),
    ("who saw the cat",    "", "the dog"),
]
```

In Stage 3 the model produces responses for 3 instructions. **Predict them yourself before running**, then check against the output.

| Instruction | Your prediction | Actual output |
|---|---|---|
| `who sat on the mat` | ? | ? |
| `who saw the dog` | ? | ? |
| `who sat on the log` | ? | ? |

Since it has completely memorized them in training, all 3 should come back with the responses exactly as in the training data.

---

## 5.3 Adding a New Instruction Example

Let's add one example to the `examples` list in `tiny_llm_instruct.py`:

```python
examples = [
    ("who sat on the mat", "", "the cat"),
    ("who sat on the log", "", "the dog"),
    ("who saw the dog",    "", "the cat"),
    ("who saw the cat",    "", "the dog"),
    ("who saw the mat",    "", "the dog"),   # ← added
]
```

Add the corresponding one to the Stage 3 test instructions as well:

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

If the 5th example is also memorized and `the dog` comes back for `who saw the mat`, you've succeeded.
We've confirmed that adding just one example lets the model memorize a new instruction × response pattern.

> As mentioned in Step 1.4, putting **a word not in the corpus** into an instruction or response you add results in a `KeyError`.
> Please combine within the range of the existing vocabulary (`the`, `cat`, `sat`, `on`, `mat`, `.`, `dog`, `log`, `saw`, `who`).

---

## 5.4 Trying OOD Instructions — Seeing the Limits of Rote Memorization

Let's try instructions that aren't in the training data and see how the model behaves.

**Revert** the line you added in §5.3, returning `examples` to its original 4 lines, then change only the Stage 3 test instructions:

```python
for ins in [
    "who sat on the cat",    # ← an instruction not in training (all vocabulary is known)
    "who saw the log",       # ← an instruction not in training
    "who sat on the dog",    # ← an instruction not in training
]:
```

Re-run and observe what responses come out. Commonly seen patterns:

- It **parrots back** the response of an existing training example (e.g. everything becomes `the cat`)
- It is strongly pulled toward the response of the last training example it learned
- A meaningless word sequence comes out

Since it has learned only the 4 training examples, **it cannot generalize meaningfully** to unknown instructions. This is a limit of tiny-LLM's size and amount of training, and it's the moment where you can **feel the message that "this is a rote-memorization model"**.

---

## 5.5 Reducing the Number of Instruction Tuning Epochs

Finally, let's observe "how much rote memorization it takes before it can respond".

Change the line that calls `train_instruct` in Stage 2 of the `if __name__ == "__main__":` block:

```python
# --- Stage 2: Instruction tuning ---
print("\n--- Stage 2: Instruction tuning ---")
examples = [...]
train_instruct(model, examples, vocab, epochs=30)   # ← change to 30, 50, 100, etc.
```

| epochs | Expected behavior |
|---|---|
| 300 (original) | correct responses for all 3 examples (memorization complete) |
| 100 | mostly correct, but occasionally falls apart |
| 50 | has memorized only about half |
| 10 | has barely learned anything; the output falls apart |

With few epochs you can observe the intermediate state where "it produces the format, but the response content is odd".
While also watching the loss convergence curve, get a feel for how much computation instruction tuning takes to complete.

---

## 5.6 (Optional) Designing a Different Task with the Same Vocabulary

This is an exercise in devising your own `(instruction, response)` pairs using only combinations of the existing vocabulary (`the cat sat on mat . dog log saw who`).

For example:

```python
examples = [
    ("what did the cat see", "", "the dog"),
    ("what did the dog see", "", "the cat"),
    ("where did the cat sit", "", "on the mat"),
    ("where did the dog sit", "", "on the log"),
]
```

However, the example above contains the words `what`, `did`, and `where`, which have never appeared, so as-is it results in a `KeyError`. To get it through, the choice is either to add "`what did where`" as dummy occurrences at the end of the corpus so they get registered in the vocabulary, or to build the instructions using only words within the vocabulary.

> Once you take on the perspective of "**designing the training data yourself**", you get a real feel for what it means when
> people say of real instruction tuning that "the quality and diversity of the dataset determine the result".

If you have time to spare, try creating your own examples list, training on it, and seeing how well it can memorize.

---

## Summary — Looking Back Over the Whole Tutorial

Through the 5 Steps, you experienced the following **hands-on**:

1. **Step 1**: confirmed that a single `uv run` gets training and generation running
2. **Step 2**: observed tokenization and the sliding-window structure of the training data as tensors
3. **Step 3**: pulled out the Embedding and Attention weights and confirmed that the model runs on "numeric tensors"
4. **Step 4**: changed hyperparameters and structure (Weight Tying, Temperature) and measured the effect on loss and generation
5. **Step 5**: ran the 3 stages of pretrain → instruction tuning → responding and saw that Response masking creates "instruction-following" behavior. At the same time, you felt "the limits of rote memorization"

The result was rote memorization, but the takeaway of this tutorial is that you assembled the procedure itself ("delimit with a format + loss on the response only") with your own hands.

This is the end of the tiny-LLM tutorial.
If you go back to the main documentation and reread each chapter, you should understand the mathematics and architecture behind the behavior you ran here much more deeply.

- [The start of the documentation](../01_data.md)
- [Chapter 5 main text (Instruction Tuning)](../05_instruction_tuning.md)
