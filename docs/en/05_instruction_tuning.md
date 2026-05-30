# Chapter 5: Instruction Tuning — How to Build an LLM That Follows Instructions

Through Chapter 4, the Transformer was completed as a model that **"predicts the next word."**
It can produce output that looks like memorized snippets of the corpus,
but it doesn't yet behave like ChatGPT, which "answers the user's question."

In this chapter, we look at the mechanism that, by adding one more stage of training,
shapes the model into one that **"follows instructions."**
We use the format standardized by Stanford's Alpaca project.

> The code in this chapter lives in a new file `tiny_llm_instruct.py`.
> The `tiny_llm.py` from Chapter 4 and earlier is imported and reused as-is.

---

## 5.1 Pretraining vs Instruction Tuning

Modern LLMs are typically built with **two stages of training**.

**Stage 1: Pretraining**

- Learn "next-word prediction" on a huge natural-language corpus (web, books, papers…)
- Exactly the approach we saw through Chapter 3
- Here the model learns **"language"**
- Once trained, it becomes a foundation that can be used for many tasks

**Stage 2: Instruction Tuning**

- Take the pretrained model and **further train** it in an "instruction → response" format
- Here the model learns **"to follow instructions"**
- It becomes able to respond to instructions like "translate this," "summarize this," "explain X"

```
[Untrained model]
        ↓ Pretraining (huge corpus)
[Pretrained model: has learned language]
        ↓ Instruction Tuning (instruction × response pairs)
[Instruction-tuned model: can follow instructions]
```

What's notable about the mechanism:

> **The model and the Cross-Entropy Loss are the same as through Chapter 4.**
> The only things that change are **the shape of the training data** and **at which positions the loss is computed**.

In other words, all of this is achieved by applying the basic mechanism of "next-word prediction."

> ### ⚠️ Most Important Point — The "pretrained model" is **literally the same `model` object**
>
> In stage 2 "Instruction Tuning" in the diagram above, **we do not create a new model**.
> We pass the `model` variable trained in Stage 1 **as-is** into Stage 2 and continue training from there.
>
> ```python
> model = TinyTransformer(len(vocab))   # ← The model is created here, exactly "once"
> train(model, ...)                      # Stage 1: weights updated by pretraining
> train_instruct(model, ...)             # Stage 2: fine-tune the same model (inheriting the weights!)
> ```
>
> If you mistakenly re-run `model = TinyTransformer(...)` in Stage 2, **all** the language knowledge learned in Stage 1 will be wiped. With only 4 instruction examples you can't even learn English word order, so the generated output becomes completely nonsensical.
>
> **This is exactly what the word fine-tune (= "additional adjustment" of existing weights) means.** Don't throw away the Stage 1 weights; paint over them. That's the heart of instruction tuning.

---

## 5.2 The Alpaca Format

To teach a model "instruction → response," it's standard to use a **template with a fixed format**.
The format proposed by Stanford's Alpaca project is the following.

```
### Instruction: <instruction>
### Input: <context or input (as needed)>
### Response: <expected response>
```

For tasks that don't need an Input, the Input section is omitted.

**Example with Input:**

```
### Instruction: Answer with the subject
### Input: The cat sat on the mat.
### Response: The cat
```

**Example without Input:**

```
### Instruction: Compute 7 * 8
### Response: 56
```

### Why a "Format" Is Needed

When there are fixed delimiters, the model finds it easier to learn
**"input ends here / response starts here."**

- During training: the model learns that the "expected response" comes **immediately after** `### Response:`
- At inference time: if you give it up through `### Instruction: <question> ### Response:`, it generates the continuation

In other words, the marker `### Response:` functions as the **signal that the response begins**.

---

## 5.3 Assembling a Toy Dataset

This project's vocabulary originally has only 10 words.

```python
{"<pad>": 0, "the": 1, "cat": 2, "sat": 3, "on": 4,
 "mat": 5, ".": 6, "dog": 7, "log": 8, "saw": 9}
```

To this we add the Alpaca-format markers and the words used by our task.

| Added token | Role |
|---|---|
| `###` | Format-delimiter symbol |
| `Instruction:` | Marker that starts the instruction section |
| `Input:` | Marker that starts the input section (unused in our toy example) |
| `Response:` | Marker that starts the response section |
| `who` | Question word for "who …?" |

This brings the vocabulary to **15** in total.

> **Reading the numbers:** As described in the "reading the numbers" table in §1.2 of Chapter 1, this project is designed so that the key numbers don't overlap with each other.
> Through Chapters 1–4 `vocab_size = 10`, but in Chapter 5 the 4 Alpaca markers plus `who` extend it to **`vocab_size = 15`**.
> **The value 15 does not overlap with any of the other parameters (2, 4, 10, 12, 16, 28, 64, 128)**, so from Chapter 5 onward, when you see 15 in a tensor shape you can immediately tell "that's the vocabulary size."
> For example, when you see `logits.shape = (4, 12, 15)` you can read it as "scores for 4 samples × 12 tokens × 15 vocabulary items."

> **How the tokenizer works (recap)**
>
> As we saw in Chapter 1, this project's tokenizer **just splits on whitespace**.
> So `### Instruction:` is split into **2 tokens**: `###` and `Instruction:`.
> (With a subword tokenizer like BPE, `###` and `Instruction` would be split into even finer units, but the flow of processing is the same.)

### A Toy Instruction Dataset

We use a toy dataset of only 4 examples.

```python
examples = [
    ("who sat on the mat", "", "the cat"),
    ("who sat on the log", "", "the dog"),
    ("who saw the dog",    "", "the cat"),
    ("who saw the cat",    "", "the dog"),
]
```

Each tuple is a triple of `(instruction, input, response)`.
This time we deliberately used only simple examples that don't use Input.

Expanded into Alpaca-format strings:

```
### Instruction: who sat on the mat ### Response: the cat
### Instruction: who sat on the log ### Response: the dog
### Instruction: who saw the dog ### Response: the cat
### Instruction: who saw the cat ### Response: the dog
```

The `format_alpaca` function performs this formatting:

```python
def format_alpaca(instruction, input_text, response):
    if input_text:
        return (f"### Instruction: {instruction} "
                f"### Input: {input_text} "
                f"### Response: {response}").strip()
    return (f"### Instruction: {instruction} "
            f"### Response: {response}").strip()
```

---

## 5.4 Response Masking — Focusing the Loss on "the Response Only"

This is the **most important point** of the chapter.

If you compute Cross-Entropy Loss at all positions in the usual way, the model ends up learning
**"to predict the instruction text too."**
This is wasteful, and it can be harmful.

- It wastes learning capacity on "the language patterns of the instruction portion"
- After training, the model can develop the behavior of "generating an instruction sentence"

What we want is:

> **"When an instruction arrives, generate the response as its continuation."**

That's all. So we compute the loss **only at the positions of the response tokens**.
At the remaining positions we **zero them out with a mask** and let them contribute nothing to gradient computation.

### Understanding via the Diagram

Let's look at how the data tensor for one example lines up at each position.

- **`inp`**: the input to the model. In the forward pass, every token the model "sees" (position 0 through seq_len-1)
- **`tgt`**: the correct answer we want predicted at each position. `inp` shifted by one (the prediction target at position `i` is `inp[i+1]`)
- **`mask`**: a 0/1 indicating whether the loss at each position is included in the final loss (multiplied with `tgt`)

```
position:  0    1    2    3    4    5    6    7    8    9    10   11
inp:       ###  Ins  who  sat  on   the  mat  ###  Res  the  cat  <pad>
tgt:       Ins  who  sat  on   the  mat  ###  Res  the  cat  <pad><pad>
mask:      0    0    0    0    0    0    0    0    1    1    1    0
```

Key points:

- Looking at the `tgt` side, positions 8 and 9 are the **response tokens** (`the`, `cat`). → mask=1 (compute the loss)
- Position 10 (`tgt[10] = <pad>`) is also included in the loss as the **stop signal immediately after the response**. → mask=1
  Without this, the model wouldn't learn during training "to stop after `cat`," and at inference time it would keep generating something all the way up to max_tokens
- The other `tgt` positions — the instruction text, the `### Response:` marker, and the second and later `<pad>`s — don't have their loss computed. → mask=0
- An important separation here: **the instruction and the `### Response:` marker themselves do exist in `inp`, and the model sees them (conditioning)**.
  But we don't make them "things to predict" (mask=0).

In other words, the heart of instruction tuning is to **handle separately what we show the model (`inp`) and what we make it learn (the parts of `tgt` selected by `mask`)**.

The `mask` row above becomes `torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0])` as-is.
When you multiply this with the per-token loss, only the positions with 1 contribute to the final loss.

---

## 5.5 Training Code

### `build_example` — Building (input, target, mask) from one example

Arguments:

- `instruction`, `input_text`, `response`: the 3 Alpaca elements (e.g. `"who sat on the mat"`, `""`, `"the cat"`)
- `vocab`: the word → ID mapping built in Chapter 1
- `seq_len`: length of the output tensor. Pass the same value as **the model's context window size** (in this project, `SEQ_LEN = 12` defined in `tiny_llm.py`). Its role is to pad short examples with `<pad>` and truncate the end of overly long examples so that all examples line up to the same length

Returns: `inp`, `tgt`, `mask` (lists, each of length `seq_len`)

```python
def build_example(instruction, input_text, response, vocab, seq_len):
    pad = vocab["<pad>"]

    # How many tokens are in the prefix up through "### Response:"
    prefix_text = format_alpaca(instruction, input_text, "")
    P = len(tokenize(prefix_text, vocab))

    # Full text ("### Instruction: ... ### Response: <response>")
    full_text = format_alpaca(instruction, input_text, response)
    full_ids = tokenize(full_text, vocab)
    F_len = len(full_ids)

    # Pad/truncate to length seq_len + 1 (the extra one is for the shift)
    if F_len > seq_len + 1:
        full_ids = full_ids[:seq_len + 1]
        F_len = seq_len + 1
    full_ids = full_ids + [pad] * (seq_len + 1 - len(full_ids))

    inp = full_ids[:seq_len]         # input (T)
    tgt = full_ids[1:seq_len + 1]    # target (T) — shifted by one

    # Mask: 1 at the response-token positions + the single <pad> immediately after the response
    # The response tokens are at indices [P, F_len - 1] in full_ids,
    # and we also want to teach the trailing <pad> as a "stop signal," so include it
    # In target indexing this is [P - 1, F_len - 1]
    mask = [1 if (P - 1) <= i <= (F_len - 1) else 0 for i in range(seq_len)]

    return inp, tgt, mask
```

#### Example of arguments and return values

Calling it with the first example `("who sat on the mat", "", "the cat")` gives the following.

```python
inp, tgt, mask = build_example(
    instruction="who sat on the mat",
    input_text="",
    response="the cat",
    vocab=vocab,      # vocab_size = 15 ( ### → 10, Instruction: → 11, who → 14, etc.)
    seq_len=12,
)

# Return values (all 3 are lists of length 12):
inp  == [10, 11, 14,  3,  4,  1,  5, 10, 13,  1,  2,  0]
tgt  == [11, 14,  3,  4,  1,  5, 10, 13,  1,  2,  0,  0]
mask == [ 0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  0]

# Converted back to words:
# inp : ###  Instruction:  who  sat  on  the  mat  ###  Response:  the  cat   <pad>
# tgt : Instruction:  who  sat  on  the  mat  ###  Response:  the  cat  <pad>  <pad>
```

Same content as the diagram in 5.4.

- The prefix `### Instruction: who sat on the mat ### Response:` is 9 tokens → `P = 9`
- The whole `### Instruction: ... ### Response: the cat` is 11 tokens → `F_len = 11`
- mask: `(P - 1) <= i <= (F_len - 1)`, i.e. `8 <= i <= 10` → positions 8, 9, 10 are 1
- `tgt[8] = "the"` (1st response token), `tgt[9] = "cat"` (2nd response token), `tgt[10] = <pad>` (stop signal)
  ← Only these 3 positions contribute to the loss

#### Aside: Why we go through `format_alpaca`

If you look carefully at `build_example`, you might notice that even though we receive a structured tuple `(instruction, input_text, response)`, we deliberately format it into a string with `format_alpaca` and then turn it back into a token sequence with `tokenize`. If we built the token sequence directly we could skip the string step, and that would be faster in pure efficiency terms.

We take this detour anyway because:

- **To match the standard data pipeline**: in normal instruction tuning, you read an external Alpaca dataset (e.g. the 52,000 examples in `alpaca_data.json`) **as strings**, then tokenize them and build the mask. tiny-LLM just starts from Python tuples instead of an external file, but the flow from there is the same
- **To share the same format function between training and inference**: `respond` in 5.6 also assembles its prompt with the same `format_alpaca`. If you implement formatting separately for training and inference, even small formatting mismatches tend to produce bugs, so we want to share the code
- **It simplifies computing the boundary position `P`**: tokenizing the formatted prefix string and taking `len(...)` is all you need to get the prefix length

### `train_instruct` — Training loop with masking

```python
def train_instruct(model, examples, vocab, epochs=300, lr=LR):
    inputs, targets, mask = make_instruction_batch(examples, vocab)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        logits = model.forward(inputs)                   # (B, T, V)

        # Compute per-token cross-entropy at all positions (reduction='none')
        per_tok = F.cross_entropy(
            logits.view(-1, model.vocab_size),
            targets.view(-1),
            reduction="none",
        ).view(targets.shape)                            # (B, T)

        # Apply the mask so that only the response positions contribute to the loss
        loss = (per_tok * mask).sum() / mask.sum().clamp(min=1.0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

Almost identical to the `train` function in Chapter 3. The only difference is **the 2 lines that multiply the per-token loss by the mask**.

>
> **Python Tips: `F.cross_entropy(..., reduction="none")`**
>
> By default, `cross_entropy` returns a scalar that is the **average** of the loss over all positions.
> If you pass `reduction="none"`, it returns the loss at each position as a tensor.
> Multiply that by a mask and you can keep only the positions you want.

>
> **Python Tips: `mask.sum().clamp(min=1.0)`**
>
> The sum of the mask is "the total number of positions where the loss is computed."
> To **prevent division by zero** in the unlikely event that everything is 0,
> we use `.clamp(min=1.0)` to pin the lower bound at 1.0.

#### Concrete example: the contents of `inputs` / `per_tok` / `loss`

Suppose the training data had just 1 example (`who sat on the mat` → `the cat`). Let's trace what values each tensor takes. Assume `SEQ_LEN = 12`.

**`inputs[0]` (the input the model sees)** — the 12 numbers obtained by tokenizing `### Instruction: who sat on the mat ### Response: the cat <pad>`:

```
position : 0      1      2      3      4      5      6      7      8      9      10     11
inputs[0]: ###    Inst:  who    sat    on     the    mat    ###    Resp:  the    cat    <pad>
targets  : Inst:  who    sat    on     the    mat    ###    Resp:  the    cat    <pad>  <pad>
mask[0]  : 0      0      0      0      0      0      0      0      1      1      1      0
```

**`per_tok[0]` (cross-entropy at each position)** — shape `(B, T)`, here `T = 12`. Each value is **the model's -log(probability) of the correct token at that position**. Smaller means the model is more confident; larger means the model is failing to get it right:

```
per_tok[0] = [2.31, 1.45, 0.92, 0.31, 0.08, 0.55, 0.12, 0.83, 1.05, 0.72, 0.41, 1.98]
```

The meaning of each value (position `i` is a 0-based index):

| Position | Value | mask | Meaning |
|---|---|---|---|
| `[0]`–`[7]` | `2.31`–`0.83` | 0 | Loss for predicting the prefix portion (not included in the final loss) |
| `[8]` | `1.05` | 1 | Predict response 1st token `the` |
| `[9]` | `0.72` | 1 | Predict response 2nd token `cat` |
| `[10]` | `0.41` | 1 | Predict the stop signal `<pad>` |
| `[11]` | `1.98` | 0 | What comes after `<pad>` is also `<pad>` (ignored since mask=0) |

(The specific numbers change as training progresses. This is just for the picture.)

**After applying the mask → loss**:

```python
per_tok * mask = [0, 0, 0, 0, 0, 0, 0, 0, 1.05, 0.72, 0.41, 0]

(per_tok * mask).sum() = 1.05 + 0.72 + 0.41 = 2.18
mask.sum()             = 3
loss                   = 2.18 / 3 ≈ 0.73
```

- Positions `[0]`–`[7]` and `[11]` have mask=0, so they become zero and don't contribute to the final loss
- Only the 3 values at positions `[8]`, `[9]`, `[10]` (response `the`, `cat`, stop `<pad>`) remain, and their average is the loss

In other words, the final loss is **the average of the loss over only the 3 response tokens (`the`, `cat`, `<pad>`)**. The portion where the model failed to predict the prefix (`### Instruction: ...`) is completely ignored, so the model doesn't "practice predicting the prefix continuation" — it just practices "**generating the response**." This is the heart of instruction tuning.

---

## 5.6 Responding — Generating the Continuation of `### Response:`

After training, we **give the model an instruction in Alpaca format** and have it generate a response.

```python
def respond(model, instruction, input_text, vocab, id2word, max_tokens=8):
    prompt = format_alpaca(instruction, input_text, "")   # Up through "### Response:"
    prefix_len = len(tokenize(prompt, vocab))

    tokens = tokenize(prompt, vocab)
    with torch.no_grad():
        for _ in range(max_tokens):
            context = tokens[-SEQ_LEN:]
            x = torch.tensor([context])
            logits = model.forward(x)
            next_id = torch.argmax(logits[0, -1, :]).item()
            if next_id == vocab["<pad>"]:
                break
            tokens.append(next_id)

    return " ".join(id2word[t] for t in tokens[prefix_len:])
```

The mechanism is almost the same as `generate` in Chapter 4. The differences are:

- The input prompt uses **the Alpaca format**
- Only the response portion is sliced out and returned (`tokens[prefix_len:]`)

#### Example of arguments and return values

Let's pass the same instruction as the first training example to an instruction-tuned model.

```python
response_text = respond(
    model=model,                      # TinyTransformer that has been through Stage 1 & 2
    instruction="who sat on the mat",
    input_text="",
    vocab=vocab,                      # vocab_size = 15
    id2word=id2word,
    max_tokens=8,
)

# Internal flow:
# 1) prompt = "### Instruction: who sat on the mat ### Response:"  (9 tokens)
#    -> tokens = [10, 11, 14, 3, 4, 1, 5, 10, 13],  prefix_len = 9
# 2) Generate one token at a time with greedy decoding:
#       iter 1: argmax -> id=1 ("the")        append to tokens
#       iter 2: argmax -> id=2 ("cat")        append to tokens
#       iter 3: argmax -> id=0 (<pad>)        break out
# 3) tokens[prefix_len:] = [1, 2]
# 4) Decode with id2word and join with spaces

response_text == "the cat"
```

Because training taught it that `the cat <pad><pad>` follows immediately after `### Response:`, at inference time the model also emits `the cat` and then picks `<pad>`, breaking out of the loop. The final return value is a string of **just the response portion**: `"the cat"`. The prompt portion (`### Instruction: ...`) is excluded by the `tokens[prefix_len:]` slice.

---

## 5.7 Running It

Running `tiny_llm_instruct.py` executes 3 stages in sequence:

```bash
uv run --with torch tiny_llm_instruct.py
```

```python
model = TinyTransformer(len(vocab))    # ← The model is created only here
train(model, inputs, targets)           # Stage 1: pretraining (updates model's weights)
train_instruct(model, examples, vocab)  # Stage 2: fine-tune the same model (inheriting weights)
respond(model, ...)                     # Stage 3: generate responses with the same model
```

1. **Stage 1**: Make it learn language with ordinary pretraining
2. **Stage 2**: Instruction-tune **the same `model` object** with 4 instruction examples (see the important point in 5.1)
3. **Stage 3**: Generate a response for each instruction with **the same `model` object**

Expected output (the numbers vary slightly across runs):

```
vocab size: 15

--- Stage 1: Pretraining ---
epoch   20  loss=...
...
epoch  200  loss=...

--- Stage 2: Instruction tuning ---
epoch   30  loss=...
...
epoch  300  loss=...

--- Stage 3: Responding ---
### Instruction: who sat on the mat
### Response: the cat

### Instruction: who saw the dog
### Response: the cat

### Instruction: who sat on the log
### Response: the dog
```

Training only 4 examples for 300 epochs leaves the model essentially memorizing them.
The output may at first glance look like it's "answering the instructions," but this is nothing more than
**the result of pretraining + memorizing 4 examples**.
Pass an instruction that isn't in the training set (e.g. `who saw the mat`) and you'll get nonsense, or at best a parroted response from the most similar training example — there's no generalization.
With a 10-word vocabulary and 4 training examples there is no way to generalize, so this is the expected result.

### Even if the result is memorization, the procedure itself is the star of the chapter

What we wanted to show in this chapter is **"the procedure itself," not "the quality of the result."** The following 4 steps are the heart of instruction tuning.

1. **Shape the data into the Alpaca format** (5.2, 5.3)
2. **Build a mask so that the loss only applies to response tokens** (5.4)
3. **Fine-tune with the same Cross-Entropy loss** (5.5)
4. **At inference, give the prompt up through `### Instruction: ... ### Response:` and generate the continuation** (5.6)

On top of the language patterns memorized in pretraining ("the cat sat on the mat" and so on), the essence of instruction tuning is to teach a meta-level structure:

> **"After `### Response:`, what comes next is the answer to the instruction"**

---

## 5.8 Going Further

Beyond the 4 steps in this chapter, the rest is mainly scaling up and adding extra stages.

| | tiny-LLM (this chapter) | When scaled up |
|---|---|---|
| Dataset | 4 examples | Hundreds of thousands to tens of millions |
| Response masking | Yes | Same |
| Loss function | Cross-Entropy only at response positions | Same |
| Parameter update | All parameters | LoRA / PEFT updating only a subset is mainstream |
| Subsequent stages | None | RLHF / DPO to reflect human preferences |
| Template | Alpaca | Alpaca, ChatML, model-specific formats |

### RLHF / DPO

Instruction tuning teaches the "correct response" directly, but
**"what makes a good response"** sometimes requires more subtle judgment.
That's where stages like RLHF (Reinforcement Learning from Human Feedback) and
DPO (Direct Preference Optimization) come in, teaching the model **"responses that humans prefer."**
We won't go into detail since it goes beyond the scope of this chapter,
but the three-stage structure `Pretrain → Instruction Tune → RLHF/DPO` is worth knowing.

---

## Summary

```
[Pretrained model]
        ↓ Alpaca-format data
        ↓ Train with Cross-Entropy + Response masking
[Instruction-tuned model]
```

To sum up the key points:

- **Don't change the model** (same Transformer, same parameters)
- **Don't change the loss** (same Cross-Entropy)
- **What changes is just the shape of the data and the positions where the loss is computed**
- Data format: `### Instruction: ... ### Response: ...`
- Loss positions: only the response tokens

The true identity of the "follows instructions" behavior is:

> **"As the continuation of `### Response:` (or an equivalent marker), predict appropriate words"**

That's all there is to it. We assembled that whole structure with our own hands in the miniature of this chapter, so we've now firmly understood the essence of the Transformer. The tiny-LLM journey pauses here.
