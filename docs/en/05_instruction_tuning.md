# Chapter 5: Instruction Tuning — How to Build an LLM That Follows Instructions

![Instruction Tuning](../images/chapter-05-instruction-tuning.png)

Up through Chapter 4, the Transformer was completed as a model that **"predicts the next word"**.
It can produce output that looks like it has memorized the corpus by rote,
but it does not yet behave like ChatGPT, "answering the user's questions".

In this chapter we look at the mechanism for shaping the model so that it
**"follows instructions"**, by adding one more stage of training.
We use the format standardized by Stanford's Alpaca project.

> The code for this chapter is in a new file, `tiny_llm_instruct.py`.
> The `tiny_llm.py` from Chapter 4 and earlier is imported and reused as-is.

---

## 5.1 Pretraining vs Instruction Tuning

Modern LLMs are usually built with **two stages of training**.

**Stage 1: Pretraining**

- Learn "next-word prediction" on a large natural-language corpus (the Web, books, papers...)
- Exactly the approach we saw through Chapter 3
- Here the model learns **"language"**
- Once trained, it becomes a foundation usable for all kinds of tasks

**Stage 2: Instruction Tuning**

- **Additionally train** the pretrained model in the form of "instruction → response"
- Here the model learns to **"follow instructions"**
- It becomes able to respond to instructions like "translate this", "summarize this", "explain XX"

```
[untrained model]
        ↓ Pretraining (a large corpus)
[pretrained model: has learned language]
        ↓ Instruction Tuning (instruction × response pairs)
[instruction-tuned model: can follow instructions]
```

What is noteworthy as a mechanism is:

> **We use the same model and the same Cross-Entropy Loss as through Chapter 4.**
> All that changes is **the shape of the training data** and **at which positions we compute the loss**.

In other words, everything can be done as an application of the basic mechanism of "next-word prediction".

> ### ⚠️ The Most Important Point — the "pretrained model" is **literally the same `model` object**
>
> In the second stage of the diagram above, "Instruction Tuning", **we do not create a new model**.
> We hand the `model` variable trained in Stage 1 **as-is** to Stage 2 and continue training from there.
>
> ```python
> model = TinyTransformer(len(vocab))   # ← this is the "only time" we create the model
> train(model, ...)                      # Stage 1: update the weights by pretraining
> train_instruct(model, ...)             # Stage 2: fine-tune the same model (inheriting the weights!)
> ```
>
> If you were to re-run `model = TinyTransformer(...)` in Stage 2, **all** the language knowledge learned in Stage 1 would be **erased**. With only 4 instruction examples you cannot even learn English word order, so the generated output would be complete nonsense.
>
> **This is exactly what the word fine-tune (= "additionally adjusting" existing weights) means.** Not throwing away the Stage 1 weights but painting over them — that is the heart of instruction tuning.

---

## 5.2 The Alpaca Format

To teach a model "instruction → response", the standard is to use **a template with a fixed format**.
What Stanford's Alpaca project proposed is the following format.

```
### Instruction: <instruction>
### Input: <context or input (as needed)>
### Response: <expected response>
```

For tasks that do not need an Input, the Input section is omitted.

**Example with Input:**

```
### Instruction: Give me the subject
### Input: The cat sat on the mat.
### Response: The cat
```

**Example without Input:**

```
### Instruction: Compute 7 * 8
### Response: 56
```

### Why Do We Need a "Format"?

With fixed-format delimiters, the model finds it easier to learn
**"the input ends here / the response starts from here"**.

- During training: the model learns that the "expected response" comes **immediately after** `### Response:`
- At inference time: if you give it up to `### Instruction: <question> ### Response:`, it generates the continuation

In other words, the marker `### Response:` functions as **the cue that the response begins**.

---

## 5.3 Assembling a Toy Dataset

This project's vocabulary originally has only 10 words.

```python
{"<pad>": 0, "the": 1, "cat": 2, "sat": 3, "on": 4,
 "mat": 5, ".": 6, "dog": 7, "log": 8, "saw": 9}
```

To this we add the Alpaca-format markers and words for the task.

| Added token | Role |
|---|---|
| `###` | the format delimiter symbol |
| `Instruction:` | the start marker of the instruction section |
| `Input:` | the start marker of the input section (unused in this toy example) |
| `Response:` | the start marker of the response section |
| `who` | the question word for "who ...?" |

This brings the vocabulary to a total of **15**.

> **How to read the numbers:** As with the "How to read the numbers" table in §1.2 of Chapter 1, this project is designed so that the major numbers do not overlap with each other.
> In Chapters 1–4 we had `vocab_size = 10`, but in Chapter 5 we add the 4 Alpaca markers plus `who`, expanding it to **`vocab_size = 15`**.
> **The value 15 does not collide with any other parameter (2, 4, 10, 12, 16, 28, 64, 128)**, so from Chapter 5 onward, whenever 15 shows up in a tensor shape you can immediately tell "that's the vocabulary size".
> For example, seeing `logits.shape = (4, 12, 15)` you can read it as "4 samples × 12 tokens × scores over a 15-word vocabulary".

> **How the tokenizer works (a recap)**
>
> As we saw in Chapter 1, this project's tokenizer **only splits on whitespace**.
> So `### Instruction:` is split into **2 tokens**, `###` and `Instruction:`.
> (With a subword tokenizer such as BPE, both `###` and `Instruction` would be split into even finer units, but the flow of processing is the same.)

### The Toy Instruction Dataset

We use a toy dataset with only 4 examples.

```python
examples = [
    ("who sat on the mat", "", "the cat"),
    ("who sat on the log", "", "the dog"),
    ("who saw the dog",    "", "the cat"),
    ("who saw the cat",    "", "the dog"),
]
```

Each tuple is a 3-tuple of `(instruction, input, response)`.
This time we made them all simple examples that do not use Input.

Expanding these into Alpaca-format strings:

```
### Instruction: who sat on the mat ### Response: the cat
### Instruction: who sat on the log ### Response: the dog
### Instruction: who saw the dog ### Response: the cat
### Instruction: who saw the cat ### Response: the dog
```

The `format_alpaca` function does this formatting:

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

## 5.4 Response Masking — Concentrating the Loss on "the Response Only"

This is the **most important point** of this chapter.

If you simply compute the Cross-Entropy Loss at every position, the model ends up learning
**"to predict the instruction text too"**.
This is wasteful, and it is also harmful.

- It squanders learning capacity on "the language patterns of the instruction part"
- After training, the model may end up behaving as if it "generates instruction text"

What we want is:

> **"When an instruction comes in, generate a response as its continuation"**

That's all. So we compute the loss **only at the positions of the response tokens**.
The remaining positions are **zeroed out by a mask** and contribute nothing to the gradient computation.

### Understanding It with a Diagram

Let's look at how the data tensors for one example line up at each position.

- **`inp`**: the input token sequence for the model. All tokens the model "sees" during the forward pass (position 0 through seq_len-1). The variable name is short for input, and it is a different thing from Alpaca's `### Input:` slot (`input_text`)
- **`tgt`**: the correct answer we want predicted at each position. `inp` shifted by one (what is predicted at position `i` is `inp[i+1]`)
- **`mask`**: a 0/1 per position for whether that position's loss is included in the final loss (applied to `tgt`)

![Diagram of response masking. Of the 12 positions, the mask is 1 at only 3 places — the response tokens "the" and "cat" plus the pad that serves as the stop cue — and only the loss at those positions is used for learning. The instruction text and markers are seen as input but are not made prediction targets.](../images/response-mask-en.svg)

Key points:

- Whatever falls short of 12 tokens after lining up the instruction plus response is **filled with `<pad>` to even out the length**.
  The `<pad>` at the end of the figure is that slack. Since the length of the instruction differs per example,
  the number of pads also differs per example (conversely, examples that are too long get truncated at the end).
  This way every example becomes the same 12 tokens and they can all be stacked into a single tensor.
- Looking at the `tgt` side, positions 8 and 9 are the **response tokens** (`the`, `cat`). → mask=1 (compute the loss)
- The other `tgt` positions — the instruction text, the `### Response:` marker, and the second and later `<pad>`s — do not get a loss. → mask=0
- An important separation here: **the instruction and the `### Response:` marker themselves do exist in `inp`, and the model does see them (conditioning)**.
  We just do not make them "prediction targets" (mask=0).
- The mask **begins immediately after `### Response:` and ends with the one `<pad>` following the response**.
  "The body of the response (`the` `cat`) + the stop cue (`<pad>`)" thus forms one contiguous span.
  We include that final `<pad>` **in order to teach the model to stop generating there**.
  Without it, the model cannot stop after `cat` and would keep generating up to `max_tokens` at inference time.
- Where this span falls among the positions **changes from example to example**. The figure above is the case where the instruction is 5 words;
  for a shorter instruction it shifts earlier by that amount. That is why we don't hard-code the position but
  compute it every time from the number of tokens up to `### Response:` (we'll see this in the code in §5.5).

In short, the core of instruction tuning is **handling separately what we show the model (`inp`) and what we make it learn (the part of `tgt` selected by `mask`)**.

### Comparing with the Stage 1 Data

New names `inp` / `tgt` / `mask` have appeared so far, but
**the way training is run is exactly the same as in Stage 1**.
We bundle several sets of 12 words into one batch, push the whole thing through the model,
and repeat forward → loss → backward → update.

What differs is only 2 things: "how the sets are made" and "which positions get scored".

| | Stage 1 (pretraining) | Stage 2 (instruction tuning) |
|---|---|---|
| How sets are made | slice the 40-token corpus shifting one word at a time | format one instruction/response pair into the Alpaca format as one set |
| Length of one set | 12 tokens | 12 tokens (the same `SEQ_LEN`) |
| Number of sets | 28 | 4 (because there are 4 instruction examples) |
| Tensors | 2 of them, `inputs` and `targets`, each (28, 12) | 3 of them, `inp`, `tgt`, and `mask`, each (4, 12) |
| Correct answer | the input shifted by one | the same (shifted by one) |
| **Positions where the loss is computed** | **every position** (28 sets × 12 tokens) | **only the positions where `mask` is 1** (out of 4 sets × 12 tokens, the response + the stop cue) |
| Number of epochs | 200 | 300 |
| Model | from random initial values | **inherits the Stage 1 weights** |

The fact that `tgt` is "the input shifted by one" is the same as `targets` in Stage 1,
and it's the same idea as the sliding window in Chapter 1.
All that's been added is `mask`, and it changes "score at all 28 × 12 positions"
into "score only at the 4 × 3 response positions".

Put differently, Stage 2 is **not a new training method, but the same training with the scored locations narrowed down**.

The `mask` row above becomes exactly `torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0])`.
Since the loss can be computed one per position, multiplying that by this mask
makes only the positions with 1 contribute to the final loss.

---

![Alice and Bob teaching each other origami at a park table](../images/chapter-05-break.png)

## 5.5 The Training Code

There are 2 things to do in this section.

1. **Build the Alpaca-format training data** — convert the instruction/response pairs into the
   triple of `inp` / `tgt` / `mask` seen in §5.4, and stack all 4 examples into `(4, 12)` tensors
2. **Train** — add the processing that applies `mask` to the training loop from Chapter 3

Three functions appear, with the following roles.

| Function | Role |
|---|---|
| `build_example` | convert **one** instruction/response pair into `inp` / `tgt` / `mask` |
| `make_instruction_batch` | gather those for all the examples and stack them into 3 tensors |
| `train_instruct` | run the training with the stacked tensors (narrowing down the scored locations with `mask`) |

### `build_example` — Building (input, target, mask) from One Example

Arguments:

- `instruction`, `input_text`, `response`: the 3 Alpaca elements (e.g. `"who sat on the mat"`, `""`, `"the cat"`)
- `vocab`: the word → ID mapping built in Chapter 1
- `seq_len`: the length of the output tensors. Pass the same value as **the model's context window size** (in this project, `SEQ_LEN = 12`, defined in `tiny_llm.py`). Its role is to make every example the same length, padding short examples with `<pad>` and truncating the end of examples that are too long

Return values: `inp`, `tgt`, `mask` (each a list of length `seq_len`)

```python
def build_example(instruction, input_text, response, vocab, seq_len):
    pad = vocab["<pad>"]

    # how many tokens the prefix up to "### Response:" has
    prefix_text = format_alpaca(instruction, input_text, "")
    P = len(tokenize(prefix_text, vocab))

    # the full text ("### Instruction: ... ### Response: <response>")
    full_text = format_alpaca(instruction, input_text, response)
    full_ids = tokenize(full_text, vocab)
    F_len = len(full_ids)

    # even out the length to seq_len + 1 (the one extra is for the shift)
    if F_len > seq_len + 1:
        full_ids = full_ids[:seq_len + 1]
        F_len = seq_len + 1
    full_ids = full_ids + [pad] * (seq_len + 1 - len(full_ids))

    inp = full_ids[:seq_len]         # input (T)
    tgt = full_ids[1:seq_len + 1]    # target (T) — shifted by one

    # mask: 1 at the response tokens + the one <pad> right after the response
    # the response tokens are at index [P, F_len - 1] of full_ids,
    # and we also include the <pad> right after so the model learns it as a "stop cue"
    # in target indexing this is [P - 1, F_len - 1]
    mask = [1 if (P - 1) <= i <= (F_len - 1) else 0 for i in range(seq_len)]

    return inp, tgt, mask
```

#### An Example of the Arguments and Return Values

Calling it with the first example `("who sat on the mat", "", "the cat")` gives this.

```python
inp, tgt, mask = build_example(
    instruction="who sat on the mat",
    input_text="",
    response="the cat",
    vocab=vocab,      # vocab_size = 15 ( ### → 10, Instruction: → 11, who → 14, etc.)
    seq_len=12,
)

# return values (all 3 are lists of length 12):
inp  == [10, 11, 14,  3,  4,  1,  5, 10, 13,  1,  2,  0]
tgt  == [11, 14,  3,  4,  1,  5, 10, 13,  1,  2,  0,  0]
mask == [ 0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  0]

# converted back to words:
# inp : ###  Instruction:  who  sat  on  the  mat  ###  Response:  the  cat   <pad>
# tgt : Instruction:  who  sat  on  the  mat  ###  Response:  the  cat  <pad>  <pad>
```

This is the same content as the figure in §5.4.

- the prefix `### Instruction: who sat on the mat ### Response:` is 9 tokens → `P = 9`
- the whole thing `### Instruction: ... ### Response: the cat` is 11 tokens → `F_len = 11`
- mask: `(P - 1) <= i <= (F_len - 1)`, that is `8 <= i <= 10` → positions 8, 9, 10 are 1
- `tgt[8] = "the"` (response token 1), `tgt[9] = "cat"` (response token 2), `tgt[10] = <pad>` (the stop cue)
  ← only these 3 positions contribute to the loss

#### A Note: Why Go Through `format_alpaca`?

We take the 3 elements passed in as a tuple, go to the trouble of formatting them into a string, and then turn them back with `tokenize`.
If efficiency were the only concern, building the token sequence directly would be faster, but in real instruction tuning
you load an Alpaca dataset (e.g. a JSON with 52,000 examples) **as strings** and do the same processing.
It is a detour taken in order to match the real flow.

### `train_instruct` — The Masked Training Loop

```python
def train_instruct(model, examples, vocab, epochs=300, lr=LR):
    inputs, targets, mask = make_instruction_batch(examples, vocab)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        logits = model.forward(inputs)                   # (B, T, V)

        # compute the per-token cross-entropy at every position (reduction='none')
        per_tok = F.cross_entropy(
            logits.view(-1, model.vocab_size),
            targets.view(-1),
            reduction="none",
        ).view(targets.shape)                            # (B, T)

        # apply the mask so that only the response positions contribute to the loss
        loss = (per_tok * mask).sum() / mask.sum().clamp(min=1.0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

This is almost identical to the `train` function from Chapter 3. The difference is only the 2 lines that **apply the mask to the per-token loss**.

>
> **Python Tips: `F.cross_entropy(..., reduction="none")`**
>
> By default, `cross_entropy` returns a scalar that is the **average** of the loss over all positions.
> If you specify `reduction="none"`, it returns the loss at each position as a tensor.
> Multiplying that by a mask lets you keep only the positions you want.

>
> **Python Tips: `mask.sum().clamp(min=1.0)`**
>
> The sum of the mask is "the total number of positions where the loss is computed".
> To **prevent division by zero** in the unlikely event that everything is 0,
> `.clamp(min=1.0)` fixes the lower bound at 1.0.

#### A Concrete Example: How the Loss Becomes a Single Number

Let's follow the numbers with the same single example as the figure in §5.4 (`who sat on the mat` → `the cat`).
First, one loss comes out at each of the 12 positions. Each value is
$-\log(\text{probability})$ for the correct answer at that position; the smaller it is, the better the model is getting it right.

```
position  0     1     2     3     4     5     6     7     8     9     10    11
per_tok   2.31  1.45  0.92  0.31  0.08  0.55  0.12  0.83  1.05  0.72  0.41  1.98
mask      0     0     0     0     0     0     0     0     1     1     1     0
```

We multiply by `mask` and average what remains.

```python
(per_tok * mask).sum() = 1.05 + 0.72 + 0.41 = 2.18
mask.sum()             = 3
loss                   = 2.18 / 3 ≈ 0.73
```

The part where it failed to get the prefix right (`2.31` and so on) is wiped out by the multiplication,
and only the average over the 3 response tokens becomes the final loss.
The model does no "practice at guessing the continuation of the prefix" and single-mindedly does
only "**practice at generating the response**". This is the heart of instruction tuning.

(The numbers change with the progress of training. They are only an illustration.)

---

## 5.6 Responding — Generating the Continuation of `### Response:`

After training, we **give the model an instruction in Alpaca format** and have it generate a response.

```python
def respond(model, instruction, input_text, vocab, id2word, max_tokens=8):
    prompt = format_alpaca(instruction, input_text, "")   # up to "### Response:"
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

- the input prompt uses the **Alpaca format**
- only the response is cut out and returned (`tokens[prefix_len:]`)

#### An Example of the Arguments and Return Values

Let's hand the instruction-tuned model the same instruction as the first training example.

```python
response_text = respond(
    model=model,                      # the TinyTransformer that went through Stage 1 & 2
    instruction="who sat on the mat",
    input_text="",
    vocab=vocab,                      # vocab_size = 15
    id2word=id2word,
    max_tokens=8,
)

# the internal flow:
# 1) prompt = "### Instruction: who sat on the mat ### Response:"  (9 tokens)
#    -> tokens = [10, 11, 14, 3, 4, 1, 5, 10, 13],  prefix_len = 9
# 2) generate one token at a time with Greedy:
#       iter 1: argmax -> id=1 ("the")        append to tokens
#       iter 2: argmax -> id=2 ("cat")        append to tokens
#       iter 3: argmax -> id=0 (<pad>)        cut off with break
# 3) tokens[prefix_len:] = [1, 2]
# 4) decode with id2word and join with spaces

response_text == "the cat"
```

Because during training it learned that `the cat <pad><pad>` comes right after `### Response:`, at inference time too the model emits `the cat` and then selects `<pad>`, and the loop breaks. The final return value is **a string containing only the response part**, `"the cat"`. The prompt part (`### Instruction: ...`) is excluded by the slice `tokens[prefix_len:]`.

---

## 5.7 Let's Run It

Running `tiny_llm_instruct.py` runs the 3 stages in order:

```bash
uv run --with torch tiny_llm_instruct.py
```

```python
model = TinyTransformer(len(vocab))    # ← this is the only place we create the model
train(model, inputs, targets)           # Stage 1: pretraining (updates the model's weights)
train_instruct(model, examples, vocab)  # Stage 2: fine-tune the same model (inheriting the weights)
respond(model, ...)                     # Stage 3: generate responses with the same model
```

1. **Stage 1**: make it learn language with ordinary pretraining
2. **Stage 2**: instruction-tune **the same `model` object** on the 4 instruction examples (see the important point in §5.1)
3. **Stage 3**: generate a response to each instruction with **the same `model` object**

The expected output (the numbers vary slightly from run to run):

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

Train on just 4 examples for 300 epochs and the model pretty much memorizes them by rote.
At first glance the output looks like it is "answering the instructions", but this is
nothing more than the result of **pretraining + rote memorization of 4 examples**.
Hand it an instruction not included in the training (e.g. `who saw the mat`) and it will either produce nonsense or just parrot back the response of the most similar training example; it does not generalize.
With a vocabulary of 10 and 4 training examples there is no way to generalize, so this is only to be expected.

### Even If the Result Is Rote Memorization, the Procedure Itself Is the Star of This Chapter

What we wanted to show in this chapter is **not "how good the result is" but "the procedure itself"**. The following 4 steps are the heart of instruction tuning.

1. **Arrange the data into the Alpaca format** (§5.2, §5.3)
2. **Build a mask so that the loss applies only to the response tokens** (§5.4)
3. **Fine-tune with the same Cross-Entropy loss** (§5.5)
4. **At inference time, give `### Instruction: ... ### Response:` as the prompt and generate the continuation** (§5.6)

On top of the language patterns learned in pretraining ("the cat sat on the mat" and so on),

> **"after `### Response:` comes the answer to the instruction"**

Making it learn this meta-level structure is the essence of instruction tuning.

---

## 5.8 Going Further

Beyond the 4 steps of this chapter, what follows consists mainly of scaling up and additional stages.

| | tiny-LLM (this chapter) | When scaled up |
|---|---|---|
| Dataset | 4 examples | hundreds of thousands to tens of millions of examples |
| Response masking | present | the same |
| Loss function | Cross-Entropy at the response positions only | the same |
| Parameter updates | all parameters | mostly only a subset, via LoRA / PEFT |
| Later stages | none | RLHF / DPO to reflect human preferences |
| Template | Alpaca | Alpaca, ChatML, model-specific formats |

### RLHF / DPO

Instruction tuning teaches "the correct response" directly, but
**"what counts as a good response"** can require a much more subtle judgment.
So stages follow in which **"responses humans prefer"** are learned with
RLHF (Reinforcement Learning from Human Feedback) or DPO (Direct Preference Optimization).
Since they go beyond the scope of this chapter we won't go into detail, but
the three-stage structure `Pretrain → Instruction Tune → RLHF/DPO` is worth knowing about.

---

## Summary

```
[pretrained model]
        ↓ Alpaca-format data
        ↓ train with Cross-Entropy plus response masking
[instruction-tuned model]
```

To sum up the key points:

- **We don't change the model** (the same Transformer, the same parameters)
- **We don't change the loss either** (the same Cross-Entropy)
- **All we change is the shape of the data and where the loss is computed**
- Data format: `### Instruction: ... ### Response: ...`
- Loss positions: the response tokens only

The true identity of the behavior called "following instructions" is

> **"predicting an appropriate word as the continuation of `### Response:` (or an equivalent marker)"**

That's all it is. The tiny-LLM journey comes to a pause here.
