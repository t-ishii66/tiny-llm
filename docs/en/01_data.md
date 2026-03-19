# Chapter 1: Data Preparation — Turning Text into a Model-Ready Format

LLMs only work with numbers.
Converting "English sentences" into "sequences of numbers" and then creating "input-target pairs" —
this chapter walks through the entire process with concrete examples.

---

## 1.1 Corpus (Training Text)

First, we prepare the starting text.

```python
corpus = (
    "the cat sat on the mat . the dog sat on the log . "
    "the cat saw the dog . the dog saw the cat . "
    "the cat sat on the log . the dog sat on the mat ."
)
```

Just 40 words, but it's enough to learn the structure of a Transformer.

---

## 1.2 `build_vocab` — Assigning Numbers to Words

```python
def build_vocab(text):
    words = text.split()
    vocab = {"<pad>": 0}
    for w in words:
        if w not in vocab:
            vocab[w] = len(vocab)
    return vocab, {i: w for w, i in vocab.items()}
```

>
> **Python Tips: Dict comprehension `{i: w for w, i in ...}`**
>
> `{i: w for w, i in vocab.items()}` is a **dict comprehension**.
> It's a shorthand for building a dictionary with a for loop:
> ```python
> # These two produce the same result
> id2word = {}
> for w, i in vocab.items():
>     id2word[i] = w
>
> id2word = {i: w for w, i in vocab.items()}  # One-liner
> ```
> `vocab.items()` returns pairs like `("the", 1), ("cat", 2), ...`.
> `for w, i in ...` unpacks each pair into `w="the", i=1`.

### What It Does

1. Split the text by whitespace into a list of words
2. Assign numbers in order of first appearance, without duplicates (`<pad>` is reserved as 0)
3. Return a "word → number" dictionary and a "number → word" reverse dictionary

### Concrete Example

```
Argument:  "the cat sat on the mat . the dog sat on the log ..."

Return value vocab:
  {"<pad>": 0, "the": 1, "cat": 2, "sat": 3, "on": 4,
   "mat": 5, ".": 6, "dog": 7, "log": 8, "saw": 9}

Return value id2word:
  {0: "<pad>", 1: "the", 2: "cat", 3: "sat", 4: "on",
   5: "mat", 6: ".", 7: "dog", 8: "log", 9: "saw"}
```

The vocabulary size is **10**. Real LLMs have tens to hundreds of thousands, but the mechanism is the same.

> **Reading the numbers:** In this project, all key numbers are designed to have
> distinct values. By looking at the numbers that appear in tensor shapes,
> you can immediately tell what they represent:
>
> | Number | Meaning |
> |--------|---------|
> | **2** | Number of Transformer layers (N_LAYERS) |
> | **4** | Number of attention heads (N_HEADS) |
> | **10** | Vocabulary size (vocab_size) |
> | **12** | Sequence length (SEQ_LEN) |
> | **16** | Head dimension (head_dim = 64 ÷ 4) |
> | **28** | Number of training samples (40 tokens − 12) |
> | **64** | Embedding dimension (D_MODEL) |
> | **128** | FFN hidden dimension (D_FF) |
>
> For example, when you see tensor shape `(28, 12, 64)`,
> you can immediately read it as "28 samples × 12 tokens × 64-dim embeddings."

> **Difference from real models:** GPT and similar models further split words into
> subwords using BPE (Byte Pair Encoding). Here we simplify: token = word.

---

## 1.3 `tokenize` — Converting Text to a Sequence of Numbers

```python
def tokenize(text, vocab):
    return [vocab[w] for w in text.split()]
```

>
> **Python Tips: List comprehension `[... for ... in ...]`**
>
> `[vocab[w] for w in text.split()]` is a **list comprehension**.
> It's a shorthand for building a list with a for loop:
> ```python
> # These two produce the same result
> result = []
> for w in text.split():
>     result.append(vocab[w])
>
> result = [vocab[w] for w in text.split()]  # One-liner
> ```

### What It Does

Split the text by whitespace and replace each word with its number from `vocab`.

### Concrete Example

```
Arguments:  text  = "the cat sat on the mat"
            vocab = {"<pad>": 0, "the": 1, "cat": 2, "sat": 3, "on": 4, "mat": 5, ...}

Return value: [1, 2, 3, 4, 1, 5]
```

```
"the"  →  1
"cat"  →  2
"sat"  →  3
"on"   →  4
"the"  →  1   ← Same word gets the same number
"mat"  →  5
```

This list of numbers (the **token sequence**) becomes the basic unit for all subsequent processing.

---

## 1.4 `make_training_data` — Creating Input-Target Pairs

This is the most important part. LLM training is essentially the task of **"predicting the next word from the preceding sequence of words."**

```python
def make_training_data(text, vocab):
    tokens = tokenize(text, vocab)
    inputs, targets = [], []
    for i in range(len(tokens) - SEQ_LEN):
        inputs.append(tokens[i : i + SEQ_LEN])
        targets.append(tokens[i + 1 : i + SEQ_LEN + 1])
    return torch.tensor(inputs), torch.tensor(targets)
```

>
> **Python Tips: Slicing `tokens[i : i + SEQ_LEN]`**
>
> List slicing `list[start:end]` extracts from start up to (but **not including**) end:
> ```python
> tokens = [1, 2, 3, 4, 5, 6, 7, 8]
> tokens[0:3]   # → [1, 2, 3]      positions 0, 1, 2 (3 is not included)
> tokens[2:5]   # → [3, 4, 5]      positions 2, 3, 4
> tokens[1:4+1] # → [2, 3, 4, 5]   from i+1 up to (but not including) i+SEQ_LEN+1
> ```

>
> **Python Tips: `torch.tensor()` — Converting lists to tensors**
>
> `torch.tensor([[1,2,3], [4,5,6]])` converts a Python list to a PyTorch tensor.
> A tensor is a "multidimensional array" that supports GPU computation and automatic differentiation:
> ```python
> import torch
> x = torch.tensor([[1, 2], [3, 4]])
> x.shape  # → torch.Size([2, 2])  ← 2 rows × 2 columns
> ```

### What It Does

1. Convert the entire corpus into a token sequence
2. Slide a window of length `SEQ_LEN` (=12)
3. For each window, create an **input** and **target** pair

### Concrete Example (with SEQ_LEN=12)

Token sequence of the corpus (40 tokens total):

```
Position:  0   1   2   3   4   5   6   7   8   9  10  11  12  13 ...
Word:     the cat sat  on the mat  .  the dog sat  on the log  .  ...
Number:    1   2   3   4   1   5   6   1   7   3   4   1   8   6 ...
```

**How the sliding window works:**

```
i=0:
  input  = tokens[ 0:12] = [1,2,3,4,1,5,6,1,7,3,4,1]
  target = tokens[ 1:13] = [2,3,4,1,5,6,1,7,3,4,1,8]
                              ↑ The "next word" at each position of input is the target

i=1:
  input  = tokens[ 1:13] = [2,3,4,1,5,6,1,7,3,4,1,8]
  target = tokens[ 2:14] = [3,4,1,5,6,1,7,3,4,1,8,6]

i=2:
  input  = tokens[ 2:14] = [3,4,1,5,6,1,7,3,4,1,8,6]
  target = tokens[ 3:15] = [4,1,5,6,1,7,3,4,1,8,6,1]

  ...(28 samples total)
```

>
> **About Slide Width (Stride)**
>
> In this code, the window slides **one word at a time** (i=0, 1, 2, ...).
> However, the slide width can be set freely:
>
> ```
> Stride 1:  i=0, 1, 2, 3, ...  → More samples, more overlap
> Stride 4:  i=0, 4, 8, 12, ... → Fewer samples, less overlap
> Stride 12: i=0, 12, 24, ...   → No overlap (windows are adjacent)
> ```
>
> - **Stride 1** maximizes data utilization, but most of each neighboring sample overlaps
> - **Larger strides** reduce overlap and speed up computation, but produce fewer samples
> - In large-scale LLM training, the corpus is large enough that
>   using a larger stride is not a problem
>
> In this project, the corpus is small, so we use stride 1 to generate
> the maximum number of samples.

### Target is Input Shifted by One

When visualized as a diagram, the essence becomes clear:

```
input:   the  cat  sat  on   the  mat   .   the  ...
target:  cat  sat  on   the  mat   .   the  dog  ...
         ↑    ↑    ↑    ↑    ↑    ↑    ↑    ↑
         Predict the "next word" at each position
```

At position 0, "the next word after the is cat"; at position 1, "the next word after cat is sat"...
The key point is that **all positions simultaneously predict the "next word."**

### Return Value Shapes

```
inputs:  torch.Tensor, shape = (28, 12)    ← 28 samples × 12 tokens
targets: torch.Tensor, shape = (28, 12)    ← Corresponding targets
```

---

## 1.5 Data Flow — The Big Picture

```
                        build_vocab
"the cat sat on ..."  ─────────────→  vocab:   {"the": 1, "cat": 2, ...}
                                       id2word: {1: "the", 2: "cat", ...}

                        tokenize
"the cat sat on ..."  ─────────────→  [1, 2, 3, 4, 1, 5, 6, ...]

                      make_training_data
[1, 2, 3, 4, ...]    ─────────────→  inputs:  tensor (28, 12)
                                       targets: tensor (28, 12)
```

At this point, the data is ready to be passed to the model.

**In the next chapter (Chapter 2)**, we'll see how this `inputs` tensor
is processed inside the Transformer.

---

## Summary

| Function | Input | Output | Role |
|----------|-------|--------|------|
| `build_vocab(text)` | String | `vocab` (dict), `id2word` (dict) | Assign numbers to words |
| `tokenize(text, vocab)` | String, vocab | `[int, ...]` (list) | Convert text to number sequence |
| `make_training_data(text, vocab)` | String, vocab | `inputs` (28,12), `targets` (28,12) | Create training pairs |
