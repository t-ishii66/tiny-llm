# Chapter 2: Transformer — The Mechanism for Understanding Context

In the previous chapter, we created the `inputs` tensor (28, 12).
In this chapter, we follow how this sequence of numbers is processed inside
the Transformer to arrive at "next word predictions."

---

## 2.1 The Overall Flow

```
token_ids (28, 12)
    │
    ▼
┌──────────────────────┐
│ Token Embedding      │  Numbers → Vectors
│ + Positional Emb.    │  Add positional information
└──────────┬───────────┘
           │  (28, 12, 64)
           ▼
┌──────────────────────┐
│ Transformer Block ×2 │  Self-Attention + FFN
└──────────┬───────────┘
           │  (28, 12, 64)
           ▼
┌──────────────────────┐
│ Layer Norm           │
│ → Logits (linear)    │  Vectors → Vocabulary scores
└──────────┬───────────┘
           │  (28, 12, 10)
           ▼
     Scores for 10 words at each position
```

---

## 2.2 Embedding — Turning Numbers into Vectors

### Why Vectors Are Needed

Word number `3` (= "sat") is just an integer with no "meaning."
By converting it to a **64-dimensional vector**, we can numerically
represent relationships between words.

### Code

```python
# From TinyTransformer.__init__
self.tok_emb = param(vocab_size, D_MODEL)   # (10, 64)
self.pos_emb = param(SEQ_LEN, D_MODEL)      # (12, 64)
```

```python
# From TinyTransformer.forward
x = self.tok_emb[token_ids] + self.pos_emb[:T]
```

>
> **Python Tips: Tensor indexing with `tok_emb[token_ids]`**
>
> When you pass a list or tensor of integers to a tensor, it retrieves those rows all at once
> (this is called **fancy indexing**):
> ```python
> table = torch.tensor([[0.1, 0.2],    # Row 0
>                        [0.3, 0.4],    # Row 1
>                        [0.5, 0.6]])   # Row 2
>
> table[[2, 0, 1]]
> # → tensor([[0.5, 0.6],   ← Row 2
> #            [0.1, 0.2],   ← Row 0
> #            [0.3, 0.4]])  ← Row 1
> ```
> `tok_emb[token_ids]` retrieves all the embedding vectors corresponding to each token number at once.

>
> **Python Tips: Slicing with `pos_emb[:T]`**
>
> `[:T]` is a slice that takes "the first T items":
> ```python
> x = torch.tensor([10, 20, 30, 40, 50])
> x[:3]   # → tensor([10, 20, 30])   First 3 items
> x[:5]   # → tensor([10, 20, 30, 40, 50])   All items
> ```
> `pos_emb` has 12 rows, but if the input is 4 tokens, `pos_emb[:4]` uses only the first 4 rows.

### Step-by-Step Explanation

**Step 1: Token Embedding**

`tok_emb` is a table of shape `(10, 64)`, where each row is one word's vector.

```
tok_emb = [
  Row 0: [0.01, -0.03, 0.02, ...],   ← "<pad>" vector (64 dimensions)
  Row 1: [0.05,  0.01, -0.02, ...],  ← "the" vector
  Row 2: [-0.01, 0.04,  0.03, ...],  ← "cat" vector
  ...
  Row 9: [0.02, -0.01,  0.05, ...],  ← "saw" vector
]
```

When `token_ids = [1, 2, 3, 4, ...]`, `tok_emb[token_ids]` simply
retrieves rows 1, 2, 3, 4, and so on.

```
token_ids:     [1,     2,     3,     4,    ...]
                ↓      ↓      ↓      ↓
From tok_emb:  [the],  [cat], [sat], [on], ...   ← Each a 64-dim vector
```

Resulting shape: `(28, 12)` → `(28, 12, 64)`

**Step 2: Positional Embedding**

Even for the same word "the," its role differs at the beginning versus the middle of a sentence.
`pos_emb` is a table that holds a vector for each position:
"position 0 has this vector, position 1 has this vector..."

```
pos_emb = [
  Position 0: [0.02, -0.01, 0.01, ...],   ← 64 dimensions
  Position 1: [0.01,  0.03, -0.02, ...],
  Position 2: [-0.03, 0.02,  0.01, ...],
  ...
  Position 11: [0.01, 0.01, -0.04, ...],
]
```

These are simply added together:

```
x = tok_emb[token_ids] + pos_emb[:T]

Position 0: "the" vector + position 0 vector = vector for "the at position 0"
Position 1: "cat" vector + position 1 vector = vector for "cat at position 1"
...
```

> **Key point:** Both `tok_emb` and `pos_emb` are **learnable parameters**.
> They start as random values but are updated to meaningful values through training.

---

## 2.3 Self-Attention — The Heart of the Transformer

Self-Attention computes "which word in the sentence should attend to which other word."
This is the most important mechanism of the Transformer.

### An Honest Note to Begin With

When learning Self-Attention for the first time, many people wonder
**"Why do we do this?"** The reasons for splitting into Query, Key, and Value,
for using dot products to compute scores, for dividing by $\sqrt{d_k}$ —
none of these are intuitively obvious.

In fact, the Q/K/V mechanism has **both theoretical foundations and empirical improvements**.

**Things with theoretical justification:**
- **Separation of Q/K/V**: Based on the concept of information retrieval (IR).
  In search engines, the "search query (Q)" and "document titles (K)" are different things,
  and you retrieve the "document body (V)" that matches.
  Since the same word needs different representations when "searching" versus "being searched for"
  — using separate projections is a rational design
- **Division by $\sqrt{d_k}$**: When dimensions are large, the variance of dot products increases,
  causing softmax outputs to become extreme. This normalization prevents that
  and can be rigorously derived statistically

**Things that worked empirically:**
- **Multi-Head Attention**: Multiple small Attentions performed better
  than one large Attention
- **Attention alone is sufficient**: The 2017 "Attention Is All You Need" paper
  showed that completely removing the previously essential RNN (recurrent neural network)
  actually improved performance

In other words, the Q/K/V framework itself has **"reasons why it should be done this way,"** while
the specific ways of combining and configuring it include **parts discovered through trial and error**.

So, when reading below and thinking "Why?":

> **For things with clear reasons → "I see, that's why it's done this way"**
> **For things without clear reasons → "This method gave the best performance"**

Distinguishing between these two makes learning easier.

### An Intuitive Example

Consider the sentence "the cat sat on the mat."
To understand the meaning of "sat," you need to know
"**who** sat (cat)" and "**where** they sat (mat)."

Self-Attention automatically learns the "degree of attention" from "sat" to "cat" and "mat."

### Query, Key, Value — Three Roles

In Self-Attention, each word has three faces:

| Role | Meaning | Analogy |
|------|---------|---------|
| **Query (Q)** | "What am I looking for?" | The person asking a question |
| **Key (K)** | "What information do I hold?" | A name tag / label |
| **Value (V)** | "The actual information I hold" | The content of the answer |

**As a search analogy:**
- Q is the "search term"
- K is the "title of each page"
- V is the "body of each page"

The higher the match (dot product) between Q and K, the more of that V gets incorporated.

### How Q, K, V Are Born from x

We said "Q, K, V are separate things," but
the input is **just a single x**. This is the first point of confusion.

**What is x?** It's the embedding vector for each word, created in the previous step.

```
"the" → x₀ = [0.05, 0.01, -0.02, ...]   ← 64-dimensional vector
"cat" → x₁ = [-0.01, 0.04, 0.03, ...]   ← 64-dimensional vector
"sat" → x₂ = [0.02, -0.03, 0.01, ...]   ← 64-dimensional vector
```

From this **same x**, we multiply by **three different weight matrices** to create Q, K, V:

```
x₂ ("sat" vector)    ← 64 dimensions
    │
    ├── × Wq (64×64 matrix) ──→ Q₂ = [0.12, -0.08, ...]  ← 64 dims "What am I looking for?"
    │
    ├── × Wk (64×64 matrix) ──→ K₂ = [-0.05, 0.15, ...]  ← 64 dims "What do I have?"
    │
    └── × Wv (64×64 matrix) ──→ V₂ = [0.07, 0.03, ...]   ← 64 dims "Information to pass"
```

**Why create three from the same x?**

Even the same person behaves differently when "asking questions" versus "answering."
The appearance of "sat" when it **goes looking** for other words (Q) and
when it **is being looked for** by other words (K) should be different.
The three matrices $W_q$, $W_k$, $W_v$ act as lenses that transform x
into a space suited to each role.

Expressed mathematically:

$$Q = x \cdot W_q, \quad K = x \cdot W_k, \quad V = x \cdot W_v$$

Here $x$ is (12, 64) and $W_q$ is (64, 64), so $Q$ is also (12, 64).
The shape doesn't change — the contents are transformed into "query-purpose,"
"search-purpose," and "information-purpose" respectively.

There is actually one more weight matrix: **$W_o$** (64×64).
This is not used for the Q/K/V computation, but rather to **integrate the final output** of Attention.
Its role is to combine the results split across Multi-Head, and it appears in Step 7.

In summary, Self-Attention has **four weight matrices**:

| Matrix | Shape | Role | When Used |
|--------|-------|------|-----------|
| $W_q$ | (64, 64) | Transform x → Query | Start (Step 1) |
| $W_k$ | (64, 64) | Transform x → Key | Start (Step 1) |
| $W_v$ | (64, 64) | Transform x → Value | Start (Step 1) |
| $W_o$ | (64, 64) | Output projection after head integration | End (Step 7) |

### Attention Scores — Computing "Who to Attend To"

Once Q and K are ready, we measure "degree of match" with dot products:

$$\text{score}(i, j) = \frac{Q_i \cdot K_j^T}{\sqrt{d_k}}$$

A **single element** of the attn matrix comes from the dot product of vectors $Q_i$ and $K_j$
(after Multi-Head splitting, these are **16-dimensional**; before splitting, 64-dimensional).
$Q_i$ is the Query of each word, so there are **12** ($Q_0$ through $Q_{11}$),
and likewise $K_j$ has **12** ($K_0$ through $K_{11}$).
Computing dot products for all combinations (12 × 12 = 144) yields a **12×12 attn matrix**:

```
Q₂ = [0.12, -0.08, 0.05, 0.21, ...]   ← "sat" Query (16 dims × 4 heads)
K₁ = [0.09,  0.15, 0.03, 0.18, ...]   ← "cat" Key   (16 dims × 4 heads)

Dot product for 1 head = 0.12×0.09 + (-0.08)×0.15 + 0.05×0.03 + 0.21×0.18 + ... (sum of 16 terms)
                       = 2.1 (a single scalar)
```

Performing this computation for all (i, j) combinations yields a 12×12 score matrix.
Looking at the row for "sat" (i=2):

```
Dot product of "sat"'s Q₂ with each word's K:

  Q₂ · K₀("the") = 0.3   ← Not very related
  Q₂ · K₁("cat") = 2.1   ← Strong match! ("Who sat?" → "cat!")
  Q₂ · K₂("sat") = 0.8   ← Moderate attention to itself
```

Dividing by $\sqrt{d_k}$ (= $\sqrt{16}$ = 4) is to prevent the dot product values from growing
too large when the dimension is high, which would cause softmax to produce extreme distributions
(nearly all 0s and 1s). This is a statistically derivable normalization.

### Softmax — Converting Scores to Probabilities

Softmax is a function that converts any sequence of numbers into a "probability distribution summing to 1.0":

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

$e$ is Euler's number (≈ 2.718). Each element is made positive with $e^{z_i}$,
then divided by the total to produce probabilities. Larger scores get larger probabilities.

Computing concretely with the example above:

```
Scores:  [0.3,   2.1,   0.8 ]
e^score: [1.35,  8.17,  2.23]     ← All positive values
Total:   1.35 + 8.17 + 2.23 = 11.75
Probs:   [1.35/11.75,  8.17/11.75,  2.23/11.75]
       = [0.11,        0.70,        0.19 ]   ← Sums to 1.0
```

→ "cat" (2.1), which had the highest score, gets the highest attention (0.70).

Expressed mathematically:

$$\text{attn}(i, j) = \text{softmax}_j(\text{score}(i, j))$$

The entire attn(i, j) is a matrix of size **sequence length × sequence length** (12 × 12).
That is, it's the number of words in the current sentence × the number of words:

```
         j=0    j=1    j=2    j=3          j=11
        "the"  "cat"  "sat"  "on"   ...   "the"
i=0 "the" [ 1.00   0      0      0    ...   0    ]
i=1 "cat" [ 0.35   0.65   0      0    ...   0    ]
i=2 "sat" [ 0.11   0.70   0.19   0    ...   0    ]  ← Same values as softmax example above
i=3 "on"  [ 0.05   0.10   0.60   0.25 ...   0    ]
 :                    :
i=11"the" [ 0.02   0.03   0.05   0.04 ...   0.12 ]
```

- Each row is the attention pattern for one token (i). Each row sums to 1.0
- The upper right being 0 is due to the causal mask (future tokens cannot be seen)
- There is **one of these matrices per head**, so with 4 heads there are four 12×12 matrices

### Output — Weighted Sum of Values

$$\text{out}_i = \sum_j \text{attn}(i, j) \cdot V_j$$

Finally, we blend each word's Value using the attention weights:

```
"sat" output = 0.11 × V₀("the") + 0.70 × V₁("cat") + 0.19 × V₂("sat")
```

→ A new vector for "sat" is obtained, with "cat"'s information mixed in the most.
"sat" has been updated to a **context-aware representation**.

### Code — The `self_attention` Function

```python
def self_attention(x, Wq, Wk, Wv, Wo):
    B, T, D = x.shape        # B=28, T=12, D=64
    head_dim = D // N_HEADS   # 64 // 4 = 16
```

**Step 1: Compute Q, K, V**

```python
    Q = x @ Wq   # (28, 12, 64) @ (64, 64) → (28, 12, 64)
    K = x @ Wk   # Same
    V = x @ Wv   # Same
```

>
> **Python Tips: The `@` operator (matrix multiplication)**
>
> Python's `@` is the **matrix multiplication** operator.
> It corresponds to $A \times B$ in mathematics:
> ```python
> import torch
> A = torch.tensor([[1, 2],
>                    [3, 4]])       # (2, 2)
> B = torch.tensor([[5, 6],
>                    [7, 8]])       # (2, 2)
> A @ B
> # → tensor([[19, 22],             1×5+2×7=19, 1×6+2×8=22
> #            [43, 50]])            3×5+4×7=43, 3×6+4×8=50
> ```
> `x @ Wq` performs "each word vector (64 dims) × weight matrix (64×64),"
> transforming each word into a different 64-dimensional space.

We multiply the weight matrices with each word's 64-dimensional vector to obtain Query, Key, and Value.

**Step 2: Split into Multi-Head**

```python
    Q = Q.view(B, T, N_HEADS, head_dim).transpose(1, 2)
    # (28, 12, 64) → (28, 12, 4, 16) → (28, 4, 12, 16)
```

>
> **Python Tips: `.view()` and `.transpose()` — Reshaping tensors**
>
> **`.view()`** changes the shape of a tensor. The data itself doesn't change:
> ```python
> x = torch.tensor([1, 2, 3, 4, 5, 6])   # shape: (6,)
> x.view(2, 3)    # → tensor([[1, 2, 3],
>                 #            [4, 5, 6]])   shape: (2, 3)
> x.view(3, 2)    # → tensor([[1, 2],
>                 #            [3, 4],
>                 #            [5, 6]])       shape: (3, 2)
> ```
> Here we change `(28, 12, 64)` to `(28, 12, 4, 16)`.
> Since 64 = 4×16, we decompose the last 64 dimensions into "4 heads × 16 dimensions."
>
> **`.transpose(1, 2)`** swaps the two specified axes:
> ```python
> x = torch.zeros(28, 12, 4, 16)
> x.transpose(1, 2).shape   # → (28, 4, 12, 16)
>                            #         ↑  ↑
>                            #    Axes 1 and 2 swapped
> ```
> This brings the "head" axis to the front, so each head can compute attention independently.

We split the 64 dimensions into 4 heads × 16 dimensions.

**Why split?**

A single Attention head can only produce **one softmax distribution** per token.
That means it can only express one attention pattern.

But "sat" should want to attend to multiple targets simultaneously:

```
What "sat" wants to know:
  · "Who sat?" → Want to attend to "cat"
  · "Where did they sit?" → Want to attend to "mat"
```

Trying to express both with a single softmax results in a compromised distribution.
With Multi-Head, each head can have **a different attention pattern**:

```
Head 0: "sat" → Strongly attends to "cat" (subject-verb relationship)
Head 1: "sat" → Strongly attends to "mat" (verb-location relationship)
Head 2: "sat" → Strongly attends to "on" (adjacent word)
Head 3: "sat" → Attends to "." (sentence boundary)
```

Each head computes Attention in a small 16-dimensional space.
Rather than 1 head with 64 dimensions, 4 heads with 16 dimensions each
can capture diverse relationships simultaneously — this is the essence of Multi-Head.

(What each head actually learns depends on the training data.)

**Step 3: Compute Attention Scores**

```python
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(head_dim)
    # (28, 4, 12, 16) @ (28, 4, 16, 12) → (28, 4, 12, 12)
```

>
> **Python Tips: `.transpose(-2, -1)` — Specifying axes with negative indices**
>
> In Python, negative numbers mean "count from the end."
> For a 4-dimensional tensor `(28, 4, 12, 16)`:
> ```
> Axis number:  0    1    2    3
>              28    4   12   16
>
> Negative:    -4   -3   -2   -1
> ```
> So `-2` is "second from last" = axis 2 (size 12),
> `-1` is "last" = axis 3 (size 16).
>
> `.transpose(-2, -1)` **swaps the last two axes**, so:
> ```python
> K.shape                     # (28, 4, 12, 16)
> K.transpose(-2, -1).shape   # (28, 4, 16, 12)
>                              #           ↑   ↑
>                              #       12 and 16 swapped
> ```
> Why use negative numbers? Because even if the total number of axes changes,
> writing "the last two" always works correctly.
> `transpose(2, 3)` would give the same result, but `(-2, -1)` is more general.

`scores[b][h][i][j]` = how much word i attends to word j in head h.

A concrete computation (imagining 1 head, 3 words):

```
         K₀    K₁    K₂          ← Keys (information labels)
Q₀  [ 0.8   0.1   0.1 ]    ← "the" strongly attends to itself
Q₁  [ 0.3   0.5   0.2 ]    ← "cat" most attends to itself, somewhat to "the"
Q₂  [ 0.2   0.6   0.2 ]    ← "sat" strongly attends to "cat" (who sat?)
```

**Step 4: Causal Mask**

```python
    mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
    scores = scores.masked_fill(mask, float("-inf"))
```

>
> **Python Tips: `torch.triu()` and `float("-inf")`**
>
> **`torch.triu()`** creates an upper triangular matrix.
> With `diagonal=1`, starting one above the diagonal:
> ```python
> torch.triu(torch.ones(3, 3), diagonal=1)
> # → tensor([[0, 1, 1],
> #            [0, 0, 1],
> #            [0, 0, 0]])
> ```
>
> **`float("-inf")`** is Python's "negative infinity."
> A special value smaller than any number — when passed through softmax, it becomes probability 0:
> ```python
> float("-inf") < -9999999   # → True
> ```
>
> **`.masked_fill(mask, value)`** fills positions where mask is True with value.

In language models, there is a constraint that "future words must not be seen."
A token at position i can only see tokens at positions 0 through i.

```
Before mask:              After mask (-inf hides the future):
     0    1    2           0     1      2
0 [ 0.8  0.1  0.1]   [ 0.8  -inf   -inf]
1 [ 0.3  0.5  0.2]   [ 0.3   0.5   -inf]
2 [ 0.2  0.6  0.2]   [ 0.2   0.6    0.2]
```

`-inf` becomes 0 in softmax, so information from future tokens is completely blocked.

**Step 5: Softmax**

```python
    attn_weights = F.softmax(scores, dim=-1)
```

>
> **Python Tips: `dim=-1` — "The last axis"**
>
> Many PyTorch functions take a `dim` argument to specify "which axis direction to process along."
> `dim=-1` means **the last axis** (= along each row):
> ```python
> x = torch.tensor([[1.0, 2.0, 3.0],
>                    [1.0, 1.0, 1.0]])
>
> F.softmax(x, dim=-1)   # Softmax along each row
> # → tensor([[0.09, 0.24, 0.67],   ← Each row sums to 1.0
> #            [0.33, 0.33, 0.33]])
>
> x.mean(dim=-1)          # Mean of each row
> # → tensor([2.0, 1.0])
> ```
> `dim=0` is along columns, `dim=1` is along rows, `dim=-1` is always the last axis.

Each row is converted to a probability distribution (sum = 1.0):

```
After softmax:
     0     1     2
0 [ 1.0   0.0   0.0]    ← "the" can only see itself (sum 1.0)
1 [ 0.35  0.65  0.0]    ← "cat" can see "the" and itself (sum 1.0)
2 [ 0.11  0.70  0.19]   ← "sat" can see all (sum 1.0)
```

**Step 6: Weighted Sum of Values**

```python
    out = attn_weights @ V   # (28, 4, 12, 12) @ (28, 4, 12, 16) → (28, 4, 12, 16)
```

> **Note on shapes: Ignore the batch size of 28**
>
> The **28** (batch size) appearing in tensor shapes simply means "processing 28 samples
> simultaneously." PyTorch handles the parallelism internally, so
> when understanding the algorithm, **just ignore the 28 and think about a single sample**.
> Similarly, **4** (number of heads) just means 4 heads independently performing the same computation.
>
> The core is this matrix multiplication:
> ```
> attn_weights (12, 12)  @  V (12, 16)  →  out (12, 16)
> ```

Let's look at what `attn_weights @ V` is doing concretely.

`attn_weights` is a 12×12 matrix of causal-masked attention weights.
`V` is a 12×16 matrix where each token's Value vector (16 dimensions) is lined up for all 12 tokens.

```
attn_weights (12×12)          V (12×16)
                               V₀("the") = [0.03, -0.01, 0.05, ...]  ← 16 dims
 "the" → [ 1.0   0    0  ...]  V₁("cat") = [0.07,  0.12, -0.03, ...]
 "cat" → [ 0.35  0.65 0  ...]  V₂("sat") = [0.01,  0.08,  0.04, ...]
 "sat" → [ 0.11  0.70 0.19 ...]    :
   :            :                V₁₁("the") = [...]
```

Through matrix multiplication, the output for "sat"'s row (i=2) is:

```
out₂ = 0.11 × V₀("the") + 0.70 × V₁("cat") + 0.19 × V₂("sat") + 0 + 0 + ...
                                                                    ↑ 0 due to causal mask

     = [0.11×0.03 + 0.70×0.07 + 0.19×0.01,      ← 1st element of 16-dim vector
        0.11×(-0.01) + 0.70×0.12 + 0.19×0.08,    ← 2nd element
        ...]                                       ← ...16 in total
```

→ A new 16-dimensional vector for "sat" is obtained, with "cat"'s Value mixed in the most (×0.70).
This is computed simultaneously for all tokens, yielding a (12, 16) matrix.

**Step 7: Head Concatenation and Output Projection**

```python
    out = out.transpose(1, 2).contiguous().view(B, T, D)  # → (28, 12, 64)
    out = out @ Wo                                          # Output projection
```

This line chains three operations. Let's trace them one by one.

**Step 7a: `.transpose(1, 2)` — Swap the head and token axes**

```
Current shape of out: (28, 4, 12, 16)
                       ↑   ↑   ↑   ↑
                      Batch Head Token HeadDim

transpose(1, 2) → Swap axis 1 (head) and axis 2 (token)

Resulting shape:      (28, 12, 4, 16)
                       ↑   ↑   ↑   ↑
                      Batch Token Head HeadDim
```

This swap places the "4 heads' results" for each token side by side.

**Step 7b: `.contiguous()` — Rearrange memory layout**

`.transpose()` doesn't actually move the data; it only changes the "reading order."
However, the next `.view()` requires data to be contiguous in memory.
`.contiguous()` physically rearranges the data in the new order.

The shape doesn't change (still `(28, 12, 4, 16)`). Only the internal memory layout is reorganized.

**Step 7c: `.view(B, T, D)` — Concatenate 4 heads into one**

```
(28, 12, 4, 16) → (28, 12, 64)
             ↑ ↑           ↑
          4 × 16 = 64 merged
```

For each token, the four heads' 16-dimensional vectors are simply concatenated back to 64 dimensions:

```
For token "sat":

  Head 0 output: [a₀, a₁, ..., a₁₅]     ← 16 dims (captured subject relationship)
  Head 1 output: [b₀, b₁, ..., b₁₅]     ← 16 dims (captured location relationship)
  Head 2 output: [c₀, c₁, ..., c₁₅]     ← 16 dims
  Head 3 output: [d₀, d₁, ..., d₁₅]     ← 16 dims

  After view: [a₀, ..., a₁₅, b₀, ..., b₁₅, c₀, ..., c₁₅, d₀, ..., d₁₅]
              └─── 64 dimensions ──────────────────────────────────────────┘
```

**Step 7d: `@ Wo` — Output projection**

```python
    out = out @ Wo   # (28, 12, 64) @ (64, 64) → (28, 12, 64)
```

Finally, we multiply by `Wo` (a 64×64 weight matrix) to blend the information from the 4 heads.
After concatenation alone, each head's results just sit side by side independently.
Multiplying by `Wo` integrates "the subject information found by Head 0" and
"the location information found by Head 1" into a single vector.

---

## 2.4 Layer Normalization — Stabilizing Values

```python
def layer_norm(x, g, b, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return g * (x - mean) / torch.sqrt(var + eps) + b
```

>
> **Python Tips: `keepdim=True` — Preserving dimensions**
>
> `mean(dim=-1)` collapses the last axis, reducing the number of dimensions by one.
> Adding `keepdim=True` keeps it as an axis of size 1:
> ```python
> x = torch.tensor([[1.0, 2.0, 3.0],
>                    [4.0, 5.0, 6.0]])   # shape: (2, 3)
>
> x.mean(dim=-1)                # → tensor([2., 5.])        shape: (2,)
> x.mean(dim=-1, keepdim=True)  # → tensor([[2.], [5.]])    shape: (2, 1)
> ```
> With `keepdim=True`, operations like `x - mean` use
> **broadcasting** (the mechanism that automatically aligns shapes) correctly.

Each vector is normalized to "mean 0, variance 1," then scale `g` and shift `b` are applied.

$$\text{LayerNorm}(x) = g \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + b$$

### Why It's Needed

In Deep Learning, as layers are stacked, vector values can become extremely large
(or small). Layer Norm prevents this and stabilizes training.

### Concrete Example

```
Input:      [2.0, 4.0, 6.0, 8.0]
Mean:       5.0
Variance:   5.0
Normalized: [-1.34, -0.45, 0.45, 1.34]   ← Mean 0, variance 1
```

---

## 2.5 Feed-Forward Network — Transforming Each Word Individually

```python
def feed_forward(x, W1, b1, W2, b2):
    return F.relu(x @ W1 + b1) @ W2 + b2
```

$$\text{FFN}(x) = \text{ReLU}(x \cdot W_1 + b_1) \cdot W_2 + b_2$$

### What It Does

1. `x @ W1 + b1`: Expand from 64 dimensions → 128 dimensions (to a richer representation space)
2. `ReLU`: Set negative values to 0 (introducing nonlinearity)
3. `@ W2 + b2`: Compress from 128 dimensions → back to 64 dimensions

```
x (64 dims) → Expand (128 dims) → ReLU → Compress (64 dims)
```

While Self-Attention captures "relationships between words,"
FFN plays the role of "transforming each word's representation individually."

---

## 2.6 Transformer Block — Combining Everything

```python
def transformer_block(x, layer):
    # Pre-LN: layer norm → self-attention → residual
    normed = layer_norm(x, layer["ln1_g"], layer["ln1_b"])
    attn_out = self_attention(normed, layer["Wq"], layer["Wk"], layer["Wv"], layer["Wo"])
    x = x + attn_out

    # Pre-LN: layer norm → feed-forward → residual
    normed = layer_norm(x, layer["ln2_g"], layer["ln2_b"])
    ff_out = feed_forward(normed, layer["W1"], layer["b1"], layer["W2"], layer["b2"])
    x = x + ff_out
    return x
```

> **Python Tips: `layer["Wq"]` — Retrieving parameters from a dictionary**
>
> `layer` is a Python **dictionary (dict)**.
> During model initialization, all parameters for one Transformer layer are grouped into a dictionary:
> ```python
> # From TinyTransformer.__init__
> layer = {
>     "Wq": param(64, 64),     # Query weight matrix
>     "Wk": param(64, 64),     # Key weight matrix
>     "Wv": param(64, 64),     # Value weight matrix
>     "Wo": param(64, 64),     # Output projection weight matrix
>     "ln1_g": param_ones(64), # LayerNorm1 scale
>     "ln1_b": param_zeros(64),# LayerNorm1 shift
>     "W1": param(64, 128),    # FFN first layer weights
>     "b1": param_zeros(128),  # FFN first layer bias
>     "W2": param(128, 64),    # FFN second layer weights
>     "b2": param_zeros(64),   # FFN second layer bias
>     "ln2_g": param_ones(64), # LayerNorm2 scale
>     "ln2_b": param_zeros(64),# LayerNorm2 shift
> }
> ```
> So `layer["Wq"]` means "retrieve this layer's Query weight matrix."
> Using a dictionary keeps all 12 parameters for one layer organized together.
>
> In this program, `N_LAYERS = 2`, so **two** of these dictionaries are created,
> stored in a list as `self.layers = [layer0, layer1]`.
> Each layer's parameters are separate and trained independently.

### Structure Diagram (Pre-LN: GPT-2+ style)

```
Input x ──────────────────┐
  │                       │ Residual Connection
  ▼                       │
Layer Norm                │
  │                       │
  ▼                       │
Self-Attention            │
  │                       │
  ▼                       │
  + ◄─────────────────────┘
  │
  ├──────────────────────┐
  │                      │ Residual Connection
  ▼                      │
Layer Norm               │
  │                      │
  ▼                      │
Feed-Forward             │
  │                      │
  ▼                      │
  + ◄────────────────────┘
  │
  ▼
Output x
```

> **Pre-LN vs Post-LN:** The original Transformer (2017) and GPT-1 used the order
> "sublayer → residual addition → Layer Norm" (Post-LN).
> From GPT-2 onward, this was changed to "Layer Norm → sublayer → residual addition" (Pre-LN),
> which was found to make training more stable. This program uses Pre-LN, the same as GPT-2+.

### What Is a Residual Connection?

Like `x + attn_out`, the transformation result is **added to the input**.

- **Why:** As layers get deeper, gradients tend to vanish. Residual connections create
  a bypass for gradients to flow directly to shallow layers
- **Intuition:** "Preserve the original information while adding new information"
- If the transformation is unnecessary, the model can learn `attn_out ≈ 0`, letting the input pass through as-is

This program stacks **2 blocks** in series (`N_LAYERS = 2`).

---

## 2.7 Output — From Vectors to Word Scores

```python
# End of TinyTransformer.forward
x = layer_norm(x, self.ln_f_g, self.ln_f_b)
logits = x @ self.tok_emb.T   # (28, 12, 64) @ (64, 10) → (28, 12, 10)
```

>
> **Python Tips: `.T` — Matrix transpose**
>
> `.T` swaps rows and columns (transpose):
> ```python
> x = torch.tensor([[1, 2, 3],
>                    [4, 5, 6]])    # shape: (2, 3)
> x.T                               # shape: (3, 2)
> # → tensor([[1, 4],
> #            [2, 5],
> #            [3, 6]])
> ```
> `tok_emb` is (10, 64), so `tok_emb.T` becomes (64, 10).
> Multiplying with `@` performs the transformation "64-dim vector → scores for 10 words."

### Weight Tying

We reuse `tok_emb.T` (the transpose of the Embedding) for the output projection.
This is a technique based on the intuition that
"if the output is close to the vector for 'cat,' then the next word is probably 'cat,'"
and it saves on parameter count.

### What Are Logits

The final output `logits` has shape `(28, 12, 10)`.

```
logits[sample][position][word] = score for that word being the "next word" at that position

Example: logits[0][3] = [0.1, -0.5, 0.3, 0.8, 2.1, -0.2, 0.4, -0.1, 0.6, 0.0]
                          pad   the   cat  sat   on   mat    .   dog   log  saw

→ At position 3 ("on"), "on" (=4) has the highest score → Prediction: "on"
  (The actual correct answer is "the")
```

Comparing these logits with the correct answers to compute the loss is the subject of the next chapter, "Training."

---

## Summary: Data Transformations Inside the Transformer

```
token_ids (28, 12)            ← Integers (word numbers)
    ↓ Embedding
x (28, 12, 64)               ← Each word becomes a 64-dim vector
    ↓ Self-Attention
x (28, 12, 64)               ← Transformed to context-aware vectors
    ↓ FFN
x (28, 12, 64)               ← Further transformed
    ↓ ×2 blocks
x (28, 12, 64)               ← 2 layers of processing
    ↓ Layer Norm + tok_emb.T
logits (28, 12, 10)           ← Scores for 10 words at each position
```

| Component | Shape Change | Role |
|-----------|-------------|------|
| Token Embedding | (28,12) → (28,12,64) | Turn words into numerical vectors |
| Positional Emb. | Addition | Add positional information |
| Self-Attention | (28,12,64) → (28,12,64) | Capture relationships between words |
| Layer Norm | Shape unchanged | Stabilize values |
| Feed-Forward | (28,12,64) → (28,12,64) | Transform each word's representation |
| Output Projection | (28,12,64) → (28,12,10) | Convert vectors to vocabulary scores |
