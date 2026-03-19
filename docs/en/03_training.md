# Chapter 3: Training — How the Model Gets Smarter

In the previous chapter, we saw how input is processed into `logits` (scores for each word).
In this chapter, we explain the **training loop** mechanism that compares those scores
with the correct answers and updates the parameters.

---

## 3.1 The Big Picture of Training

```
┌───────────────────────────────────────────────────┐
│              Training Loop (200 iterations)        │
│                                                   │
│  ① Forward:  inputs → Transformer → logits        │
│  ② Loss:     Compare logits with targets → loss   │
│  ③ Backward: Compute gradients for all parameters │
│  ④ Update:   Slightly adjust parameters           │
│              in the gradient direction             │
│                                                   │
│  → Return to ① (repeat until loss decreases)      │
└───────────────────────────────────────────────────┘
```

---

## 3.2 The Complete Code

```python
def train(model, inputs, targets):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        logits = model.forward(inputs)               # ① Forward
        loss = F.cross_entropy(                       # ② Loss
            logits.view(-1, model.vocab_size),
            targets.view(-1),
        )

        optimizer.zero_grad()                         # Reset previous gradients
        loss.backward()                               # ③ Backward (automatic differentiation)
        optimizer.step()                              # ④ Update

        if (epoch + 1) % 20 == 0:
            print(f"epoch {epoch+1:4d}  loss={loss.item():.4f}")
```

>
> **Python Tips: `loss.item()` and f-strings**
>
> **`.item()`** extracts a plain Python number from a tensor:
> ```python
> t = torch.tensor(3.14)
> t          # → tensor(3.14)   ← tensor type
> t.item()   # → 3.14           ← float type
> ```
>
> **f-strings** `f"..."` are strings that can embed variables. After `:` is the format specifier:
> ```python
> x = 42
> pi = 3.14159
> f"x={x:4d}"      # → "x=  42"      4-digit integer (right-aligned)
> f"pi={pi:.4f}"    # → "pi=3.1416"   4 decimal places
> ```

Let's look at each step in detail below.

---

## 3.3 ① Forward — Producing Predictions

```python
logits = model.forward(inputs)   # (28, 12, 10)
```

This is exactly the processing explained in the previous chapter. For each of the
28 samples × 12 positions, scores for 10 words are output.

---

## 3.4 ② Cross-Entropy Loss — Comparing Predictions with Correct Answers

### What It Computes

```python
loss = F.cross_entropy(
    logits.view(-1, model.vocab_size),   # (336, 10) ← 28×12=336 predictions
    targets.view(-1),                     # (336,)    ← 336 correct answers
)
```

These two lines are a bit tricky, so let's go through them carefully.

### Shapes of `logits` and `targets`

The result of the forward pass, `logits`, is a 3-dimensional tensor:

```
logits shape: (28, 12, 10)
               ↑   ↑   ↑
               |   |   └─ Scores for each of the 10 words (next word predictions)
               |   └───── 12 predictions (predicting the "next word" for each input word)
               └───────── 28 samples
```

As we saw in Chapter 1, inputs and targets are in a "shifted by one" relationship:

```
Input:   the  cat  sat  on  the  mat   .  the  dog  sat  on  the
Target:  cat  sat  on   the mat   .   the dog  sat  on   the log
```

The logits at each position are **the scores predicting "the word that comes next" based on the context up to that position**.
For example, the logits at position 3 ("on") are scores for 10 words predicting
"the word that comes after 'the cat sat on'." The correct answer is "the."

`targets` contains the correct word number for each position:

```
targets shape: (28, 12)
                ↑   ↑
                |   └─ Correct word number at each position (= the word that should come next)
                └───── 28 samples
```

In other words, there are 28 samples × 12 positions = **336 "next word" predictions**,
each with **one correct answer**.

### Why Reshape with `view`

`F.cross_entropy` requires the following shapes:

- First argument: `(number of predictions, number of classes)` — 2D
- Second argument: `(number of predictions,)` — 1D

However, `logits` is 3D with shape `(28, 12, 10)`, and `targets` is 2D with shape `(28, 12)`.
So we use `view` to flatten "28 samples × 12 positions" into a single row:

```
logits:  (28, 12, 10)  →  view(-1, 10)  →  (336, 10)
                                              ↑    ↑
                                             336   Scores for 10 words

targets: (28, 12)      →  view(-1)      →  (336,)
                                              ↑
                                             336 correct answer numbers
```

To visualize concretely:

```
Before view (3D):
  Sample 0: [[0.1, 2.5, ...], [0.3, -0.1, ...], ..., [...]]  ← 12 predictions
  Sample 1: [[...], [...], ..., [...]]                         ← 12 predictions
  ...
  Sample 27: [[...], [...], ..., [...]]                        ← 12 predictions

After view (2D):
  Prediction  0: [0.1, 2.5, 0.3, -0.1, ...]   ← Sample 0, position 0: 10-word scores
  Prediction  1: [0.3, -0.1, 1.2, 0.5, ...]   ← Sample 0, position 1: 10-word scores
  ...
  Prediction 11: [...]                          ← Sample 0, position 11
  Prediction 12: [...]                          ← Sample 1, position 0 (continues to next sample)
  ...
  Prediction 335: [...]                         ← Sample 27, position 11
```

In short, **we remove the distinction of "which sample, which position"
and line up all 336 predictions in a single row, computing the loss all at once**.

>
> **Python Tips: `.view(-1, ...)` — `-1` means "auto-calculate"**
>
> When you specify `-1` in `.view()`, the size is automatically calculated from the other dimensions:
> ```python
> x = torch.zeros(28, 12, 10)      # 28×12×10 = 3360 elements
>
> x.view(-1, 10)    # → shape: (336, 10)   -1 → 3360÷10 = 336
> x.view(-1)        # → shape: (3360,)     Completely flattened to 1D
> ```

### The Meaning of Cross-Entropy Loss

Suppose the logits at a certain position are `[0.1, 2.5, 0.3, -0.1, ...]` and
the correct answer is word number `1` ("the").

**Step 1: Convert to probabilities with Softmax**

$$p_i = \frac{e^{\text{logit}_i}}{\sum_j e^{\text{logit}_j}}$$

```
logits:  [0.1,  2.5,  0.3, -0.1,  0.5, -0.2,  0.4, -0.1,  0.6,  0.0]
softmax: [0.04, 0.47, 0.05, 0.03, 0.06, 0.03, 0.06, 0.03, 0.07, 0.04]
                 ↑
              47% probability for the correct "the" → still low
```

**Step 2: Negative log of the correct answer's probability**

$$\text{loss} = -\log(p_{\text{correct}})$$

```
loss = -log(0.47) = 0.76
```

- Correct answer's probability close to 1.0 → loss ≈ 0 (good prediction)
- Correct answer's probability close to 0.0 → loss → ∞ (bad prediction)

### Loss Progression

Looking at the execution results (exact numbers vary per run, but the trend is the same):

```
epoch   20  loss=1.9469    ← Nearly random prediction (for 10 words, -log(1/10) ≈ 2.30)
epoch   40  loss=1.5257
epoch   60  loss=0.8140
epoch   80  loss=0.5469
epoch  100  loss=0.3880    ← Predictions are quite accurate now
epoch  120  loss=0.3099
epoch  140  loss=0.2568
epoch  160  loss=0.2227
epoch  180  loss=0.1862
epoch  200  loss=0.1147    ← Predictions are nearly perfect
```

The loss drops from around 2.3 (random) to 0.11 (nearly correct).

---

## 3.5 ③ Backward — Backpropagation

```python
optimizer.zero_grad()
loss.backward()
```

### What Is a "Gradient"?

> For the mathematical details of gradients, see [Supplement: Mathematical Intuition for Gradients](03a_gradient.md)
> with concrete numerical examples.

The loss `loss` is a function of all parameters.
"If I slightly increase a parameter, how much does the loss change?" —
this is the **gradient**.

Expressed mathematically, the gradient for parameter $\theta$:

$$\frac{\partial \text{loss}}{\partial \theta}$$

- Gradient is **positive** → **decreasing** that parameter will lower the loss
- Gradient is **negative** → **increasing** that parameter will lower the loss

### How Backpropagation Works

`loss.backward()` uses the **chain rule** to propagate gradients
backward from the output toward the input.

```
tok_emb → Embedding → Attention → FFN → logits → loss
  ←─────────────────────────────────────────────────
          backward: gradients flow in reverse from loss
```

The chain rule states:

$$\frac{\partial \text{loss}}{\partial W_q} = \frac{\partial \text{loss}}{\partial \text{logits}} \cdot \frac{\partial \text{logits}}{\partial \text{attn}} \cdot \frac{\partial \text{attn}}{\partial W_q}$$

By multiplying together the local derivatives at each layer, the gradients for all parameters can be obtained.

> **PyTorch's autograd**: With just the single line `loss.backward()`, PyTorch
> automatically computes the gradients for all computations performed during the forward pass.
> In this program, the forward pass is hand-written, while the backward pass is left to PyTorch.

### Why `zero_grad()` Is Necessary

PyTorch **accumulates** gradients, so without resetting them each time,
the previous gradients would remain and accumulate.

---

## 3.6 ④ Update — Updating the Parameters

```python
optimizer.step()
```

### The Basic Update Rule (Gradient Descent)

$$\theta \leftarrow \theta - \eta \cdot \frac{\partial \text{loss}}{\partial \theta}$$

$\eta$ (learning rate, `LR = 0.001`) is "how much to move in a single update."

```
Example:
  An element of W_q is 0.05, and its gradient is +0.2
  → New value = 0.05 - 0.001 × 0.2 = 0.0498
  → Moved slightly in the direction that lowers the loss
```

### Adam Optimizer

This program uses **Adam** instead of simple gradient descent.
Adam automatically adjusts the learning rate for each parameter individually, leading to faster convergence.

- Parameters updated frequently → smaller learning rate
- Parameters updated rarely → larger learning rate

---

## 3.7 Tracing the Entire Learning Process

### Epoch 1 (Initial)

```
Parameters: Random
         ↓
Forward: logits are nonsensical
         ↓
Loss:    About 2.3 (random prediction)
         ↓
Backward: Compute gradients for all parameters
         ↓
Update:  Fine-tune parameters in the gradient direction
```

### Epoch 100

```
Parameters: Considerably adjusted
         ↓
Forward: logits are quite accurate
         ↓
Loss:    About 0.39
         ↓
Backward + Update: Further fine-tuning
```

### Epoch 200 (Final)

```
Parameters: Nearly optimal
         ↓
Forward: Nearly correct predictions
         ↓
Loss:    About 0.11 (nearly perfect)
```

---

## 3.8 What Gets Learned

List of parameters in this program:

| Parameter | Shape | What It Learns |
|-----------|-------|----------------|
| `tok_emb` | (10, 64) | Meaning vector for each word |
| `pos_emb` | (12, 64) | Information vector for each position |
| `Wq, Wk, Wv` | (64, 64) ×3 ×2 layers | How to query in Attention |
| `Wo` | (64, 64) ×2 layers | How to integrate Attention output |
| `W1, b1` | (64,128), (128,) ×2 layers | FFN transformation (first half) |
| `W2, b2` | (128,64), (64,) ×2 layers | FFN transformation (second half) |
| `ln*_g, ln*_b` | (64,) various | Layer Norm scale and shift |

All of these are gradually adjusted over 200 training iterations
**in the direction that minimizes the loss**.

---

## Summary

```
Forward:   inputs → predictions (logits)
                          ↓
Loss:      Difference between predictions and correct answers → a single number
                          ↓
Backward:  Automatically compute gradients for all parameters
                          ↓
Update:    Fine-tune parameters using the gradients
                          ↓
                    Repeat → loss keeps getting smaller
```

| Concept | Meaning |
|---------|---------|
| Loss | A measure of how wrong the predictions are |
| Gradient | The direction and magnitude to move a parameter |
| Backpropagation | A method to efficiently compute gradients using the chain rule |
| Learning rate (LR) | How much to move in a single update |
| Epoch | One complete pass through the training data |
