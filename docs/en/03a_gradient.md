# Supplement: Mathematical Intuition for Gradients

Chapter 3 introduced the concept of "gradients."
Here, we explain the meaning of gradients mathematically using concrete numerical examples.

---

## Single Variable — Derivatives

Let's start with the simplest case.

Suppose there is only one parameter ($w$) and the loss is the following function:

$$\text{loss}(w) = (w - 3)^2$$

This function reaches its minimum value of 0 when $w = 3$.

The **derivative** is the ratio of how much the loss changes when $w$ is moved slightly:

$$\frac{d\,\text{loss}}{d\,w} = 2(w - 3)$$

Let's look at concrete values:

```
When w = 5.0:
  loss = (5-3)² = 4.0
  derivative = 2(5-3) = +4.0    ← Positive: decreasing w will reduce loss

When w = 1.0:
  loss = (1-3)² = 4.0
  derivative = 2(1-3) = -4.0    ← Negative: increasing w will reduce loss

When w = 3.0:
  loss = (3-3)² = 0.0
  derivative = 2(3-3) = 0.0     ← Zero: optimal point reached
```

The **update rule** moves $w$ in the **opposite direction** of the derivative:

$$w \leftarrow w - \eta \cdot \frac{d\,\text{loss}}{d\,w}$$

With $\eta = 0.1$ (learning rate):

```
Step 1:  w = 5.0  →  5.0 - 0.1 × 4.0  = 4.6
Step 2:  w = 4.6  →  4.6 - 0.1 × 3.2  = 4.28
Step 3:  w = 4.28 →  4.28 - 0.1 × 2.56 = 4.024
  ...gradually approaching w = 3.0
```

This is the essence of **Gradient Descent**.

---

## Two Variables — Partial Derivatives

When there are two parameters ($w_1, w_2$), we use **partial derivatives**:

$$\text{loss}(w_1, w_2) = (w_1 - 3)^2 + (w_2 + 1)^2$$

The minimum is loss = 0 when $w_1 = 3, w_2 = -1$.

A partial derivative is "the rate of change when moving only one parameter while keeping the others fixed":

$$\frac{\partial\,\text{loss}}{\partial\,w_1} = 2(w_1 - 3), \quad \frac{\partial\,\text{loss}}{\partial\,w_2} = 2(w_2 + 1)$$

The vector combining these two is the **gradient**:

$$\nabla \text{loss} = \left(\frac{\partial\,\text{loss}}{\partial\,w_1},\; \frac{\partial\,\text{loss}}{\partial\,w_2}\right)$$

Updates are performed on all parameters **simultaneously**:

```
When w1 = 5.0, w2 = 1.0:
  ∂loss/∂w1 = 2(5-3)  = +4.0
  ∂loss/∂w2 = 2(1+1)  = +4.0

Update:
  w1 = 5.0 - 0.1 × 4.0 = 4.6
  w2 = 1.0 - 0.1 × 4.0 = 0.6
```

Even with two variables, it's the same thing as one variable — just done **independently for each parameter**.

---

## 68,000 Variables — tiny-LLM

tiny-LLM has approximately 68,000 parameters.
But what it does is exactly the same as the two-variable case:

$$w_i \leftarrow w_i - \eta \cdot \frac{\partial\,\text{loss}}{\partial\,w_i} \quad (i = 1, 2, \ldots, 68000)$$

Computing 68,000 partial derivatives by hand is impossible.
However, using the **chain rule**, they can be calculated mechanically.

---

## Chain Rule

The derivative of a composite function $y = f(g(x))$ is:

$$\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}$$

Let's see a concrete example. When $g(x) = 2x + 1$ and $f(g) = g^2$:

```
When x = 3:
  g = 2×3 + 1 = 7
  y = 7² = 49

  dg/dx = 2
  dy/dg = 2g = 14
  dy/dx = 14 × 2 = 28     ← Just multiplication
```

A Transformer is a composition of many functions:

```
x → Embedding → Attention → FFN → ... → logits → loss
```

Applying the chain rule repeatedly:

$$\frac{\partial\,\text{loss}}{\partial\,W_q} = \frac{\partial\,\text{loss}}{\partial\,\text{logits}} \cdot \frac{\partial\,\text{logits}}{\partial\,\text{attn}} \cdot \frac{\partial\,\text{attn}}{\partial\,W_q}$$

Just by **multiplying the local derivatives at each layer**, we can obtain gradients even for parameters close to the input.

This is **Backpropagation**.
PyTorch's `loss.backward()` performs this chain rule computation automatically.

---

## Summary

| Concept | Meaning |
|---------|---------|
| Derivative $\frac{d\,\text{loss}}{dw}$ | How much loss changes when $w$ is moved slightly |
| Partial derivative $\frac{\partial\,\text{loss}}{\partial w_i}$ | Rate of change when only $w_i$ is moved while others are fixed |
| Gradient $\nabla\text{loss}$ | Vector combining all partial derivatives |
| Chain rule | Computing derivatives of composite functions as the product of each stage's derivatives |
| Gradient descent | Moving parameters in the opposite direction of the gradient to reduce loss |
| `loss.backward()` | Automatically applies the chain rule to compute gradients for all parameters |

The important point is that the **principle is the same** whether there is 1 variable or 68,000.
"Move each parameter slightly in the direction that reduces loss" — that's all there is to it.
