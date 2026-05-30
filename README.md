---
title: "tiny-LLM from scratch — Learn LLMs with a Minimal Transformer"
description: "tiny-LLM is a minimal Transformer learning project. Understand Self-Attention, QKV, multi-head attention, training, and text generation with about 140 lines of executable code (+ ~100 lines for instruction tuning)."
keywords: "tiny-LLM, Transformer, LLM, GPT, Self-Attention, Query Key Value, QKV, Multi-Head Attention, LayerNorm, Residual Connection, PyTorch, machine learning, deep learning, NLP, language model, generative AI, AI tutorial"
permalink: /
canonical_url: "https://t-ishii66.github.io/tiny-llm/"
---

**English** | [日本語](README-jp.md)

<p>
  <img src="top.png" alt="tiny-LLM top image" width="720" style="max-width: 100%; height: auto;">
</p>

# tiny-LLM from scratch

A single-file Transformer implementation designed to teach the core algorithms behind large language models — self-attention, Query/Key/Value, multi-head attention, and next-token prediction — in the most concise Python code possible.

## What This Is

This project strips a GPT-style Transformer down to its bare essentials. The model body fits in one file (`tiny_llm.py`, ~140 lines of executable code; the instruction tuning covered in Chapter 5 adds ~100 more lines in `tiny_llm_instruct.py`) and trains in seconds on a toy corpus. The forward pass is written by hand; only backpropagation is delegated to PyTorch's autograd.

```
"the cat sat on" → Transformer → "the" (predicted next word)
```

## What You'll Learn

- **Embedding**: how words become vectors
- **Positional Embedding (learned)**: how position information is injected
- **Self-Attention (Q, K, V)**: how tokens attend to each other
- **Multi-Head Attention**: how multiple attention patterns work in parallel
- **Causal Masking**: how future tokens are hidden during training
- **Feed-Forward Network**: how each token is individually transformed
- **Residual Connections & Layer Norm**: how deep networks stay trainable
- **Training with Cross-Entropy Loss**: how the model learns to predict the next word
- **Text Generation**: how trained models produce text one token at a time

## Simplifications

This is a learning tool, not a production model. Key simplifications include:

| Aspect | tiny-LLM | Production LLMs |
|--------|----------|-----------------|
| Tokenizer | Whitespace split (word = token) | BPE / SentencePiece (subword) |
| Vocabulary | 10 words | 50,000–200,000+ tokens |
| Parameters | ~68,000 | Billions to trillions |
| Training data | 40 words | Trillions of tokens |
| Generation | Greedy (argmax) | Sampling with temperature, top-k, top-p |
| Dropout / regularization | None | Dropout, weight decay, etc. |
| **Core algorithm** | **Same** | **Same** |

## Why It's Still Useful

Even with these simplifications, the core algorithms implemented here are the same ones used in GPT, LLaMA, and other state-of-the-art models. The differences are primarily about scale, while the foundational structure is shared. Understanding this code gives you a solid foundation for reading real-world Transformer implementations, because every concept here — Q/K/V projections, scaled dot-product attention, causal masks, residual connections, layer normalization, and autoregressive generation — carries over directly.

## Quick Start

```bash
uv run --with torch tiny_llm.py
```

If you don't have uv installed yet, see the install steps in [Tutorial Step 1](docs/en/tutorial/01_setup.md).

Running the script displays the training progress and generation results (exact numbers may vary between runs):

```
epoch   20  loss=1.9469
epoch   40  loss=1.5257
...
epoch  200  loss=0.1147

prompt: "the cat sat on"
output: the cat sat on the mat . the dog sat on the log .
        the cat saw the dog . the dog saw the
```

## Documentation

| Document | Content |
|---|---|
| [Chapter 1: Data Preparation](docs/en/01_data.md) | Vocabulary, tokenization, and training data construction |
| [Chapter 2: Transformer](docs/en/02_transformer.md) | Embedding, self-attention, FFN, and the full forward pass |
| [Chapter 3: Training](docs/en/03_training.md) | Loss function, backpropagation, and parameter updates |
| [Chapter 3 Supplement: Gradient Math](docs/en/03a_gradient.md) | Derivatives, partial derivatives, and chain rule with concrete examples |
| [Chapter 4: Generation](docs/en/04_generation.md) | Next-word prediction, greedy decoding, and comparison with real LLMs |
| [Chapter 5: Instruction Tuning](docs/en/05_instruction_tuning.md) | Alpaca format, response masking, and building an instruction-following LLM |

### Tutorial

| Tutorial | Content | Time |
|---|---|---|
| [Step 1: Setup and Run](docs/en/tutorial/01_setup.md) | Environment setup, running the code, checking the output | 5 min |
| [Step 2: Exploring the Data](docs/en/tutorial/02_explore_data.md) | Examine tokenization and training data with your own eyes | 10 min |
| [Step 3: Peeking Inside the Transformer](docs/en/tutorial/03_explore_model.md) | Visualize attention weights and embedding vectors | 15 min |
| [Step 4: Experiments and Modifications](docs/en/tutorial/04_experiments.md) | Change parameters, modify the corpus, and experiment | 15 min |
| [Step 5: Try Instruction Tuning](docs/en/tutorial/05_instruction.md) | Run the 3-stage Alpaca-style instruction tuning pipeline and see the limits of memorization | 15 min |

## Credits

- Project Planning: t-ishii66
- Architecture Design: t-ishii66
- Programming: Claude Opus 4.7, t-ishii66
- Document: Claude Opus 4.7, GPT 5.3 Codex, t-ishii66
- Review: t-ishii66
- English translation: Claude Opus 4.7, GPT 5.3 Codex

Copyright(C) 2026 t-ishii66. All rights reserved.
