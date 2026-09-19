---
title: "tiny-LLM from scratch — Learn LLMs with a Minimal Transformer"
description: "tiny-LLM is a minimal Transformer learning project. Understand Self-Attention, QKV, Multi-Head Attention, training, and text generation in about 140 lines of executable code (instruction tuning adds about 100 more)."
keywords: "tiny-LLM, Transformer, LLM, GPT, Self-Attention, Query Key Value, QKV, Multi-Head Attention, LayerNorm, Residual Connection, PyTorch, machine learning, deep learning, NLP, language model, generative AI, AI tutorial"
lang: en
permalink: /
canonical_url: "https://t-ishii66.github.io/tiny-llm/"
---

**English** | [日本語](README-jp.md)

<p>
  <img src="docs/images/top.png" alt="tiny-LLM top image" width="720" style="max-width: 100%; height: auto;">
</p>

# tiny-LLM from scratch

A single-file Transformer implementation for learning, in the most concise Python code possible, the algorithms at the core of large language models (LLMs) like GPT — Self-Attention, Query/Key/Value, Multi-Head Attention, and next-token prediction.

## About This Project

This is a GPT-style Transformer pared down to the bare minimum. The model body fits in one file (`tiny_llm.py`, about 140 lines of executable code; the instruction tuning covered in Chapter 5 adds about 100 more lines in `tiny_llm_instruct.py`), and it trains on a toy corpus in a few seconds. The forward pass is written by hand, and only the backward pass is left to PyTorch's autograd.

```
"the cat sat on" → Transformer → "the" (predicting the next word)
```

## What You Can Learn

- **Embedding**: how words are converted into vectors
- **Positional Embedding (learned)**: how position information is embedded
- **Self-Attention (Q, K, V)**: how tokens direct attention at each other
- **Multi-Head Attention**: how multiple attention patterns are run in parallel
- **Causal Masking**: how future tokens are hidden during training
- **Feed-Forward Network**: how each token is transformed individually
- **Residual Connections and Layer Norm**: how deep networks are made trainable
- **Training with Cross-Entropy Loss**: how the model learns to predict the next word
- **Text Generation**: how a trained model generates text one token at a time

## Simplifications

This is a learning tool, not a production-grade model. The main simplifications are as follows.

| Aspect | tiny-LLM | Production LLMs |
|--------|----------|-----------------|
| Tokenizer | whitespace split (word = token) | BPE / SentencePiece (subword) |
| Vocabulary | 10 words | 50,000 to 200,000+ tokens |
| Number of parameters | about 68,000 | billions to trillions |
| Training data | 40 tokens drawn from a 10-word vocabulary (a total count including repeats of the same word) | trillions of tokens |
| Generation | Greedy (argmax) | sampling with temperature, top-k, top-p |
| Dropout / regularization | none | Dropout, weight decay, etc. |
| **Core algorithm** | **the same** | **the same** |

## Why It's Still Useful

Even with all these simplifications, the core algorithms implemented here are used as-is in state-of-the-art models such as GPT and LLaMA. The difference is mainly one of scale, while the underlying structure is shared. Once you understand this code, Q/K/V projections, Scaled Dot-Product Attention, the Causal Mask, Residual Connections, Layer Normalization, and autoregressive generation — all of these carry over directly to practical Transformer implementations, so it becomes a foundation for reading real-world code.

## Quick Start

```bash
uv run --with torch tiny_llm.py
```

If you don't have uv installed, see the installation instructions in [Tutorial Step 1](docs/en/tutorial/01_setup.md).

Running the script displays the training progress and the generation results (the numbers vary slightly from run to run).

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
| [Chapter 1: Data Preparation](docs/en/01_data.md) | Building the vocabulary, tokenization, and how to make training data |
| [Chapter 2: Transformer](docs/en/02_transformer.md) | Embedding, Self-Attention, FFN, and the whole Forward Pass |
| [Chapter 3: Training](docs/en/03_training.md) | Cross-Entropy Loss, backpropagation, and parameter updates |
| [Chapter 3 Supplement: Gradient Math](docs/en/03a_gradient.md) | Derivatives, partial derivatives, and the chain rule explained with concrete numbers |
| [Chapter 4: Text Generation](docs/en/04_generation.md) | Next-word prediction, Greedy Decoding, and comparison with real LLMs |
| [Chapter 5: Instruction Tuning](docs/en/05_instruction_tuning.md) | Alpaca format, Response masking, and how to build an instruction-following LLM |

### Tutorial

| Tutorial | Content |
|---|---|
| [Step 1: Setup and Running](docs/en/tutorial/01_setup.md) | Setting up the environment, running the code, checking the output |
| [Step 2: Exploring the Data](docs/en/tutorial/02_explore_data.md) | Check tokenization and the contents of the training data with your own eyes |
| [Step 3: Peeking Inside the Transformer](docs/en/tutorial/03_explore_model.md) | Visualize the attention weights and embedding vectors |
| [Step 4: Experiments and Modifications](docs/en/tutorial/04_experiments.md) | Experiment by changing parameters and changing the corpus |
| [Step 5: Try Instruction Tuning](docs/en/tutorial/05_instruction.md) | Run Alpaca-style Instruction Tuning in 3 stages and feel the limits of rote memorization |

## Credits

- Project Planning: t-ishii66
- Architecture Design: t-ishii66
- Programming: Claude Opus 6, t-ishii66
- Documentation: Claude Opus 6, GPT 5.3 Codex, t-ishii66
- Review: t-ishii66
- English translation: Claude Opus 6, GPT 5.3 Codex
- Release date: 2026/9/19
- Version: 2.0.0

Copyright(C) 2026 t-ishii66. All rights reserved.
