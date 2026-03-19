# tiny-LLM

A single-file Transformer implementation designed to teach the core algorithms behind large language models — self-attention, Query/Key/Value, multi-head attention, and next-token prediction — in the most concise Python code possible.

## What This Is

This project strips a GPT-style Transformer down to its bare essentials. The entire model fits in one file (`tiny_llm.py`, ~140 lines of executable code) and trains in seconds on a toy corpus. The forward pass is written by hand; only backpropagation is delegated to PyTorch's autograd.

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

## Why It's Still Useful

Despite these simplifications, the core algorithms are **identical** to those used in GPT, LLaMA, and other state-of-the-art models. The difference is scale, not structure. Understanding this code gives you a solid foundation for reading real-world Transformer implementations, because every concept here — Q/K/V projections, scaled dot-product attention, causal masks, residual connections, layer normalization, and autoregressive generation — carries over directly.

## Quick Start

```bash
pip install torch
python tiny_llm.py
```

## Documentation

### English

1. **[Data Preparation](docs/en/01_data.md)** — Vocabulary, tokenization, and training data construction
2. **[Transformer](docs/en/02_transformer.md)** — Embedding, self-attention, FFN, and the full forward pass
3. **[Training](docs/en/03_training.md)** — Loss function, backpropagation, and parameter updates
   - **[Gradient Math Supplement](docs/en/03a_gradient.md)** — Derivatives, partial derivatives, and chain rule with concrete examples
4. **[Generation](docs/en/04_generation.md)** — Next-word prediction and the path to real LLMs

**Tutorial:**
[Step 1: Setup and Run](docs/en/tutorial/01_setup.md) → [Step 2: Exploring the Data](docs/en/tutorial/02_explore_data.md) → [Step 3: Peeking Inside the Transformer](docs/en/tutorial/03_explore_model.md) → [Step 4: Experiments and Modifications](docs/en/tutorial/04_experiments.md)

### 日本語

1. **[データの準備](docs/ja/01_data.md)** — 語彙構築、トークン化、訓練データの作り方
2. **[Transformer](docs/ja/02_transformer.md)** — Embedding、Self-Attention、FFN、Forward Pass の全て
3. **[訓練](docs/ja/03_training.md)** — Cross-Entropy Loss、誤差逆伝播、パラメータ更新
   - **[勾配の数学（補足）](docs/ja/03a_gradient.md)** — 微分・偏微分・連鎖律を具体的な数値で解説
4. **[テキスト生成](docs/ja/04_generation.md)** — 次の単語の予測、Greedy Decoding、本物の LLM との比較

**チュートリアル:**
[Step 1: セットアップと実行](docs/ja/tutorial/01_setup.md) → [Step 2: データを観察する](docs/ja/tutorial/02_explore_data.md) → [Step 3: Transformer の中を覗く](docs/ja/tutorial/03_explore_model.md) → [Step 4: 改造してみる](docs/ja/tutorial/04_experiments.md)

## Credits

- Planning: t-ishii66
- Programming: Claude Opus 4.6, t-ishii66
- Document: Claude Opus 4.6, t-ishii66
- Review: t-ishii66
- English translation: Claude Opus 4.6, GPT 5.3 Codex

Copyright(C) 2026 t-ishii66. All rights reserved.
