---
layout: default
title: tiny-LLM — Learn LLMs with a Minimal Transformer
---

# tiny-LLM

**Learn the entire Transformer architecture in just 1 file and ~140 lines of code**

tiny-LLM is a project that implements the core algorithms of large language models (LLMs)
like GPT in minimal code.
Self-Attention, Query/Key/Value, Multi-Head, Residual Connections —
you can understand **nearly the same mechanisms as real LLMs** through hands-on experimentation.

```
"the cat sat on" → Transformer → "the" (predict the next word)
```

---

## Tutorial / チュートリアル

### English

Start here if you're new. Run the code yourself and
build your understanding of the Transformer step by step.

| Tutorial | Content | Time |
|---|---|---|
| [Step 1: Setup and Run](docs/en/tutorial/01_setup.md) | Environment setup, running the code, checking the output | 5 min |
| [Step 2: Exploring the Data](docs/en/tutorial/02_explore_data.md) | Examine tokenization and training data with your own eyes | 10 min |
| [Step 3: Peeking Inside the Transformer](docs/en/tutorial/03_explore_model.md) | Visualize attention weights and embedding vectors | 15 min |
| [Step 4: Experiments and Modifications](docs/en/tutorial/04_experiments.md) | Change parameters, modify the corpus, and experiment | 15 min |

### 日本語

初めての方はこちらから。コードを実際に動かしながら、Transformer の仕組みを段階的に理解していきます。

| チュートリアル | 内容 | 所要時間 |
|---|---|---|
| [Step 1: セットアップと実行](docs/ja/tutorial/01_setup.md) | 環境構築、コードの実行、出力の確認 | 5分 |
| [Step 2: データを観察する](docs/ja/tutorial/02_explore_data.md) | トークン化・訓練データの中身を自分の目で確認 | 10分 |
| [Step 3: Transformer の中を覗く](docs/ja/tutorial/03_explore_model.md) | Attention の重み、埋め込みベクトルを可視化 | 15分 |
| [Step 4: 改造してみる](docs/ja/tutorial/04_experiments.md) | パラメータを変えたり、コーパスを変えて実験 | 15分 |

---

## Documentation / ドキュメント

### English

Each chapter explains the code's mechanisms in detail, along with the mathematical background.

| Chapter | Title | Content |
|---|---|---|
| [Chapter 1](docs/en/01_data.md) | Data Preparation | Vocabulary building, tokenization, creating training data |
| [Chapter 2](docs/en/02_transformer.md) | Transformer | Embedding, Self-Attention, FFN, the complete Forward Pass |
| [Chapter 3](docs/en/03_training.md) | Training | Cross-Entropy Loss, Backpropagation, Parameter Updates |
| [Supplement](docs/en/03a_gradient.md) | Gradient Math | Derivatives, partial derivatives, and the chain rule with concrete numbers |
| [Chapter 4](docs/en/04_generation.md) | Text Generation | Next word prediction, Greedy Decoding, comparison with real LLMs |

### 日本語

各章で、コードの仕組みを数学的背景とともに詳しく解説しています。

| 章 | タイトル | 内容 |
|---|---|---|
| [第1章](docs/ja/01_data.md) | データの準備 | 語彙構築、トークン化、訓練データの作り方 |
| [第2章](docs/ja/02_transformer.md) | Transformer | Embedding、Self-Attention、FFN、Forward Pass の全て |
| [第3章](docs/ja/03_training.md) | 訓練 | Cross-Entropy Loss、誤差逆伝播、パラメータ更新 |
| [補足](docs/ja/03a_gradient.md) | 勾配の数学 | 微分・偏微分・連鎖律を具体的な数値で解説 |
| [第4章](docs/ja/04_generation.md) | テキスト生成 | 次の単語の予測、Greedy Decoding、本物の LLM との比較 |

---

## Quick Start

```bash
pip install torch
python tiny_llm.py
```

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

---

## Project Characteristics

| | tiny-LLM | GPT-4 class |
|---|---|---|
| Vocabulary size | 10 | 100,000+ |
| Parameters | ~68,000 | Hundreds of billions to trillions |
| Training data | 40 words | Trillions of tokens |
| Training time | Seconds | Months (thousands of GPUs) |
| **Core algorithm** | **Same** | **Same** |

The only difference is scale. The mechanism is the same.

---

## Source Code

- [tiny_llm.py](https://github.com/t-ishii66/tiny-llm/blob/main/tiny_llm.py) — Complete implementation (1 file, ~140 lines of code)
- [GitHub Repository](https://github.com/t-ishii66/tiny-llm)

---

<small>Copyright (C) 2026 t-ishii66. All rights reserved.</small>
