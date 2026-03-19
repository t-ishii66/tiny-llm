---
layout: default
title: tiny-LLM — 最小の Transformer で LLM を学ぶ
---

# tiny-LLM

**たった1ファイル・260行で Transformer の全てを学ぶ**

tiny-LLM は、GPT などの大規模言語モデル（LLM）のコアアルゴリズムを
最小限のコードで実装したプロジェクトです。
Self-Attention、Query/Key/Value、マルチヘッド、残差接続——
本物の LLM と **まったく同じ仕組み** を、手を動かしながら理解できます。

```
"the cat sat on" → Transformer → "the" (次の単語を予測)
```

---

## チュートリアル

初めての方はこちらから。コードを実際に動かしながら、
Transformer の仕組みを段階的に理解していきます。

| チュートリアル | 内容 | 所要時間 |
|---|---|---|
| [Step 1: セットアップと実行](docs/tutorial/01_setup.md) | 環境構築、コードの実行、出力の確認 | 5分 |
| [Step 2: データを観察する](docs/tutorial/02_explore_data.md) | トークン化・訓練データの中身を自分の目で確認 | 10分 |
| [Step 3: Transformer の中を覗く](docs/tutorial/03_explore_model.md) | Attention の重み、埋め込みベクトルを可視化 | 15分 |
| [Step 4: 改造してみる](docs/tutorial/04_experiments.md) | パラメータを変えたり、コーパスを変えて実験 | 15分 |

---

## ドキュメント

各章で、コードの仕組みを数学的背景とともに詳しく解説しています。

| 章 | タイトル | 内容 |
|---|---|---|
| [第1章](docs/01_data.md) | データの準備 | 語彙構築、トークン化、訓練データの作り方 |
| [第2章](docs/02_transformer.md) | Transformer | Embedding、Self-Attention、FFN、Forward Pass の全て |
| [第3章](docs/03_training.md) | 訓練 | Cross-Entropy Loss、誤差逆伝播、パラメータ更新 |
| [補足](docs/03a_gradient.md) | 勾配の数学 | 微分・偏微分・連鎖律を具体的な数値で解説 |
| [第4章](docs/04_generation.md) | テキスト生成 | 次の単語の予測、Greedy Decoding、本物の LLM との比較 |

---

## クイックスタート

```bash
pip install torch
python tiny_llm.py
```

実行すると、訓練の経過と生成結果が表示されます：

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

## このプロジェクトの特徴

| | tiny-LLM | GPT-4 クラス |
|---|---|---|
| 語彙数 | 10 | 100,000+ |
| パラメータ数 | 約 40,000 | 数千億〜数兆 |
| 訓練データ | 40 単語 | 数兆トークン |
| 訓練時間 | 数秒 | 数ヶ月（数千GPU） |
| **コアアルゴリズム** | **同じ** | **同じ** |

違いはスケールだけ。仕組みは同じです。

---

## ソースコード

- [tiny_llm.py](https://github.com/t-ishii66/tiny-llm/blob/main/tiny_llm.py) — 全実装（1ファイル、約260行）
- [GitHub リポジトリ](https://github.com/t-ishii66/tiny-llm)

---

<small>Copyright (C) 2026 t-ishii66. All rights reserved.</small>
