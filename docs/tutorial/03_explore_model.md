# Step 3: Transformer の中を覗く

訓練済みモデルの内部を実際に観察してみましょう。
Embedding ベクトルや Attention の重みが、実際にどんな値になっているか確認します。

引き続き `python -i tiny_llm.py` の対話モードで作業します。

---

## 3.1 パラメータの数を確認する

```python
>>> total = sum(p.numel() for p in model.parameters())
>>> print(f"Total parameters: {total}")
Total parameters: 41034
```

約 40,000 個のパラメータが、200回の訓練で調整されました。

---

## 3.2 Embedding ベクトルを観察する

各単語は 64 次元のベクトルで表現されています：

```python
>>> model.tok_emb.shape
torch.Size([10, 64])
```

"cat"（番号2）のベクトルの最初の10要素を見てみましょう：

```python
>>> model.tok_emb[2][:10]
tensor([...], requires_grad=True)
```

訓練によって、似た役割の単語は似たベクトルになっているはずです。
コサイン類似度で確認してみましょう：

```python
>>> import torch.nn.functional as F
>>>
>>> def similarity(word1, word2):
...     v1 = model.tok_emb[vocab[word1]]
...     v2 = model.tok_emb[vocab[word2]]
...     return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
...
>>> similarity("cat", "dog")    # 似た文脈で使われる
>>> similarity("cat", ".")      # まったく違う役割
>>> similarity("mat", "log")    # どちらも "sat on the ___" の後に来る
```

"cat" と "dog" の類似度が高く、"cat" と "." の類似度が低ければ、
モデルが単語の意味的な関係を（小さいながらも）学習したことを示しています。

---

## 3.3 Attention の重みを可視化する

Transformer がどの単語に注目しているか見てみましょう。

```python
>>> # テスト文をトークン化して Forward pass
>>> test_tokens = tokenize("the cat sat on the mat", vocab)
>>> x = torch.tensor([test_tokens])   # (1, 6)
>>> x.shape
torch.Size([1, 6])
```

Attention スコアを取得するために、手動で途中まで計算します：

```python
>>> import math
>>>
>>> # Embedding
>>> emb = model.tok_emb[x] + model.pos_emb[:6]   # (1, 6, 64)
>>>
>>> # 第1層の Q, K を計算
>>> layer = model.layers[0]
>>> Q = emb @ layer["Wq"]    # (1, 6, 64)
>>> K = emb @ layer["Wk"]    # (1, 6, 64)
>>>
>>> # Attention スコア（マスク前）
>>> scores = Q @ K.transpose(-2, -1) / math.sqrt(64)   # (1, 6, 6)
>>>
>>> # Causal mask 適用 + Softmax
>>> mask = torch.triu(torch.ones(6, 6), diagonal=1).bool()
>>> scores = scores.masked_fill(mask, float("-inf"))
>>> attn = torch.softmax(scores, dim=-1)
>>>
>>> print(attn[0].detach())
```

6×6 の Attention 行列が表示されます。各行が「その位置がどの位置に注目しているか」です：

```
行0 (the):  [1.00, 0.00, 0.00, 0.00, 0.00, 0.00]  ← 自分だけ見える
行1 (cat):  [0.??, 0.??, 0.00, 0.00, 0.00, 0.00]  ← the と cat が見える
行2 (sat):  [0.??, 0.??, 0.??, 0.00, 0.00, 0.00]
...
```

右上が 0.00 になっているのが **因果マスク** の効果です。
未来の単語を見ることはできません。

---

## 3.4 全ヘッドの Attention を比較する

マルチヘッド（4ヘッド）では、各ヘッドが異なるパターンで注目します：

```python
>>> # Q, K を4ヘッドに分割（head_dim = 64 / 4 = 16）
>>> B, T, D = 1, 6, 64
>>> head_dim = D // 4
>>>
>>> q = Q.view(B, T, 4, head_dim).transpose(1, 2)  # (1, 4, 6, 16)
>>> k = K.view(B, T, 4, head_dim).transpose(1, 2)  # (1, 4, 6, 16)
>>>
>>> scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
>>> mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
>>> scores = scores.masked_fill(mask, float("-inf"))
>>> attn_heads = torch.softmax(scores, dim=-1)
>>>
>>> # 各ヘッドの Attention を表示
>>> for h in range(4):
...     print(f"\n--- Head {h} ---")
...     print(attn_heads[0, h].detach().round(decimals=2))
```

各ヘッドが異なるパターンで注目しているのを観察できます。
あるヘッドは直前の単語に注目し、別のヘッドは文頭の単語に注目するかもしれません。

---

## 3.5 生成の過程を1ステップずつ追う

```python
>>> # "the cat sat on" から次の単語を予測
>>> prompt = "the cat sat on"
>>> tokens = tokenize(prompt, vocab)
>>> print(tokens)
[1, 2, 3, 4]

>>> # Forward pass
>>> x = torch.tensor([tokens])
>>> logits = model.forward(x)          # (1, 4, 10)
>>> next_logit = logits[0, -1, :]      # 最後の位置のスコア

>>> # 各単語のスコアを表示
>>> for i, score in enumerate(next_logit.tolist()):
...     print(f"  {id2word[i]:>5s}: {score:.3f}")
```

最もスコアの高い単語が、`argmax` で選ばれます：

```python
>>> next_id = torch.argmax(next_logit).item()
>>> print(f"predicted: {id2word[next_id]}")
```

"the cat sat on" の次に "the" が予測されるはずです
（コーパスに "the cat sat on the mat" があるため）。

---

## 3.6 ここまでのポイント

- **Embedding**: 訓練により、似た文脈の単語は似たベクトルになる
- **Attention 行列**: 因果マスクにより三角行列になる。各ヘッドが異なるパターン
- **生成**: 全単語のスコアが出て、最もスコアの高い単語が次の予測になる
- すべてが **数値テンソル** として確認できる——ブラックボックスではない

---

次へ: [Step 4: 改造してみる](04_experiments.md)
