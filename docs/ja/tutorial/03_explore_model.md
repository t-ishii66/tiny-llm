# Step 3: Transformer の中を覗く

訓練済みモデルの内部を実際に観察してみましょう。
Embedding ベクトルや Attention の重みが、実際にどんな値になっているか確認します。

引き続き `uv run --with torch python -i tiny_llm.py` の対話モードで作業します。

---

## 3.1 パラメータの数を確認する

```python
>>> total = sum(p.numel() for p in model.parameters())
>>> print(f"Total parameters: {total}")
Total parameters: 67968
```

約 68,000 個のパラメータ（主に重み行列）が、200回の訓練で調整されました。

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
tensor([-0.05,  0.13,  0.27, ...], requires_grad=True)
```

数値は訓練ごとにランダム初期化から学習されるので、実行のたびに変わります。

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

<details>
<summary>コピペ用 (プロンプト記号なし)</summary>

`>>>` / `...` を取り除いた生コード。対話モードに**そのまま貼り付け可能**です。

```python
import torch.nn.functional as F

def similarity(word1, word2):
    v1 = model.tok_emb[vocab[word1]]
    v2 = model.tok_emb[vocab[word2]]
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()

similarity("cat", "dog")    # 似た文脈で使われる
similarity("cat", ".")      # まったく違う役割
similarity("mat", "log")    # どちらも "sat on the ___" の後に来る
```

</details>

"cat" と "dog" の類似度が高く、"cat" と "." の類似度が低ければ、
モデルが単語の意味的な関係を（小さいながらも）学習したことを示しています。

---

## 3.3 Attention 重みを覗いてみる

Transformer の核は「どのトークンがどこに注目しているか」を表す **Attention 重み行列**。第1層の attention 重みを取り出して眺めてみましょう。

計算の詳細は本編 [第2章 Self-Attention](../02_transformer.md) でじっくり解説しているので、ここでは **結果を見る** ことに集中します。以下のヘルパーを対話モードに貼り付けてください：

```python
>>> import math
>>> def attn_layer0(text):
...     x = torch.tensor([tokenize(text, vocab)])
...     T = x.shape[1]
...     emb = model.tok_emb[x] + model.pos_emb[:T]
...     L = model.layers[0]
...     n = layer_norm(emb, L["ln1_g"], L["ln1_b"])
...     Q, K = n @ L["Wq"], n @ L["Wk"]
...     scores = (Q @ K.transpose(-2, -1)) / math.sqrt(64)
...     mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
...     return torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)[0]
...
```

<details>
<summary>コピペ用 (プロンプト記号なし)</summary>

`>>>` / `...` を取り除いた生コード。対話モードに**そのまま貼り付け可能**です。

```python
import math

def attn_layer0(text):
    x = torch.tensor([tokenize(text, vocab)])
    T = x.shape[1]
    emb = model.tok_emb[x] + model.pos_emb[:T]
    L = model.layers[0]
    n = layer_norm(emb, L["ln1_g"], L["ln1_b"])
    Q, K = n @ L["Wq"], n @ L["Wk"]
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(64)
    mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
    return torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)[0]
```

</details>

実行してみましょう：

```python
>>> attn = attn_layer0("the cat sat on the mat")
>>> print(attn.detach().round(decimals=2))
```

6×6 の attention 重み行列が表示されます。各行が「その位置がどこに注目しているか」を示します：

```
行 0 (the): [1.00, 0.00, 0.00, 0.00, 0.00, 0.00]   ← 自分しか見えない
行 1 (cat): [0.??, 0.??, 0.00, 0.00, 0.00, 0.00]   ← the と cat が見える
行 2 (sat): [0.??, 0.??, 0.??, 0.00, 0.00, 0.00]
...
```

**右上の三角形が 0 なのが causal mask の効果** — 「未来のトークンを見るのを禁止する」しくみが、こうして数値で確認できます。

> マルチヘッド (4 ヘッド) ごとの違いを見たい人は、ヘルパーの `Q`, `K` を 4 ヘッドに分割してから同じ計算を回してみてください。手順は本編 [§2 Multi-Head Attention](../02_transformer.md) に。

---

## 3.4 (Optional) Attention をヒートマップで描く

数値表より目で見たほうが分かりやすいので、matplotlib があれば可視化してみましょう。

```bash
# まず matplotlib を入れた上で対話モードを起動
uv run --with torch --with matplotlib python -i tiny_llm.py
```

3.3 のヘルパー `attn_layer0` を再度貼り付けてから：

```python
>>> import matplotlib.pyplot as plt
>>> tokens = "the cat sat on the mat".split()
>>> attn = attn_layer0("the cat sat on the mat")
>>>
>>> fig, ax = plt.subplots(figsize=(5, 4))
>>> im = ax.imshow(attn.detach().numpy(), cmap="Blues")
>>> ax.set_xticks(range(6)); ax.set_yticks(range(6))
>>> ax.set_xticklabels(tokens); ax.set_yticklabels(tokens)
>>> ax.set_xlabel("attended to"); ax.set_ylabel("from position")
>>> plt.colorbar(im)
>>> plt.tight_layout(); plt.show()
```

<details>
<summary>コピペ用 (プロンプト記号なし)</summary>

`>>>` を取り除いた生コード。対話モードに**そのまま貼り付け可能**です。

```python
import matplotlib.pyplot as plt
tokens = "the cat sat on the mat".split()
attn = attn_layer0("the cat sat on the mat")

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(attn.detach().numpy(), cmap="Blues")
ax.set_xticks(range(6)); ax.set_yticks(range(6))
ax.set_xticklabels(tokens); ax.set_yticklabels(tokens)
ax.set_xlabel("attended to"); ax.set_ylabel("from position")
plt.colorbar(im)
plt.tight_layout(); plt.show()
```

</details>

下三角だけが色付き (右上の causal mask 部分は白)、対角線付近が濃い、というパターンが見えるはずです。
ヘッドごとに異なる注目パターンを描き比べると、Multi-Head Attention の意義が直感的にわかります。

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

<details>
<summary>コピペ用 (プロンプト記号なし)</summary>

`>>>` / `...` を取り除いた生コード。対話モードに**そのまま貼り付け可能**です。

```python
# "the cat sat on" から次の単語を予測
prompt = "the cat sat on"
tokens = tokenize(prompt, vocab)
print(tokens)

# Forward pass
x = torch.tensor([tokens])
logits = model.forward(x)          # (1, 4, 10)
next_logit = logits[0, -1, :]      # 最後の位置のスコア

# 各単語のスコアを表示
for i, score in enumerate(next_logit.tolist()):
    print(f"  {id2word[i]:>5s}: {score:.3f}")
```

</details>

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
- **Attention 重み行列**: 因果マスクにより三角行列になる。各ヘッドが異なるパターン
- **生成**: 全単語のスコアが出て、最もスコアの高い単語が次の予測になる
- すべてが **数値テンソル** として確認できる——ブラックボックスではない

---

次へ: [Step 4: 改造してみる](04_experiments.md)
