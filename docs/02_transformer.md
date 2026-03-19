# 第2章: Transformer — 文脈を理解する仕組み

前章で `inputs` テンソル (28, 12) を作りました。
この章では、この数値の列が Transformer の中でどう処理され、
「次の単語の予測」に至るかを追いかけます。

---

## 2.1 全体の流れ

```
token_ids (28, 12)
    │
    ▼
┌──────────────────────┐
│ Token Embedding      │  数値 → ベクトル
│ + Positional Emb.    │  位置情報を付加
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
│ → Logits (線形変換)   │  ベクトル → 語彙スコア
└──────────┬───────────┘
           │  (28, 12, 10)
           ▼
     各位置で10単語分のスコア
```

---

## 2.2 Embedding — 数値をベクトルに変える

### なぜベクトルが必要か

単語番号 `3`（= "sat"）はただの整数で、「意味」を持ちません。
これを **64次元のベクトル** に変換することで、単語同士の関係を
数値的に表現できるようになります。

### コード

```python
# TinyTransformer.__init__ より
self.tok_emb = param(vocab_size, D_MODEL)   # (10, 64)
self.pos_emb = param(SEQ_LEN, D_MODEL)      # (12, 64)
```

```python
# TinyTransformer.forward より
x = self.tok_emb[token_ids] + self.pos_emb[:T]
```

### ステップごとの説明

**Step 1: Token Embedding（トークン埋め込み）**

`tok_emb` は形状 `(10, 64)` のテーブルで、各行が1単語のベクトルです。

```
tok_emb = [
  行0: [0.01, -0.03, 0.02, ...],   ← "<pad>" のベクトル（64次元）
  行1: [0.05,  0.01, -0.02, ...],  ← "the" のベクトル
  行2: [-0.01, 0.04,  0.03, ...],  ← "cat" のベクトル
  ...
  行9: [0.02, -0.01,  0.05, ...],  ← "saw" のベクトル
]
```

`token_ids = [1, 2, 3, 4, ...]` のとき、`tok_emb[token_ids]` は
1行目、2行目、3行目、4行目…を取り出すだけです。

```
token_ids:     [1,     2,     3,     4,    ...]
                ↓      ↓      ↓      ↓
tok_emb から:  [the],  [cat], [sat], [on], ...   ← 各64次元ベクトル
```

結果の形状: `(28, 12)` → `(28, 12, 64)`

**Step 2: Positional Embedding（位置埋め込み）**

同じ単語 "the" でも、文頭と文中では役割が違います。
`pos_emb` は「位置0はこのベクトル、位置1はこのベクトル…」という
位置ごとの情報をもったテーブルです。

```
pos_emb = [
  位置0: [0.02, -0.01, 0.01, ...],   ← 64次元
  位置1: [0.01,  0.03, -0.02, ...],
  位置2: [-0.03, 0.02,  0.01, ...],
  ...
  位置11: [0.01, 0.01, -0.04, ...],
]
```

これを単純に足し合わせます：

```
x = tok_emb[token_ids] + pos_emb[:T]

位置0: "the" のベクトル + 位置0のベクトル = 「位置0にある the」のベクトル
位置1: "cat" のベクトル + 位置1のベクトル = 「位置1にある cat」のベクトル
...
```

> **ポイント:** `tok_emb` も `pos_emb` も **学習可能なパラメータ** です。
> 最初はランダムですが、訓練を通じて意味のある値に更新されます。

---

## 2.3 Self-Attention — Transformer の核心

Self-Attention は「文中のどの単語がどの単語に注目すべきか」を計算します。
これが Transformer の最も重要な仕組みです。

### 直感的な例

「the cat sat on the mat」という文を考えます。
"sat" の意味を理解するには「**誰が** 座ったのか（cat）」と
「**どこに** 座ったのか（mat）」を知る必要があります。

Self-Attention は、"sat" から "cat" と "mat" への「注目度」を
自動的に学習します。

### Query, Key, Value — 3つの役割

Self-Attention では、各単語が3つの顔を持ちます：

| 役割 | 意味 | たとえ |
|------|------|--------|
| **Query (Q)** | 「私は何を探しているか」 | 質問する人 |
| **Key (K)** | 「私はどんな情報を持っているか」 | 名札・ラベル |
| **Value (V)** | 「私の持つ実際の情報」 | 回答の内容 |

**検索にたとえると:**
- Q は「検索ワード」
- K は「各ページのタイトル」
- V は「各ページの本文」

Q と K の一致度（内積）が高いほど、その V が多く取り込まれます。

### 数式

入力 $x$ を3つの重み行列で射影します：

$$Q = x \cdot W_q, \quad K = x \cdot W_k, \quad V = x \cdot W_v$$

Attention スコア（単語 $i$ が単語 $j$ にどれだけ注目するか）：

$$\text{score}(i, j) = \frac{Q_i \cdot K_j^T}{\sqrt{d_k}}$$

$\sqrt{d_k}$ で割るのは、次元が大きいと内積の値が大きくなりすぎて
softmax が極端な値になるのを防ぐためです。

Softmax で正規化して注目度（重み）にします：

$$\text{attn}(i, j) = \text{softmax}_j(\text{score}(i, j))$$

最終出力は、Value の重み付き和です：

$$\text{out}_i = \sum_j \text{attn}(i, j) \cdot V_j$$

### コード — `self_attention` 関数

```python
def self_attention(x, Wq, Wk, Wv, Wo):
    B, T, D = x.shape        # B=28, T=12, D=64
    head_dim = D // N_HEADS   # 64 // 4 = 16
```

**Step 1: Q, K, V を計算**

```python
    Q = x @ Wq   # (28, 12, 64) @ (64, 64) → (28, 12, 64)
    K = x @ Wk   # 同上
    V = x @ Wv   # 同上
```

各単語の 64 次元ベクトルに重み行列を掛けて、Query, Key, Value を得ます。

**Step 2: Multi-Head に分割**

```python
    Q = Q.view(B, T, N_HEADS, head_dim).transpose(1, 2)
    # (28, 12, 64) → (28, 12, 4, 16) → (28, 4, 12, 16)
```

64 次元を 4 つのヘッド × 16 次元に分割します。
各ヘッドが異なる「観点」で注目パターンを学習します。

```
Head 0: 「主語と動詞の関係」に注目するかもしれない
Head 1: 「前置詞と名詞の関係」に注目するかもしれない
Head 2: 「隣接する単語」に注目するかもしれない
Head 3: 「文末のピリオド」に注目するかもしれない
```

（実際に何を学習するかは訓練データ次第です）

**Step 3: Attention スコアの計算**

```python
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(head_dim)
    # (28, 4, 12, 16) @ (28, 4, 16, 12)^T → (28, 4, 12, 12)
```

`scores[b][h][i][j]` = ヘッド h において、単語 i が単語 j にどれだけ注目するか。

具体的な計算（1ヘッド、3単語の場合のイメージ）：

```
         K₀    K₁    K₂          ← Key（情報のラベル）
Q₀  [ 0.8   0.1   0.1 ]    ← "the" は自分自身に強く注目
Q₁  [ 0.3   0.5   0.2 ]    ← "cat" は自分に最も注目、"the" にも少し
Q₂  [ 0.2   0.6   0.2 ]    ← "sat" は "cat" に強く注目（誰が座った？）
```

**Step 4: Causal Mask（因果マスク）**

```python
    mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
    scores = scores.masked_fill(mask, float("-inf"))
```

言語モデルでは「未来の単語を見てはいけない」という制約があります。
位置 i のトークンは、位置 0〜i のトークンだけ見ることができます。

```
マスク前:              マスク後（-inf で未来を隠す）:
     0    1    2         0     1      2
0 [ 0.8  0.1  0.1]   [ 0.8  -inf   -inf]
1 [ 0.3  0.5  0.2]   [ 0.3   0.5   -inf]
2 [ 0.2  0.6  0.2]   [ 0.2   0.6    0.2]
```

`-inf` は softmax で 0 になるため、未来のトークンの情報は完全に遮断されます。

**Step 5: Softmax**

```python
    attn_weights = F.softmax(scores, dim=-1)
```

各行を確率分布に変換します（合計 = 1.0）：

```
softmax 後:
     0     1     2
0 [ 1.0   0.0   0.0]    ← "the" は自分だけ見える
1 [ 0.35  0.65  0.0]    ← "cat" は "the" と自分
2 [ 0.15  0.55  0.30]   ← "sat" は全員見える
```

**Step 6: Value の重み付き和**

```python
    out = attn_weights @ V   # (28, 4, 12, 12) @ (28, 4, 12, 16) → (28, 4, 12, 16)
```

Attention weight で Value を混ぜ合わせます。
"sat" の出力は `0.15×V("the") + 0.55×V("cat") + 0.30×V("sat")`
→ "cat" の情報を最も多く取り込む。

**Step 7: ヘッドの結合と出力射影**

```python
    out = out.transpose(1, 2).contiguous().view(B, T, D)  # → (28, 12, 64)
    out = out @ Wo                                          # 出力射影
```

4 つのヘッドを連結して 64 次元に戻し、最後に `Wo` で混ぜ合わせます。

---

## 2.4 Layer Normalization — 値を安定させる

```python
def layer_norm(x, g, b, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return g * (x - mean) / torch.sqrt(var + eps) + b
```

各ベクトルを「平均 0、分散 1」に正規化し、スケール `g` とシフト `b` を適用します。

$$\text{LayerNorm}(x) = g \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + b$$

### なぜ必要か

Deep Learning では、層を重ねるうちにベクトルの値が極端に大きく（または小さく）
なることがあります。Layer Norm はこれを防ぎ、学習を安定させます。

### 具体例

```
入力:  [2.0, 4.0, 6.0, 8.0]
平均:  5.0
分散:  5.0
正規化: [-1.34, -0.45, 0.45, 1.34]   ← 平均0、分散1に
```

---

## 2.5 Feed-Forward Network — 各単語を個別に変換

```python
def feed_forward(x, W1, b1, W2, b2):
    return F.relu(x @ W1 + b1) @ W2 + b2
```

$$\text{FFN}(x) = \text{ReLU}(x \cdot W_1 + b_1) \cdot W_2 + b_2$$

### 何をしているか

1. `x @ W1 + b1`: 64 次元 → 128 次元に拡張（より豊かな表現空間へ）
2. `ReLU`: 負の値を 0 にする（非線形性の導入）
3. `@ W2 + b2`: 128 次元 → 64 次元に戻す

```
x (64次元) → 拡張 (128次元) → ReLU → 圧縮 (64次元)
```

Self-Attention が「単語間の関係」を捉えるのに対し、
FFN は「各単語の表現を個別に変換する」役割を担います。

---

## 2.6 Transformer Block — すべてを組み合わせる

```python
def transformer_block(x, layer):
    # Multi-head self-attention with residual connection
    attn_out = self_attention(x, layer["Wq"], layer["Wk"], layer["Wv"], layer["Wo"])
    x = layer_norm(x + attn_out, layer["ln1_g"], layer["ln1_b"])

    # Feed-forward with residual connection
    ff_out = feed_forward(x, layer["W1"], layer["b1"], layer["W2"], layer["b2"])
    x = layer_norm(x + ff_out, layer["ln2_g"], layer["ln2_b"])
    return x
```

### 構造図

```
入力 x ──────────────────┐
  │                      │ Residual Connection（残差接続）
  ▼                      │
Self-Attention           │
  │                      │
  ▼                      │
  + ◄────────────────────┘
  │
  ▼
Layer Norm
  │
  ├──────────────────────┐
  │                      │ Residual Connection
  ▼                      │
Feed-Forward             │
  │                      │
  ▼                      │
  + ◄────────────────────┘
  │
  ▼
Layer Norm
  │
  ▼
出力 x
```

### Residual Connection（残差接続）とは

`x + attn_out` のように、変換結果を **入力に足し算** します。

- **なぜ:** 層が深くなると勾配が消失しやすい。残差接続により、
  勾配が直接浅い層に伝わるバイパスが生まれる
- **直感:** 「元の情報は保持しつつ、追加情報を足す」
- 変換が不要なら `attn_out ≈ 0` を学習すれば、入力がそのまま通過する

本プログラムでは **2 ブロック** を直列に重ねます（`N_LAYERS = 2`）。

---

## 2.7 出力 — ベクトルから単語スコアへ

```python
# TinyTransformer.forward の最後
x = layer_norm(x, self.ln_f_g, self.ln_f_b)
logits = x @ self.tok_emb.T   # (28, 12, 64) @ (64, 10) → (28, 12, 10)
```

### Weight Tying（重み共有）

出力射影に `tok_emb.T`（Embedding の転置）を使い回します。
「"cat" のベクトルに近い出力なら、次の単語は "cat" だろう」という
直感に基づいた手法で、パラメータ数を節約できます。

### Logits とは

最終出力 `logits` の形状は `(28, 12, 10)` です。

```
logits[サンプル][位置][単語] = その位置で、その単語が「次の単語」である度合い

例: logits[0][3] = [0.1, -0.5, 0.3, 0.8, 2.1, -0.2, 0.4, -0.1, 0.6, 0.0]
                     pad   the   cat  sat   on   mat    .   dog   log  saw

→ 位置 3（"on"）の次の単語として "on"(=4) のスコアが最も高い → 予測: "on"
  （実際の正解は "the"）
```

この logits を正解と比較して損失を計算するのが、次章の「訓練」です。

---

## まとめ: Transformer 内部のデータ変換

```
token_ids (28, 12)            ← 整数（単語番号）
    ↓ Embedding
x (28, 12, 64)               ← 各単語が64次元ベクトルに
    ↓ Self-Attention
x (28, 12, 64)               ← 文脈を考慮したベクトルに変化
    ↓ FFN
x (28, 12, 64)               ← さらに変換
    ↓ ×2 blocks
x (28, 12, 64)               ← 2層分の処理
    ↓ Layer Norm + tok_emb.T
logits (28, 12, 10)           ← 各位置で10単語分のスコア
```

| コンポーネント | 形状変化 | 役割 |
|---------------|----------|------|
| Token Embedding | (28,12) → (28,12,64) | 単語を数値ベクトルに |
| Positional Emb. | 加算 | 位置情報を付加 |
| Self-Attention | (28,12,64) → (28,12,64) | 単語間の関係を捉える |
| Layer Norm | 形状不変 | 値を安定させる |
| Feed-Forward | (28,12,64) → (28,12,64) | 各単語の表現を変換 |
| 出力射影 | (28,12,64) → (28,12,10) | ベクトルを語彙スコアに |
