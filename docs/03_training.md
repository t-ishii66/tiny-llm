# 第3章: 訓練 — モデルはどうやって賢くなるのか

前章では入力がどう処理されて `logits`（各単語のスコア）になるかを見ました。
この章では、そのスコアと正解を比較し、パラメータを更新する
**訓練ループ** の仕組みを解説します。

---

## 3.1 訓練の全体像

```
┌───────────────────────────────────────────────────┐
│                  訓練ループ (200回)                │
│                                                   │
│  ① Forward:  inputs → Transformer → logits        │
│  ② Loss:     logits と targets を比較 → 損失      │
│  ③ Backward: 損失から全パラメータの勾配を計算      │
│  ④ Update:   勾配の方向にパラメータを少し調整      │
│                                                   │
│  → ①に戻る（損失が小さくなるまで繰り返す）         │
└───────────────────────────────────────────────────┘
```

---

## 3.2 コード全体

```python
def train(model, inputs, targets):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        logits = model.forward(inputs)               # ① Forward
        loss = F.cross_entropy(                       # ② Loss
            logits.view(-1, model.vocab_size),
            targets.view(-1),
        )

        optimizer.zero_grad()                         # 前回の勾配をリセット
        loss.backward()                               # ③ Backward（自動微分）
        optimizer.step()                              # ④ Update

        if (epoch + 1) % 20 == 0:
            print(f"epoch {epoch+1:4d}  loss={loss.item():.4f}")
```

以下、各ステップを詳しく見ていきます。

---

## 3.3 ① Forward — 予測を出す

```python
logits = model.forward(inputs)   # (28, 12, 10)
```

前章で解説した処理そのものです。28 サンプル × 12 位置のそれぞれで、
10 単語の「スコア」が出力されます。

---

## 3.4 ② Cross-Entropy Loss — 予測と正解を比較する

### 何を計算しているか

```python
loss = F.cross_entropy(
    logits.view(-1, model.vocab_size),   # (336, 10) ← 28×12=336 予測
    targets.view(-1),                     # (336,)    ← 336 個の正解
)
```

`view(-1, ...)` は2次元テンソルを1次元に「平坦化」します。
全ての位置の予測を一列に並べて、一括で損失を計算します。

### Cross-Entropy Loss の意味

ある位置の logits が `[0.1, 2.5, 0.3, -0.1, ...]` で、
正解が単語番号 `1`（"the"）だったとします。

**Step 1: Softmax で確率に変換**

$$p_i = \frac{e^{\text{logit}_i}}{\sum_j e^{\text{logit}_j}}$$

```
logits:  [0.1,  2.5,  0.3, -0.1,  0.5, -0.2,  0.4, -0.1,  0.6,  0.0]
softmax: [0.04, 0.47, 0.05, 0.03, 0.06, 0.03, 0.06, 0.03, 0.07, 0.04]
                 ↑
              正解の "the" に 47% の確率 → まだ低い
```

**Step 2: 正解の確率の負の対数**

$$\text{loss} = -\log(p_{\text{correct}})$$

```
loss = -log(0.47) = 0.76
```

- 正解の確率が 1.0 に近い → loss ≈ 0（良い予測）
- 正解の確率が 0.0 に近い → loss → ∞（悪い予測）

### 損失の推移

実行結果を見ると：

```
epoch   20  loss=1.9469    ← ほぼランダムな予測（10単語の場合 -log(1/10) ≈ 2.30）
epoch   40  loss=1.5257
epoch   60  loss=0.8140
epoch   80  loss=0.5469
epoch  100  loss=0.3880    ← かなり正確に予測できている
epoch  120  loss=0.3099
epoch  140  loss=0.2568
epoch  160  loss=0.2227
epoch  180  loss=0.1862
epoch  200  loss=0.1147    ← ほぼ完璧に予測できている
```

loss が 2.3 付近（ランダム）から 0.11（ほぼ正解）へと下がっています。

---

## 3.5 ③ Backward — 誤差逆伝播（Backpropagation）

```python
optimizer.zero_grad()
loss.backward()
```

### 「勾配」とは何か

損失 `loss` は全パラメータの関数です。
「あるパラメータを少し増やしたら loss はどれだけ変わるか」——
これが **勾配（gradient）** です。

数式で書くと、パラメータ $\theta$ に対する勾配：

$$\frac{\partial \text{loss}}{\partial \theta}$$

- 勾配が **正** → そのパラメータを **減らせば** loss が下がる
- 勾配が **負** → そのパラメータを **増やせば** loss が下がる

### 逆伝播の仕組み

`loss.backward()` は **連鎖律（chain rule）** を使って、
出力から入力に向かって勾配を逆方向に伝播させます。

```
tok_emb → Embedding → Attention → FFN → logits → loss
  ←─────────────────────────────────────────────────
          backward: 勾配が loss から逆向きに流れる
```

連鎖律とは：

$$\frac{\partial \text{loss}}{\partial W_q} = \frac{\partial \text{loss}}{\partial \text{logits}} \cdot \frac{\partial \text{logits}}{\partial \text{attn}} \cdot \frac{\partial \text{attn}}{\partial W_q}$$

各層の局所的な微分を掛け合わせるだけで、全パラメータの勾配が求まります。

> **PyTorch の autograd**: `loss.backward()` 一行で、forward で行った
> 全計算の勾配を自動的に求めてくれます。本プログラムでは forward を
> 手書きし、backward は PyTorch に任せています。

### `zero_grad()` の必要性

PyTorch は勾配を **累積** するため、毎回リセットしないと前回の勾配が
残ってしまいます。

---

## 3.6 ④ Update — パラメータを更新する

```python
optimizer.step()
```

### 基本の更新則（勾配降下法）

$$\theta \leftarrow \theta - \eta \cdot \frac{\partial \text{loss}}{\partial \theta}$$

$\eta$（学習率、`LR = 0.001`）は「1回の更新でどれだけ動かすか」です。

```
例：
  W_q の要素が 0.05 で、勾配が +0.2 のとき
  → 新しい値 = 0.05 - 0.001 × 0.2 = 0.0498
  → loss を下げる方向に少しだけ動いた
```

### Adam Optimizer

本プログラムでは単純な勾配降下法ではなく **Adam** を使っています。
Adam は各パラメータごとに学習率を自動調整する手法で、収束が速いです。

- 頻繁に更新されるパラメータ → 学習率を小さく
- あまり更新されないパラメータ → 学習率を大きく

---

## 3.7 学習の全過程を追う

### Epoch 1（最初）

```
パラメータ: ランダム
         ↓
Forward: logits はでたらめ
         ↓
Loss:    約 2.3（ランダム予測）
         ↓
Backward: 全パラメータの勾配を計算
         ↓
Update:  勾配の方向にパラメータを微調整
```

### Epoch 100

```
パラメータ: だいぶ調整済み
         ↓
Forward: logits がかなり正確に
         ↓
Loss:    約 0.39
         ↓
Backward + Update: さらに微調整
```

### Epoch 200（最後）

```
パラメータ: ほぼ最適
         ↓
Forward: ほぼ正解の予測
         ↓
Loss:    約 0.11（ほぼ完璧）
```

---

## 3.8 学習されるもの

本プログラムのパラメータ一覧：

| パラメータ | 形状 | 何を学ぶか |
|-----------|------|-----------|
| `tok_emb` | (10, 64) | 各単語の意味ベクトル |
| `pos_emb` | (12, 64) | 各位置の情報ベクトル |
| `Wq, Wk, Wv` | (64, 64) ×3 ×2層 | Attention の問い合わせ方 |
| `Wo` | (64, 64) ×2層 | Attention 出力の統合方法 |
| `W1, b1` | (64,128), (128,) ×2層 | FFN の変換（前半） |
| `W2, b2` | (128,64), (64,) ×2層 | FFN の変換（後半） |
| `ln*_g, ln*_b` | (64,) 各種 | Layer Norm のスケールとシフト |

これらがすべて、200 回の訓練で **loss を最小にする方向** に
少しずつ調整されていきます。

---

## まとめ

```
Forward（順伝播）:  inputs → 予測(logits)
                          ↓
Loss（損失計算）:    予測と正解の差 → 1つの数値
                          ↓
Backward（逆伝播）:  全パラメータの勾配を自動計算
                          ↓
Update（更新）:      勾配を使ってパラメータを微調整
                          ↓
                    繰り返し → 損失がどんどん小さくなる
```

| 概念 | 意味 |
|------|------|
| 損失（loss） | 予測がどれだけ間違っているかの指標 |
| 勾配（gradient） | パラメータを動かす方向と大きさ |
| 逆伝播（backprop） | 連鎖律で勾配を効率的に計算する手法 |
| 学習率（LR） | 1回の更新でどれだけ動かすか |
| エポック（epoch） | 訓練データを1周すること |
