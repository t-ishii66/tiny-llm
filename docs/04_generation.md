# 第4章: テキスト生成 — 次の単語を予測する

訓練が終わりました。モデルの全パラメータは
「次の単語を正しく予測できる」ように調整されています。

この章では、訓練済みモデルを使って **新しいテキストを生成** する仕組みを見ます。
ここに LLM の本質があります。

---

## 4.1 LLM の本質：次の単語の予測

ChatGPT も GPT-4 も、根本的にやっていることは同じです：

> **「これまでの単語列が与えられたとき、次に来る単語を予測する」**

これを繰り返すだけで、文章が生成されます。

```
入力:  "the cat"
予測:  "sat"        ← 次の1単語を予測

入力:  "the cat sat"
予測:  "on"         ← また次の1単語を予測

入力:  "the cat sat on"
予測:  "the"        ← さらに次の1単語を予測

...これを繰り返す
```

---

## 4.2 コード — `generate` 関数

```python
def generate(model, prompt, vocab, id2word, max_tokens=20):
    tokens = tokenize(prompt, vocab)

    for _ in range(max_tokens):
        context = tokens[-SEQ_LEN:]                   # 直近16トークンを取得
        x = torch.tensor([context])                   # (1, T)
        logits = model.forward(x)                     # (1, T, 10)
        next_logit = logits[0, -1, :]                 # 最後の位置のスコア
        next_id = torch.argmax(next_logit).item()     # 最もスコアの高い単語
        tokens.append(next_id)

    return " ".join(id2word[t] for t in tokens)
```

### ステップごとの説明

**Step 1: プロンプトをトークン化**

```python
tokens = tokenize("the cat sat on", vocab)
# → [1, 2, 3, 4]
```

**Step 2: Transformer に通す**

```python
context = tokens[-SEQ_LEN:]    # [1, 2, 3, 4]  ← 直近16トークン（今は4つだけ）
x = torch.tensor([context])    # shape: (1, 4)
logits = model.forward(x)      # shape: (1, 4, 10)
```

4つの位置すべてで予測が出ますが、必要なのは **最後の位置** だけです。
（最後の位置＝「ここまでの文脈を全て見た上での予測」）

**Step 3: 次の単語を選ぶ**

```python
next_logit = logits[0, -1, :]   # (10,) ← 最後の位置のスコア
# 例: [0.1, 0.3, -0.2, 0.1, 0.8, 2.1, -0.3, -0.1, 0.4, 0.0]
#      pad   the   cat  sat   on   mat    .   dog   log  saw

next_id = torch.argmax(next_logit).item()   # → 5 (= "mat")
```

`argmax` は最もスコアの高いインデックスを返します。
→ "the cat sat on" の次は "mat" と予測。

**Step 4: トークン列に追加して繰り返す**

```python
tokens.append(5)
# tokens = [1, 2, 3, 4, 5]  ← "the cat sat on mat"
# → 次のループで "the cat sat on mat" から次の単語を予測
```

### 生成の流れ（具体例）

```
"the cat sat on"
                  → 予測: "the"  → "the cat sat on the"
                  → 予測: "mat"  → "the cat sat on the mat"
                  → 予測: "."    → "the cat sat on the mat ."
                  → 予測: "the"  → "the cat sat on the mat . the"
                  → 予測: "dog"  → "the cat sat on the mat . the dog"
                  ...
```

---

## 4.3 Greedy Decoding の限界

本プログラムでは `argmax`（最もスコアの高い単語を選ぶ）を使っています。
これを **Greedy Decoding** と呼びます。

```
スコア: [0.1, 0.3, -0.2, 0.1, 0.8, 2.1, -0.3, -0.1, 0.4, 0.0]
                                          ↑
                                     常にこれを選ぶ
```

シンプルですが、常に「一番確率の高い単語」しか選ばないため、
同じパターンをループしやすいという欠点があります。

> **本物の LLM** では、確率分布からサンプリングしたり（temperature）、
> 上位 k 個から選んだり（top-k）する手法で多様性を出しています。

---

## 4.4 実行結果を見る

```
prompt: "the cat sat on"
output: the cat sat on the mat . the dog sat on the log .
        the cat saw the dog . the dog saw the

prompt: "the dog saw"
output: the dog saw the cat . the cat sat on the log .
        the dog sat on the mat . the dog sat
```

訓練コーパスに沿った自然な文が生成されています。
これはコーパスを丸暗記しているだけのように見えますが——
実際その通りです。たった 40 単語・10 語彙では、丸暗記が最適解です。

---

## 4.5 ここから本物の LLM へ

ここまでの tiny-LLM と、GPT-4 のような本物の LLM の違いは
**本質的な仕組みの違いではなく、スケールの違い** です。

| | tiny-LLM | GPT-4 クラス |
|---|---|---|
| 語彙数 | 10 | 100,000+ |
| 埋め込み次元 | 64 | 12,288+ |
| Attention ヘッド数 | 4 | 96+ |
| Transformer 層数 | 2 | 96+ |
| パラメータ数 | 約 40,000 | 数千億〜数兆 |
| 訓練データ | 40 単語 | 数兆トークン |
| 訓練時間 | 数秒 | 数ヶ月（数千GPU） |

しかし、核となるアルゴリズムは **まったく同じ** です：

1. 単語をベクトルに変換する（Embedding）
2. Self-Attention で文脈を理解する（Q, K, V）
3. Feed-Forward で表現を変換する
4. 「次の単語」を予測する損失で訓練する
5. 訓練済みモデルで1単語ずつ生成する

スケールを大きくすると、丸暗記ではなく **汎化** が起き始めます。
「見たことのない文章」でも、学習したパターンから適切な次の単語を
予測できるようになる——これが大規模言語モデルの力です。

---

## まとめ

```
"the cat sat on"
       ↓
   Transformer（訓練済み）
       ↓
   "mat" を予測
       ↓
"the cat sat on the mat"
       ↓
   Transformer
       ↓
   "." を予測
       ↓
   ...繰り返し
```

**LLM の全ては「次の単語を予測する」に集約されます。**

- Embedding は単語に意味を与え
- Self-Attention は文脈を理解し
- 訓練は予測の精度を高め
- 生成は予測を繰り返す

tiny-LLM は小さなおもちゃですが、
Transformer の本質——Self-Attention、Q/K/V、残差接続、
Layer Norm、そして「次の単語の予測」——は本物とまったく同じです。
