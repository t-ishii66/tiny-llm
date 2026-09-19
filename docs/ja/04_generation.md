# 第4章: テキスト生成 — 次の単語を予測する

![テキスト生成](../images/chapter-04-generation.png)

訓練が終わりました。モデルの全パラメータは
「次の単語を正しく予測できる」ように調整されています。

この章では、訓練済みモデルを使って **新しいテキストを生成** する仕組みを見ます。
ここに LLM の本質があります。

---

## 4.1 LLM の本質：次の単語の予測

ChatGPT も GPT-4 も、基本的にやっていることは同じです：

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

    with torch.no_grad():                                # 推論時は勾配計算不要
        for _ in range(max_tokens):
            context = tokens[-SEQ_LEN:]                  # 直近12トークンを取得
            x = torch.tensor([context])                  # (1, T)
            logits = model.forward(x)                    # (1, T, 10)
            next_logit = logits[0, -1, :]                # 最後の位置のスコア
            next_id = torch.argmax(next_logit).item()    # 最もスコアの高い単語
            tokens.append(next_id)

    return " ".join(id2word[t] for t in tokens)
```

>
> **Python Tips: `" ".join(...)` — リストを文字列に結合**
>
> `" ".join(リスト)` はリストの要素をスペースで繋いで1つの文字列にします：
> ```python
> words = ["the", "cat", "sat"]
> " ".join(words)     # → "the cat sat"
> "-".join(words)     # → "the-cat-sat"
> ```
> `id2word[t] for t in tokens` は **ジェネレータ式** で、
> 各トークン番号を単語に変換しながら `join` に渡しています。

### ステップごとの説明

**Step 1: プロンプトをトークン化**

```python
tokens = tokenize("the cat sat on", vocab)
# → [1, 2, 3, 4]
```

**Step 2: Transformer に通す**

```python
context = tokens[-SEQ_LEN:]    # [1, 2, 3, 4]  ← 直近12トークン（今は4つだけ）
x = torch.tensor([context])    # shape: (1, 4)
logits = model.forward(x)      # shape: (1, 4, 10)
```

形状の先頭にある **1** は、バッチ（セット）の数です。
訓練では 28 セットをまとめて流していましたが、生成で入力するのは
いま手元にある文脈 1 本だけなので、バッチの大きさは 1 になります。
`torch.tensor([context])` と `context` をリストで 1 段包んでいるのは、
この 1 セット分の軸を作るためです。

4つの位置すべてで予測が出ますが、必要なのは **最後の位置** だけです。
（最後の位置＝「ここまでの文脈を全て見た上での予測」）

**Step 3: 次の単語を選ぶ**

```python
next_logit = logits[0, -1, :]   # (10,) ← 最後の位置のスコア
# 例: [0.1, 2.8, -0.2, 0.1, 0.8, 0.3, -0.3, -0.1, 0.4, 0.0]
#      pad   the   cat  sat   on   mat    .   dog   log  saw

next_id = torch.argmax(next_logit).item()   # → 1 (= "the")
```

>
> **Python Tips: テンソルの多次元インデックス `logits[0, -1, :]`**
>
> カンマ区切りで各軸の位置を指定します。`-1` は「最後」、`:` は「全部」：
> ```python
> x = torch.zeros(3, 4, 10)   # 3サンプル × 4位置 × 10単語
>
> x[0]         # → shape: (4, 10)   最初のサンプル全体
> x[0, -1]     # → shape: (10,)     最初のサンプルの最後の位置
> x[0, -1, :]  # → shape: (10,)     同上（: は「全部」なので省略可）
> x[0, -1, 3]  # → スカラー         特定の1要素
> ```

>
> **Python Tips: `torch.argmax()` — 最大値のインデックス**
>
> テンソルの中で最も大きい値の **位置（インデックス）** を返します：
> ```python
> scores = torch.tensor([0.1, 0.3, 2.1, -0.5, 0.8])
> torch.argmax(scores)          # → tensor(2)   ← 2.1 が最大、その位置は 2
> torch.argmax(scores).item()   # → 2            ← .item() でPythonのintに
> ```

`logits[0, -1, :]` の **0** は「唯一のセット」を指します
（訓練時の 28 セットと違い、ここには 1 本しか入っていません）。
**-1** が最後の位置、**:** が語彙 10 単語分のスコアです。

`argmax` は最もスコアの高いインデックスを返します。
→ "the cat sat on" の次は "the" と予測（コーパスでは "on" の後は常に "the"）。

**Step 4: 予測した単語を入力に加えて、もう一度**

ここが生成の肝です。Step 3 で予測した `"the"`（単語番号 1）を、
**入力だったトークン列の末尾にそのまま足します**。

```python
tokens.append(1)
# 追加前: [1, 2, 3, 4]      ← "the cat sat on"       （元のプロンプト）
# 追加後: [1, 2, 3, 4, 1]   ← "the cat sat on the"   （予測した the が増えた）
```

すると次のループでは、この 5 単語が新しい入力になります。
モデルから見れば、さっき自分が出した答えが、今度は「読むべき文脈」として戻ってくる形です。
5 単語を読んで 6 単語目を予測し、それをまた末尾に足して……と繰り返すことで、
文章がどんどん伸びていきます。

第2章の最後で触れた「予測した単語を入力の末尾に足せば、もう一度同じことができる」を、
`tokens.append()` の 1 行で実現しているわけです。

### 生成の流れ（具体例）

入力は 1 ステップごとに 1 単語ずつ伸びていきます。
前のステップの入力に、予測した単語が足されたものが次の入力です。

| ステップ | 入力（モデルに渡すトークン列） | 長さ | 予測 |
|---|---|---|---|
| 1 | `the cat sat on` | 4 | `the` |
| 2 | `the cat sat on the` | 5 | `mat` |
| 3 | `the cat sat on the mat` | 6 | `.` |
| 4 | `the cat sat on the mat .` | 7 | `the` |
| 5 | `the cat sat on the mat . the` | 8 | `dog` |
| … | （1 ステップごとに 1 単語ずつ伸びる） | … | … |

長さが `SEQ_LEN = 12` を超えると、`tokens[-SEQ_LEN:]` によって
**古いほうから文脈の外に落ちて**いき、常に直近 12 単語だけがモデルに渡されます
（第1章のコンテキスト長の話です）。

---

![木陰で絵本を読みながら話すAliceとBob](../images/chapter-04-break.png)

## 4.3 Greedy Decoding の限界

本プログラムでは `argmax`（最もスコアの高い単語を選ぶ）を使っています。
これを **Greedy Decoding** と呼びます。

```
スコア: [0.1, 0.3, -0.2, 0.1, 0.8, 2.1, -0.3, -0.1, 0.4, 0.0]
```

この例では、最大値の **2.1** を常に選びます。

シンプルですが、常に「一番確率の高い単語」しか選ばないため、
同じパターンをループしやすいという欠点があります。

> このため生成では、確率分布からサンプリングする（temperature）、
> 上位 k 個から選ぶ（top-k）、といった手法で多様性を出すのが一般的です。

### Temperature はどう効くのか

Temperature は、サンプリング前に logits をどれだけ「尖らせる／なだらかにする」かを決める係数です。

$$p_i = \text{softmax}\left(\frac{\text{logit}_i}{T}\right)$$

この $p_i$ を**次単語の確率**として使い、そこから1語を選んで次の単語を決めます
（temperature sampling の場合）。

- `T < 1.0`: 分布が尖る（高スコア語に集中）→ 出力はより決定的
- `T = 1.0`: モデル本来の分布をそのまま使う
- `T > 1.0`: 分布が平らになる（低スコア語にも確率が回る）→ 出力はより多様

直感的には、`T` を下げると「慎重」、上げると「冒険」になります。
`T` が極端に小さいと `argmax` に近づき、極端に大きいとほぼランダムに近づきます。

※ 本ファイルの `generate()` 実装は Greedy (`argmax`) のままで、temperature sampling は実装していません。

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

## 4.5 スケールの違い

tiny-LLM と GPT-4 クラスのモデルを並べてみます。

| | tiny-LLM | GPT-4 クラス |
|---|---|---|
| 語彙数 | 10 | 100,000+ |
| 埋め込み次元 | 64 | 12,288+ |
| Attention ヘッド数 | 4 | 96+ |
| Transformer 層数 | 2 | 96+ |
| パラメータ数 | 約 68,000 | 数千億〜数兆 |
| 訓練データ | 40 トークン | 数兆トークン |
| 訓練時間 | 数秒 | 数ヶ月（数千GPU） |

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
   "the" を予測
       ↓
"the cat sat on the"
       ↓
   Transformer
       ↓
   "mat" を予測
       ↓
   ...繰り返し
```

**LLM の全ては「次の単語を予測する」に集約されます。**

- Embedding は単語に意味を与え
- Self-Attention は文脈を理解し
- 訓練は予測の精度を高め
- 生成は予測を繰り返す

tiny-LLM は小さなおもちゃですが、
ここで実装した Transformer の核——Self-Attention、Q/K/V、残差接続、
Layer Norm、そして「次の単語の予測」——を、自分の手で組み立てたものです。
