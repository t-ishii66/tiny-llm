# 第1章: データの準備 — 文章をモデルが扱える形にする

LLM が扱うのは数値だけです。
「英語の文章」を「数値の列」に変換し、さらに「入力と正解のペア」を作る——
この章ではその全過程を、具体例とともに追いかけます。

---

## 1.1 コーパス（学習テキスト）

まず出発点となるテキストを用意します。

```python
corpus = (
    "the cat sat on the mat . the dog sat on the log . "
    "the cat saw the dog . the dog saw the cat . "
    "the cat sat on the log . the dog sat on the mat ."
)
```

たった 40 単語ですが、Transformer の構造を学ぶには十分です。

---

## 1.2 `build_vocab` — 単語に番号を振る

```python
def build_vocab(text):
    words = text.split()
    vocab = {"<pad>": 0}
    for w in words:
        if w not in vocab:
            vocab[w] = len(vocab)
    return vocab, {i: w for w, i in vocab.items()}
```

### 何をしているか

1. テキストを空白で分割して単語のリストにする
2. 出現順に、重複なく番号を振る（`<pad>` は予約語で 0 番）
3. 「単語→番号」の辞書と「番号→単語」の逆引き辞書を返す

### 具体例

```
引数:  "the cat sat on the mat . the dog sat on the log ..."

戻り値 vocab:
  {"<pad>": 0, "the": 1, "cat": 2, "sat": 3, "on": 4,
   "mat": 5, ".": 6, "dog": 7, "log": 8, "saw": 9}

戻り値 id2word:
  {0: "<pad>", 1: "the", 2: "cat", 3: "sat", 4: "on",
   5: "mat", 6: ".", 7: "dog", 8: "log", 9: "saw"}
```

語彙数は **10**。本物の LLM では数万〜数十万ですが、仕組みは同じです。

> **数値の読み方:** 本プロジェクトでは、すべての主要な数値が異なる値に
> なるよう設計しています。テンソルの形状に現れる数値を見れば、
> それが何を意味しているか即座にわかります：
>
> | 数値 | 意味 |
> |------|------|
> | **2** | Transformer の層数（N_LAYERS） |
> | **4** | Attention ヘッド数（N_HEADS） |
> | **10** | 語彙数（vocab_size） |
> | **12** | シーケンス長（SEQ_LEN） |
> | **16** | ヘッドの次元数（head_dim = 64 ÷ 4） |
> | **28** | 訓練サンプル数（40トークン − 12） |
> | **64** | 埋め込み次元（D_MODEL） |
> | **128** | FFN の隠れ層次元（D_FF） |
>
> 例えばテンソル形状 `(28, 12, 64)` を見たら、
> 「28サンプル × 12トークン × 64次元の埋め込み」と即座に読めます。

> **本物との違い:** GPT などは BPE（Byte Pair Encoding）で単語をさらに
> サブワードに分割します。ここでは「トークン＝単語」と簡略化しています。

---

## 1.3 `tokenize` — 文章を数値の列に変換する

```python
def tokenize(text, vocab):
    return [vocab[w] for w in text.split()]
```

### 何をしているか

文章を空白で分割し、各単語を `vocab` で番号に置き換えます。

### 具体例

```
引数:  text  = "the cat sat on the mat"
       vocab = {"<pad>": 0, "the": 1, "cat": 2, "sat": 3, "on": 4, "mat": 5, ...}

戻り値: [1, 2, 3, 4, 1, 5]
```

```
"the"  →  1
"cat"  →  2
"sat"  →  3
"on"   →  4
"the"  →  1   ← 同じ単語は同じ番号
"mat"  →  5
```

この数値のリスト（**トークン列**）が、以降すべての処理の基本単位になります。

---

## 1.4 `make_training_data` — 「入力」と「正解」のペアを作る

ここが最も重要です。LLM の学習は本質的に **「直前の単語列から、次の単語を予測する」** タスクです。

```python
def make_training_data(text, vocab):
    tokens = tokenize(text, vocab)
    inputs, targets = [], []
    for i in range(len(tokens) - SEQ_LEN):
        inputs.append(tokens[i : i + SEQ_LEN])
        targets.append(tokens[i + 1 : i + SEQ_LEN + 1])
    return torch.tensor(inputs), torch.tensor(targets)
```

### 何をしているか

1. コーパス全体をトークン列に変換する
2. 長さ `SEQ_LEN`（=12）のウィンドウをスライドさせる
3. 各ウィンドウで **入力（input）** と **正解（target）** のペアを作る

### 具体例（SEQ_LEN=12 の場合）

コーパスのトークン列（全40トークン）:

```
位置:  0   1   2   3   4   5   6   7   8   9  10  11  12  13 ...
単語: the cat sat  on the mat  .  the dog sat  on the log  .  ...
番号:  1   2   3   4   1   5   6   1   7   3   4   1   8   6 ...
```

**スライドウィンドウの仕組み:**

```
i=0:
  input  = tokens[ 0:12] = [1,2,3,4,1,5,6,1,7,3,4,1]
  target = tokens[ 1:13] = [2,3,4,1,5,6,1,7,3,4,1,8]
                              ↑ input の各位置の「次の単語」が target

i=1:
  input  = tokens[ 1:13] = [2,3,4,1,5,6,1,7,3,4,1,8]
  target = tokens[ 2:14] = [3,4,1,5,6,1,7,3,4,1,8,6]

i=2:
  input  = tokens[ 2:14] = [3,4,1,5,6,1,7,3,4,1,8,6]
  target = tokens[ 3:15] = [4,1,5,6,1,7,3,4,1,8,6,1]

  ...（合計 28 サンプル）
```

### target は input を1つずらしたもの

これを図にすると、本質が見えます：

```
input:   the  cat  sat  on   the  mat   .   the  ...
target:  cat  sat  on   the  mat   .   the  dog  ...
         ↑    ↑    ↑    ↑    ↑    ↑    ↑    ↑
         各位置で「次に来るべき単語」を予測させる
```

位置 0 では「the の次は cat」、位置 1 では「cat の次は sat」…
**すべての位置で同時に「次の単語」を予測する** のがポイントです。

### 戻り値の形状

```
inputs:  torch.Tensor, shape = (28, 12)    ← 28サンプル × 12トークン
targets: torch.Tensor, shape = (28, 12)    ← 対応する正解
```

---

## 1.5 データの流れ — 全体像

```
                        build_vocab
"the cat sat on ..."  ─────────────→  vocab:   {"the": 1, "cat": 2, ...}
                                       id2word: {1: "the", 2: "cat", ...}

                        tokenize
"the cat sat on ..."  ─────────────→  [1, 2, 3, 4, 1, 5, 6, ...]

                      make_training_data
[1, 2, 3, 4, ...]    ─────────────→  inputs:  tensor (28, 12)
                                       targets: tensor (28, 12)
```

ここまでで、モデルに渡す準備が整いました。

**次の章（第2章）** では、この `inputs` テンソルが Transformer の中で
どのように処理されるかを見ていきます。

---

## まとめ

| 関数 | 入力 | 出力 | 役割 |
|------|------|------|------|
| `build_vocab(text)` | 文字列 | `vocab` (dict), `id2word` (dict) | 単語に番号を振る |
| `tokenize(text, vocab)` | 文字列, vocab | `[int, ...]` (list) | 文章を番号列に変換 |
| `make_training_data(text, vocab)` | 文字列, vocab | `inputs` (28,12), `targets` (28,12) | 学習用ペアを作成 |
