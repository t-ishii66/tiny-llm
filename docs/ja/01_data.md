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

>
> **Python Tips: 辞書内包表記 `{i: w for w, i in ...}`**
>
> `{i: w for w, i in vocab.items()}` は **辞書内包表記（dict comprehension）** です。
> for ループで辞書を作る短縮記法です：
> ```python
> # 以下の2つは同じ結果
> id2word = {}
> for w, i in vocab.items():
>     id2word[i] = w
>
> id2word = {i: w for w, i in vocab.items()}  # 1行で書ける
> ```
> `vocab.items()` は `("the", 1), ("cat", 2), ...` のようなペアを返します。
> `for w, i in ...` でそのペアを `w="the", i=1` のように分解しています。

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

>
> **Python Tips: リスト内包表記 `[... for ... in ...]`**
>
> `[vocab[w] for w in text.split()]` は **リスト内包表記（list comprehension）** です。
> for ループでリストを作る短縮記法です：
> ```python
> # 以下の2つは同じ結果
> result = []
> for w in text.split():
>     result.append(vocab[w])
>
> result = [vocab[w] for w in text.split()]  # 1行で書ける
> ```

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

### なぜ `SEQ_LEN` で切り出すのか（= コンテキスト長）

Transformer は、1回の計算で見られるトークン数に上限があります。
この上限が **コンテキスト長（context length / context window）** で、
本プロジェクトでは `SEQ_LEN = 12` がそれに相当します。

そのため学習時は、コーパスを「長さ12の窓」に切り出してモデルへ渡しています。
生成時も同様で、`generate()` 内では常に直近 `SEQ_LEN` トークンだけを文脈として使います。
つまり、トークン列が `SEQ_LEN` を超えて長くなると、より古いトークンは文脈から外れます。

### パディング（padding）との関係

大規模な事前学習では、通常はテキストを句点で厳密に区切らず、
トークン列を連結して固定長チャンク（`SEQ_LEN`）で切り出して学習します。
この運用では長さが最初から揃うため、padding は基本的に不要です。

一方で、SFT など可変長サンプルを同一バッチに詰める学習では、
短い系列を `<pad>` で埋める（padding）ことがあります。
本コードは固定長 `SEQ_LEN` で学習サンプルを作るため、
訓練時には実質的に padding 処理を使いません（`<pad>` は予約トークンとしてのみ定義）。

>
> **Python Tips: スライス `tokens[i : i + SEQ_LEN]`**
>
> リストのスライスは `リスト[開始:終了]` で、開始から終了の **手前まで** を取り出します：
> ```python
> tokens = [1, 2, 3, 4, 5, 6, 7, 8]
> tokens[0:3]   # → [1, 2, 3]      位置0, 1, 2（3は含まない）
> tokens[2:5]   # → [3, 4, 5]      位置2, 3, 4
> tokens[1:4+1] # → [2, 3, 4, 5]   i+1 から i+SEQ_LEN+1 の手前まで
> ```

>
> **Python Tips: `torch.tensor()` — リストをテンソルに変換**
>
> `torch.tensor([[1,2,3], [4,5,6]])` は Python のリストを PyTorch のテンソルに変換します。
> テンソルは「多次元配列」で、GPU演算や自動微分に対応しています：
> ```python
> import torch
> x = torch.tensor([[1, 2], [3, 4]])
> x.shape  # → torch.Size([2, 2])  ← 2行×2列
> ```

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

>
> **スライド幅（ストライド）について**
>
> 本コードではウィンドウを **1単語ずつ** スライドしています（i=0, 1, 2, ...）。
> しかし、スライド幅は本来自由に設定できます：
>
> ```
> ストライド 1:  i=0, 1, 2, 3, ...  → サンプル数 多い、重複 多い
> ストライド 4:  i=0, 4, 8, 12, ... → サンプル数 少ない、重複 少ない
> ストライド 12: i=0, 12, 24, ...   → 重複なし（ウィンドウが隣接）
> ```
>
> - **ストライド 1** はデータを最大限に活用できますが、隣り合うサンプルの
>   大部分が重複します
> - **大きなストライド** は重複が減り計算が速くなりますが、サンプル数が減ります
> - 大規模 LLM の訓練では、コーパスが十分に大きいため
>   ストライドを大きくしても問題になりません
>
> 本プロジェクトではコーパスが小さいので、ストライド 1 で最大限のサンプルを
> 生成しています。

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

> **バッチについて（ここでは簡略化）**
>
> この `(28, 12)` は「28サンプルを1つの塊（1バッチ）」として
> まとめて処理する形です。つまり本プロジェクトでは、
> 説明を簡単にするため **バッチ数を 1** にしています。
>
> 本物の LLM 訓練ではコーパスが非常に大きいため、データを複数バッチに分割し、
> **バッチごとに forward → loss → backward → update** を繰り返します。
> ただし、各バッチ内で「次の単語を予測する」という学習の本質は同じです。

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
