# Step 4: 改造してみる

ここまでで Transformer の仕組みが見えてきました。
最後に、コードを改造して実験してみましょう。

---

## 4.1 コーパスを変えてみる

`tiny_llm.py` の `if __name__ == "__main__":` ブロックにあるコーパスを変えてみましょう：

```python
# 元のコーパス
corpus = (
    "the cat sat on the mat . the dog sat on the log . "
    "the cat saw the dog . the dog saw the cat . "
    "the cat sat on the log . the dog sat on the mat ."
)
```

例えば、新しいパターンを追加してみます：

```python
corpus = (
    "the cat sat on the mat . the dog sat on the log . "
    "the cat saw the dog . the dog saw the cat . "
    "the cat sat on the log . the dog sat on the mat . "
    "the bird sat on the log . the bird saw the cat ."
)
```

実行して、"bird" が正しく学習されるか確認してみましょう：

```bash
python tiny_llm.py
```

> **注意**: 新しい単語を追加すると語彙数が変わります（10 → 11）。
> コード自体は語彙数を自動検出するので、そのまま動きます。

---

## 4.2 ハイパーパラメータを変えてみる

`tiny_llm.py` 冒頭のハイパーパラメータを変えて、訓練結果がどう変わるか観察します。

### 実験 1: Attention ヘッド数を変える

```python
N_HEADS = 1    # 1ヘッドだけ（マルチヘッドなし）
```

1ヘッドでも学習できますが、Attention パターンが1種類に制限されます。

### 実験 2: 層数を変える

```python
N_LAYERS = 1   # 1層だけ
```

1層でもこの小さなコーパスなら学習できるでしょう。
ただし、より複雑なパターンには深い層が必要です。

### 実験 3: 埋め込み次元を変える

```python
D_MODEL = 16   # 64 → 16 に縮小
D_FF = 32      # 通常 D_MODEL の2〜4倍
```

次元が小さいと、単語の意味を十分に表現できなくなるかもしれません。
loss の収束が遅くなるか、最終的な loss が高くなるか観察してください。

### 実験 4: 学習率を変える

```python
LR = 0.01     # 10倍に（学習が速いが不安定かも）
LR = 0.0001   # 1/10に（安定だが学習が遅い）
```

### 実験 5: エポック数を変える

```python
EPOCHS = 50    # 少なすぎると学習不足
EPOCHS = 1000  # 多すぎても、この小さなコーパスでは過学習するだけ
```

---

## 4.3 生成方法を変えてみる

### Temperature Sampling

`generate()` 関数では `argmax`（常に最高スコア）で次の単語を選んでいます。
これを確率的なサンプリングに変えてみましょう：

> **注意**: ここでも prompt は学習コーパス内の単語で構成してください。
> 語彙外単語を含む prompt は現実装では例外になります。

```python
def generate(model, prompt, vocab, id2word, max_tokens=20, temperature=1.0):
    tokens = tokenize(prompt, vocab)

    with torch.no_grad():
        for _ in range(max_tokens):
            context = tokens[-SEQ_LEN:]
            x = torch.tensor([context])
            logits = model.forward(x)
            next_logit = logits[0, -1, :] / temperature    # ← temperature で割る

            probs = torch.softmax(next_logit, dim=-1)      # 確率に変換
            next_id = torch.multinomial(probs, 1).item()    # 確率に従ってサンプリング
            tokens.append(next_id)

    return " ".join(id2word[t] for t in tokens)
```

- `temperature = 0.1`: ほぼ argmax と同じ（確信度の高い単語を選ぶ）
- `temperature = 1.0`: モデルの確率分布に忠実にサンプリング
- `temperature = 2.0`: よりランダムに（意外な単語も出やすい）

> このコーパスは非常に小さいので差が出にくいですが、
> 本物の LLM では temperature が生成テキストの多様性を大きく左右します。

---

## 4.4 Weight Tying を外してみる

`tiny_llm.py` の Forward Pass の最後で、`tok_emb` を出力の射影に再利用しています：

```python
logits = x @ self.tok_emb.T    # Weight Tying: Embedding を再利用
```

これを独立の重み行列に変えてみましょう。`__init__` に追加：

```python
self.out_proj = param(D_MODEL, vocab_size)   # (64, 10)
```

Forward の最後を変更：

```python
logits = x @ self.out_proj     # 独立の出力射影
```

`parameters()` にも `self.out_proj` を追加するのを忘れずに。

パラメータ数がどれだけ増えるか、loss の収束に差があるか、比較してみてください。

---

## 4.5 さらなる挑戦

ここまでの実験で Transformer の仕組みが体感できたら、以下にも挑戦してみてください：

- **Layer Norm を外す**: 残差接続だけで学習できるか？
- **因果マスクを外す**: 未来の単語も見える状態で訓練すると何が起きるか？
- **残差接続を外す**: `x = x + attention(x)` を `x = attention(x)` にすると？
- **より大きなコーパス**: 短い英文を増やして、語彙を 30〜50 に拡大

これらの実験を通じて、Transformer の各要素が
**なぜ必要なのか** を実感できるはずです。

---

## まとめ

このチュートリアルでは：

1. **コードを実行** して訓練と生成を確認しました
2. **データの中身** をトークン単位で観察しました
3. **モデルの内部**（Embedding、Attention）を数値で確認しました
4. **コードを改造** してハイパーパラメータやアルゴリズムの影響を実験しました

tiny-LLM は小さなおもちゃですが、GPT などの本物の LLM と
**まったく同じ仕組み** で動いています。
ここで得た理解は、本格的な LLM の学習に直接つながります。

より詳しい仕組みを知りたい方は、[ドキュメント](../01_data.md) へ進んでください。
