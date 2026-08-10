# Step 4: 改造してみる

![改造してみる](../../images/tutorial-04-experiments.png)

ここまでで Transformer の仕組みが見えてきました。コードを改造して実験してみましょう。

> **ここからは作業スタイルが変わります。**
> Step 2〜3 までは `uv run --with torch python -i tiny_llm.py` で対話モードに入り、訓練済みモデルをその場で観察する流れでした。
> Step 4 では **エディタで `tiny_llm.py` を直接書き換え → 保存 → `uv run --with torch tiny_llm.py` で再訓練 → 出力を比較** というサイクルを回します (対話モードは使いません)。
> 各節を試したあとは、変更を元に戻してから次の節に進むと、1 要素ずつの効果が分かりやすくなります。

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
uv run --with torch tiny_llm.py
```

> **注意**: 新しい単語を追加すると語彙数が変わります（10 → 11）。
> コード自体は語彙数を自動検出するので、そのまま動きます。

---

## 4.2 ハイパーパラメータを変えてみる

`tiny_llm.py` 冒頭のハイパーパラメータを変えて、訓練結果がどう変わるか観察します。
**まず「予想」を書いてから実行 → 実測値と並べてみる** と学びの密度が上がります。

| 設定 | 200 epoch 時の最終 loss の予想 | 実測 |
|---|---|---|
| デフォルト (N_HEADS=4, N_LAYERS=2, D_MODEL=64, LR=0.001) | 0.1 前後 | ? |
| N_HEADS=1 | やや高め (0.2〜0.5)? | ? |
| N_LAYERS=1 | デフォルトと近い? | ? |
| D_MODEL=16, D_FF=32 | 表現力不足で下げ止まる? | ? |
| LR=0.01 | 速いが不安定? | ? |
| LR=0.0001 | 遅すぎて下がりきらない? | ? |
| EPOCHS=50 | 学習不足、高めで止まる? | ? |

設定の変更箇所はそれぞれ：

```python
N_HEADS = 1     # 1ヘッドだけ（マルチヘッドなし）
N_LAYERS = 1    # 1層だけ
D_MODEL = 16    # 64 → 16 に縮小、D_FF = 32 もセット
LR = 0.01       # 10倍に
LR = 0.0001     # 1/10に
EPOCHS = 50     # 少なすぎ
EPOCHS = 1000   # 多すぎ（過学習）
```

> **コツ**: 1 つだけ変えて 1 回回す → 値を記録 → デフォルトに戻して次へ、を繰り返すと
> 1 要素の効果が見えます。複数を同時に変えると何が効いたか分かりません。

---

## 4.3 生成方法を変えてみる

### Temperature Sampling

`generate()` 関数では `argmax`（常に最高スコア）で次の単語を選んでいます。
これを確率的なサンプリングに変えてみましょう：

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

> **`torch.multinomial(probs, 1)` とは**: 確率分布 `probs` に従って 1 個サンプリングする関数です。
> たとえば `probs = [0.7, 0.2, 0.1]` なら 70%/20%/10% の確率で 0/1/2 のいずれかを返します。
> `argmax` だと毎回 0 しか出ませんが、`multinomial` を使うとそのときの分布に応じて他の候補も選ばれ得るので、生成にゆらぎが生まれます。

- `temperature = 0.1`: ほぼ argmax と同じ（確信度の高い単語を選ぶ）
- `temperature = 1.0`: モデルの確率分布に忠実にサンプリング
- `temperature = 2.0`: よりランダムに（意外な単語も出やすい）

> このコーパスは非常に小さいので差が出にくいですが、
> 実際の LLM では temperature が生成テキストの多様性を大きく左右します。

---

## 4.4 Weight Tying を外してみる

`tiny_llm.py` の Forward Pass の最後で、`tok_emb` を出力の射影に再利用しています：

```python
logits = x @ self.tok_emb.T    # Weight Tying: Embedding を再利用
```

これを独立の重み行列に変えてみましょう。`TinyTransformer.__init__` の `--- 埋め込み ---` 直後あたりに `out_proj` を新設：

```python
# --- 埋め込み ---
self.tok_emb = param(vocab_size, D_MODEL)
self.pos_emb = param(SEQ_LEN, D_MODEL)
self.out_proj = param(D_MODEL, vocab_size)   # ← 追加 (64, 10)
```

`forward` の最後を `out_proj` を使う形に変更：

```python
logits = x @ self.out_proj     # ← Weight Tying を外して独立の出力射影に
```

最後に `parameters()` で `out_proj` も学習対象に含めます：

```python
def parameters(self):
    params = [self.tok_emb, self.pos_emb, self.out_proj,   # ← out_proj を追加
              self.ln_f_g, self.ln_f_b]
    for layer in self.layers:
        params.extend(layer.values())
    return params
```

実行してみて、

- パラメータ数 (3.1 で見た `sum(p.numel() for p in model.parameters())`) がどれだけ増えるか
- loss の収束カーブがデフォルトとどう変わるか

を比較してみてください。`D_MODEL * vocab_size = 64 * 10 = 640` パラメータ増える計算です。

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

ここまでで「素の言語モデル」を訓練して動かす一巡が終わりました。
次は instruction tuning を試して、「ChatGPT のような指示応答型」の挙動がどう作られるかを見てみましょう。

次へ: [Step 5: インストラクションチューニングを試す](05_instruction.md)
