# Step 5: インストラクションチューニングを試す

第5章 (本編) で見た Alpaca 形式の Instruction Tuning を、実際に動かして体感します。
事前学習済みモデルが、4 つの指示×応答ペアを学んで「指示に応答する」挙動を獲得するまでを観察します。
ただし結果は「丸暗記」であり、実際の LLM のような汎化はないことも自分の目で確認します。

> Step 5 から扱うコードファイルが変わります。`tiny_llm.py` ではなく `tiny_llm_instruct.py` を実行します
> (中身は `tiny_llm.py` を import して、instruction tuning 用の処理を足した別ファイル)。

---

## 5.1 実行する

```bash
uv run --with torch tiny_llm_instruct.py
```

3 つの Stage が順に走ります。

```
--- Stage 1: Pretraining ---        ← 通常の言語モデル訓練 (200 epoch)
epoch   20  loss=...
...
epoch  200  loss=...

--- Stage 2: Instruction tuning --- ← Response マスク付き fine-tune (300 epoch)
epoch   30  loss=...
...
epoch  300  loss=...

--- Stage 3: Responding ---         ← 訓練後の応答生成
### Instruction: who sat on the mat
### Response: (5.2 で確認します — まだ見ないで)

### Instruction: who saw the dog
### Response: (5.2 で確認します — まだ見ないで)

### Instruction: who sat on the log
### Response: (5.2 で確認します — まだ見ないで)
```

Stage 1 と Stage 2 で loss がしっかり下がっていることを確認してください。
Stage 1 と Stage 2 の loss はそれぞれ別の損失関数 (全位置 / 応答位置のみ) なので、絶対値の比較は意味がありません。**それぞれが下がりきっているか** だけ見れば OK です。

(Stage 3 の実際の応答は、まず 5.2 で予測してから答え合わせします。)

---

## 5.2 出力を予測してから確認する

`tiny_llm_instruct.py` の `examples` リストを見ると、訓練データは以下の 4 例だけ：

```python
examples = [
    ("who sat on the mat", "", "the cat"),
    ("who sat on the log", "", "the dog"),
    ("who saw the dog",    "", "the cat"),
    ("who saw the cat",    "", "the dog"),
]
```

Stage 3 で 3 つの指示についてモデルが応答を出します。**実行前に自分で予測してから**、出力と照合してみましょう。

| 指示 | あなたの予測 | 実際の出力 |
|---|---|---|
| `who sat on the mat` | ? | ? |
| `who saw the dog` | ? | ? |
| `who sat on the log` | ? | ? |

訓練で完全に丸暗記しているので、3 つとも訓練データどおりの応答が返ってくるはずです。

---

## 5.3 新しい指示例を追加してみる

`tiny_llm_instruct.py` の `examples` リストに 1 例追加してみます：

```python
examples = [
    ("who sat on the mat", "", "the cat"),
    ("who sat on the log", "", "the dog"),
    ("who saw the dog",    "", "the cat"),
    ("who saw the cat",    "", "the dog"),
    ("who saw the mat",    "", "the dog"),   # ← 追加
]
```

Stage 3 のテスト指示にも対応するものを追加：

```python
for ins in [
    "who sat on the mat",
    "who saw the dog",
    "who sat on the log",
    "who saw the mat",    # ← 追加
]:
```

再実行：

```bash
uv run --with torch tiny_llm_instruct.py
```

5 例目も丸暗記されて、`who saw the mat` に対して `the dog` が返ってくれば成功です。
たった 1 例増やすだけで、モデルが新しい指示×応答パターンを覚えられることを確認できました。

> Step 1.4 でも触れたとおり、追加する指示・応答に **コーパスに無い単語** を入れると `KeyError` になります。
> 既存の語彙 (`the`, `cat`, `sat`, `on`, `mat`, `.`, `dog`, `log`, `saw`, `who`) の範囲で組み合わせてください。

---

## 5.4 OOD 指示を試す — 丸暗記の限界を見る

訓練データに無い指示を試して、モデルがどう振る舞うか見てみます。

5.3 で追加した行は **元に戻し**、`examples` を元の 4 行に戻してから、Stage 3 のテスト指示だけを変えます：

```python
for ins in [
    "who sat on the cat",    # ← 訓練に無い指示 (語彙は全部既知)
    "who saw the log",       # ← 訓練に無い指示
    "who sat on the dog",    # ← 訓練に無い指示
]:
```

再実行して、どんな応答が出るか観察します。よく見られるパターン：

- 既存の訓練例の応答を **オウム返し** する (例: 全部 `the cat` になる)
- 最後に学習した訓練例の応答に強く引きずられる
- 意味の合わない単語列が出る

訓練データの 4 例しか学んでいないので、未知の指示に対して **意味のある汎化はできません**。これは tiny-LLM のサイズと訓練量の限界であり、 **「丸暗記モデルである」というメッセージを体感** できる瞬間です。

---

## 5.5 Instruction tuning の epoch 数を減らしてみる

最後に「どれくらい丸暗記すれば応答ができるようになるか」を観察します。

`if __name__ == "__main__":` ブロックの Stage 2 で `train_instruct` を呼んでいる行を変えます：

```python
# --- Stage 2: Instruction tuning ---
print("\n--- Stage 2: Instruction tuning ---")
examples = [...]
train_instruct(model, examples, vocab, epochs=30)   # ← 30, 50, 100 などに変える
```

| epochs | 期待される挙動 |
|---|---|
| 300 (元) | 全 3 例とも正しい応答 (丸暗記完了) |
| 100 | だいたい正しいが、たまに崩れる |
| 50 | 半分くらいしか覚えていない |
| 10 | ほとんど学習できておらず、出力が崩れる |

epoch 数が少ないと「format は出すが応答内容が変」という中間状態が観察できます。
Loss の収束カーブも見ながら、instruction tuning がどのくらいの計算量で完了するかの感覚を掴んでください。

---

## 5.6 (Optional) 同じ語彙で別タスクを設計してみる

既存の語彙 (`the cat sat on mat . dog log saw who`) だけを組み合わせて、自分で `(指示, 応答)` ペアを考えてみる課題です。

例：

```python
examples = [
    ("what did the cat see", "", "the dog"),
    ("what did the dog see", "", "the cat"),
    ("where did the cat sit", "", "on the mat"),
    ("where did the dog sit", "", "on the log"),
]
```

ただし上の例では `what`, `did`, `where` という未登場の単語が含まれるため、そのままでは `KeyError` になります。これを通すには、コーパス末尾にダミー出現として「`what did where`」を足して語彙に登録しておくか、語彙内の単語だけで指示を組むかの選択になります。

> 「**訓練データを自分で設計する**」という視点を一度持つと、本物の instruction tuning で
> 「データセットの質と多様性が結果を決める」と言われる意味が体感できます。

時間に余裕があれば、独自の examples リストを作って訓練して、どの程度丸暗記できるか試してみてください。

---

## まとめ — チュートリアル全体の振り返り

5 つの Step を通じて、以下を **手を動かして** 体験しました：

1. **Step 1**: `uv run` 一発で訓練と生成が走るところまで確認した
2. **Step 2**: トークン化・訓練データのスライドウィンドウ構造をテンソルで観察した
3. **Step 3**: Embedding と Attention 重みを取り出して、モデルが「数値テンソル」で動いていることを確認した
4. **Step 4**: ハイパーパラメータや構造 (Weight Tying、Temperature) を変えて、loss と生成への影響を測った
5. **Step 5**: pretrain → instruction tuning → 応答の 3 段階を回し、Response マスキングが「指示に従う」挙動を作ることを見た。同時に「丸暗記の限界」も体感した

結果は丸暗記でしたが、手順そのもの (「format で区切る + 応答だけ loss」) を自分の手で組み立てたことが本チュートリアルの収穫です。

これで tiny-LLM チュートリアルは終わりです。
本編のドキュメントに戻り、各章を読み返すと、ここで動かした挙動の裏側にある数学・アーキテクチャがより深く理解できるはずです。

- [ドキュメント先頭](../01_data.md)
- [第5章本編 (Instruction Tuning)](../05_instruction_tuning.md)
