# Step 1: セットアップと実行

まずはコードを動かしてみましょう。

---

## 1.1 必要なもの

- Python 3.8 以上
- PyTorch（CPU 版で十分）

```bash
pip install torch
```

GPU は不要です。tiny-LLM は数秒で訓練が完了します。

---

## 1.2 コードの取得

```bash
git clone https://github.com/t-ishii66/tiny-llm.git
cd tiny-llm
```

ファイルは1つだけです：

```
tiny-llm/
├── tiny_llm.py          ← 全実装（実コード約140行）
├── docs/                ← ドキュメント
│   ├── 01_data.md
│   ├── 02_transformer.md
│   ├── 03_training.md
│   ├── 03a_gradient.md
│   ├── 04_generation.md
│   └── tutorial/        ← このチュートリアル
└── README.md
```

---

## 1.3 実行する

```bash
python tiny_llm.py
```

以下のような出力が表示されます（数値は実行のたびに多少変わります）：

```
epoch   20  loss=1.9469
epoch   40  loss=1.5257
epoch   60  loss=0.8140
epoch   80  loss=0.5469
epoch  100  loss=0.3880
epoch  120  loss=0.3099
epoch  140  loss=0.2568
epoch  160  loss=0.2227
epoch  180  loss=0.1862
epoch  200  loss=0.1147

prompt: "the cat sat on"
output: the cat sat on the mat . the dog sat on the log . the cat saw the dog . the dog saw the

prompt: "the dog saw"
output: the dog saw the cat . the cat sat on the log . the dog sat on the mat . the dog sat
```

---

## 1.4 出力の読み方

### 訓練の経過

```
epoch   20  loss=1.9469    ← まだでたらめな予測
epoch  200  loss=0.1147    ← ほぼ完璧な予測
```

- **epoch**: 訓練データを何周したか
- **loss**: 予測の悪さ（小さいほど良い）。ランダムなら約 2.3、完璧なら 0

### 生成結果

```
prompt: "the cat sat on"
output: the cat sat on the mat . the dog sat on the log ...
```

- **prompt**: モデルに与えた入力テキスト
- **output**: モデルが1単語ずつ予測して生成したテキスト

訓練コーパスに沿った自然な英文が生成されています。

> **注意**: `generate()` に渡す prompt は、学習コーパスに含まれる単語だけで作ってください。
> コーパス外の単語（語彙外トークン）を含むと、現実装では例外になります。

---

## 1.5 何が起きたのか

たった数秒で、以下が実行されました：

1. **データ準備**: 40単語のコーパスを数値に変換し、訓練データを作成
2. **モデル構築**: 約40,000パラメータの Transformer を初期化
3. **訓練**: 200回の学習ループで「次の単語の予測」を学習
4. **生成**: 学習済みモデルでテキストを生成

次のステップでは、この各段階の中身を自分の目で確認していきます。

---

次へ: [Step 2: データを観察する](02_explore_data.md)
