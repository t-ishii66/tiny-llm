"""tiny-LLM: 学習用の最小 Transformer 実装。

このファイル 1 つで、ゼロから書いた Transformer 言語モデルを完成させる:
  - 単語レベルのトークナイザ (空白分割、サブワードなし)
  - トークン埋め込みと位置埋め込み
  - Causal mask 付き Multi-head Self-Attention
  - 位置ごとの Feed-Forward Network
  - Layer Normalization と残差接続
  - Cross-Entropy 損失による訓練ループ (誤差逆伝播は torch.autograd)
  - Greedy なテキスト生成

順伝播は手書き、逆伝播は PyTorch の autograd に任せている。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==============================================================================
# パラメータ用ヘルパー
# ==============================================================================
# nn.Parameter は PyTorch に「このテンソルは学習対象 — 勾配を追跡せよ」と伝えるラッパー。
# 初期値を小さな乱数 (× 0.02) にすることで、初期出力をゼロ近くに保ち、
# 訓練初期の安定性を確保している。

def param(*shape):
    """学習可能なパラメータを小さな乱数で初期化して返す。"""
    return nn.Parameter(torch.randn(*shape) * 0.02)

def param_ones(*shape):
    """学習可能なパラメータを 1 で初期化して返す (LayerNorm のスケール用)。"""
    return nn.Parameter(torch.ones(*shape))

def param_zeros(*shape):
    """学習可能なパラメータを 0 で初期化して返す (バイアス用)。"""
    return nn.Parameter(torch.zeros(*shape))


# ==============================================================================
# ハイパーパラメータ
# ==============================================================================

D_MODEL = 64       # 埋め込み次元 — 各単語ベクトルのサイズ
N_HEADS = 4        # Attention ヘッド数 (D_MODEL がこれで割り切れる必要あり)
D_FF = 128         # Feed-Forward Network の隠れ次元
N_LAYERS = 2       # 積み重ねる Transformer ブロックの数
SEQ_LEN = 12       # 最大系列長 (コンテキストウィンドウサイズ)
EPOCHS = 200       # データセット全体を走る訓練イテレーション数
LR = 0.001         # Adam オプティマイザの学習率


# ==============================================================================
# トークナイザ — 単語レベル、空白分割
# ==============================================================================
# 実用 LLM はサブワードトークン化 (BPE など) を使う。
# ここでは単純化のため「1 単語 = 1 トークン」とする。

def build_vocab(text):
    """テキストコーパスから word→id と id→word の辞書を作る。

    Args:
        text: 生のテキスト文字列 (例: "the cat sat on the mat")

    Returns:
        vocab:   word → id の dict (例: {"the": 1, "cat": 2, ...})
        id2word: id → word の dict (例: {1: "the", 2: "cat", ...})

    特殊トークン <pad> は常に id 0 として予約される。
    その他の単語は、初登場順に id が割り当てられる。
    """
    words = text.split()
    vocab = {"<pad>": 0}  # id 0 はパディング用に予約
    for w in words:
        if w not in vocab:
            vocab[w] = len(vocab)  # 次に空いている id を割り当て
    return vocab, {i: w for w, i in vocab.items()}


def tokenize(text, vocab):
    """テキスト文字列をトークン id のリストに変換する。

    Args:
        text:  生のテキスト文字列 (例: "the cat sat on")
        vocab: build_vocab() で作った word → id マッピング

    Returns:
        整数のリスト (例: [1, 2, 3, 4])
    """
    return [vocab[w] for w in text.split()]


# ==============================================================================
# モデルコンポーネント — 手書きの順伝播
# ==============================================================================

class TinyTransformer:
    """GPT 型の最小 Transformer 言語モデル。

    アーキテクチャ:
        token_ids → tok_emb + pos_emb → N × TransformerBlock → LayerNorm → logits

    すべてのパラメータは生の nn.Parameter テンソル (nn.Module を使わない)。
    順伝播は各計算ステップを明示的に書き下している。
    """

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

        # --- 埋め込み ---
        # tok_emb: ルックアップテーブル、各行は 1 単語ぶんの学習可能ベクトル
        #   shape: (vocab_size, D_MODEL) = (10, 64)
        self.tok_emb = param(vocab_size, D_MODEL)

        # pos_emb: 系列内の各位置に対応する学習可能ベクトル 1 本
        #   shape: (SEQ_LEN, D_MODEL) = (12, 64)
        self.pos_emb = param(SEQ_LEN, D_MODEL)

        # --- Transformer 層 ---
        # 各層は Multi-head Attention + FFN を含み、それぞれに LayerNorm が付く
        self.layers = []
        for _ in range(N_LAYERS):
            layer = {
                # --- Multi-head Self-Attention の重み ---
                # Wq: 入力を Query に射影  ("何を探しているか?")
                # Wk: 入力を Key に射影    ("自分はどんな情報を持つか?")
                # Wv: 入力を Value に射影  ("どんな内容を提供するか?")
                # Wo: 各ヘッドを連結してから再度 D_MODEL に射影する出力重み
                "Wq": param(D_MODEL, D_MODEL),  # (64, 64)
                "Wk": param(D_MODEL, D_MODEL),
                "Wv": param(D_MODEL, D_MODEL),
                "Wo": param(D_MODEL, D_MODEL),

                # --- Attention 直前の Layer Norm ---
                # g = scale (gamma), b = shift (beta)
                "ln1_g": param_ones(D_MODEL),   # (64,) 初期値 1
                "ln1_b": param_zeros(D_MODEL),  # (64,) 初期値 0

                # --- Feed-Forward Network の重み ---
                # 2 つの線形層: D_MODEL → D_FF → D_MODEL
                # W1, b1: 第 1 層  (64 → 128, 拡張)
                # W2, b2: 第 2 層  (128 → 64, 圧縮)
                "W1": param(D_MODEL, D_FF),     # (64, 128)
                "b1": param_zeros(D_FF),         # (128,)
                "W2": param(D_FF, D_MODEL),     # (128, 64)
                "b2": param_zeros(D_MODEL),      # (64,)

                # --- FFN 直前の Layer Norm ---
                "ln2_g": param_ones(D_MODEL),
                "ln2_b": param_zeros(D_MODEL),
            }
            self.layers.append(layer)

        # --- 最終 Layer Norm (出力射影の直前で適用) ---
        self.ln_f_g = param_ones(D_MODEL)
        self.ln_f_b = param_zeros(D_MODEL)

    def parameters(self):
        """すべての学習可能パラメータをフラットなリストにしてオプティマイザに渡す。"""
        params = [self.tok_emb, self.pos_emb, self.ln_f_g, self.ln_f_b]
        for layer in self.layers:
            params.extend(layer.values())
        return params

    def forward(self, token_ids):
        """順伝播全体を実行: token id → logits。

        Args:
            token_ids: shape (batch, seq_len) の整数テンソル
                       例: tensor([[1, 2, 3, 4, ...]])

        Returns:
            logits: shape (batch, seq_len, vocab_size) の浮動小数テンソル。
                    各位置で、語彙中のすべての単語に対するスコアが入る。
                    高いスコア = モデルが「次にこの単語が来やすい」と判断したもの。
        """
        B, T = token_ids.shape  # B = バッチサイズ、T = 系列長

        # ---- Step 1: 埋め込み ----
        # トークンベクトルを引き、位置ベクトルを足す。
        # tok_emb[token_ids]: (B, T) → (B, T, 64)  — 各 id が 64 次元ベクトルになる
        # pos_emb[:T]:        (T, 64)              — 全サンプルにブロードキャストで加算
        x = self.tok_emb[token_ids] + self.pos_emb[:T]  # (B, T, D_MODEL)

        # ---- Step 2: Transformer ブロック ----
        # N_LAYERS 個の Transformer ブロックを順番に通す。
        # 各ブロックが Attention と FFN で表現を refine していく。
        for layer in self.layers:
            x = transformer_block(x, layer)

        # ---- Step 3: 出力射影 ----
        # 最終 Layer Norm をかけてから、語彙サイズに射影する。
        # Weight tying: tok_emb を出力射影行列としても再利用する。
        # 直感: 「出力ベクトルが単語 X の埋め込みに近ければ、
        #         次に来る単語は X の可能性が高い」
        x = layer_norm(x, self.ln_f_g, self.ln_f_b)
        logits = x @ self.tok_emb.T  # (B, T, 64) @ (64, 10) → (B, T, 10)
        return logits


def layer_norm(x, g, b, eps=1e-5):
    """Layer Normalization: 各ベクトルを mean=0, var=1 に正規化し、スケール+シフトする。

    式: g * (x - mean) / sqrt(var + eps) + b

    多数の層を通過してもベクトルが大きく/小さくなりすぎず、訓練が安定する。

    Args:
        x: shape (..., D_MODEL) の入力テンソル
        g: スケールパラメータ (gamma)、shape (D_MODEL,)
        b: シフトパラメータ (beta)、 shape (D_MODEL,)
        eps: ゼロ除算を防ぐための小さな定数

    Returns:
        正規化されたテンソル、x と同形状
    """
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return g * (x - mean) / torch.sqrt(var + eps) + b


def self_attention(x, Wq, Wk, Wv, Wo):
    """Causal mask 付き Multi-head Self-Attention。

    Transformer の核心機構。各トークンが系列中の他のトークンを「見て」必要な情報を集める。

    手順:
        1. 入力を Query / Key / Value に射影
        2. 複数ヘッドに分割 (各ヘッドが異なる Attention パターンを学習)
        3. Attention スコアを計算: トークン i は j にどれだけ注意を向けるべきか?
        4. Causal mask を適用: 未来のトークンへの注意を禁止
        5. Softmax: スコアを確率に変換
        6. Value の加重和: 注目したトークンから情報を集める
        7. ヘッドを連結して出力射影

    Args:
        x:  入力テンソル、shape (B, T, D_MODEL) — B=バッチ、T=系列長
        Wq: Query 射影重み、shape  (D_MODEL, D_MODEL)
        Wk: Key 射影重み、shape    (D_MODEL, D_MODEL)
        Wv: Value 射影重み、shape  (D_MODEL, D_MODEL)
        Wo: 出力射影重み、shape    (D_MODEL, D_MODEL)

    Returns:
        出力テンソル、shape (B, T, D_MODEL)
    """
    B, T, D = x.shape
    head_dim = D // N_HEADS  # 64 // 4 = 16 次元/ヘッド

    # --- Step 1: Query, Key, Value を計算 ---
    # 各トークンが 3 種類の射影結果を持つ:
    #   Query = "何を探しているか?"
    #   Key   = "自分はどんな情報を持つか?"
    #   Value = "どんな内容を提供するか?"
    Q = x @ Wq  # (B, T, D) @ (D, D) → (B, T, D)
    K = x @ Wk
    V = x @ Wv

    # --- Step 2: 複数ヘッドに分割 ---
    # D_MODEL を (N_HEADS, head_dim) に reshape し、ヘッド軸を 1 軸に持ってくる。
    # (B, T, D) → (B, T, 4, 16) → (B, 4, T, 16)
    # 各ヘッドが独立に 16 次元ベクトルで Attention を計算する。
    Q = Q.view(B, T, N_HEADS, head_dim).transpose(1, 2)
    K = K.view(B, T, N_HEADS, head_dim).transpose(1, 2)
    V = V.view(B, T, N_HEADS, head_dim).transpose(1, 2)

    # --- Step 3: Attention スコアを計算 ---
    # score(i,j) = Q_i と K_j の内積 (類似度の尺度)。
    # sqrt(head_dim) で割って値の範囲を抑える
    # (内積が大きすぎると softmax が極端な値を返すため)。
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(head_dim)  # (B, H, T, T)

    # --- Step 4: Causal mask を適用 ---
    # 言語モデリングでは、位置 i のトークンは未来 (i+1, i+2, ...) を見てはならない。
    # 未来位置を -inf にしておき、softmax 後に確率 0 にする。
    # torch.triu(diagonal=1) は対角の 1 つ上から 1 の上三角行列を作る:
    #   [[0, 1, 1],
    #    [0, 0, 1],   ← 1 = 「この位置は未来、マスクで隠す」
    #    [0, 0, 0]]
    mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
    scores = scores.masked_fill(mask, float("-inf"))

    # --- Step 5: Softmax → Attention 重み ---
    # スコアを確率に変換 (各行の合計が 1.0)。
    # -inf は softmax 後に 0 になるので、未来トークンは実質隠される。
    attn_weights = F.softmax(scores, dim=-1)  # (B, H, T, T)

    # --- Step 6: Value の加重和 ---
    # 各トークンの出力は、見える Value ベクトル群の重み付き和。
    # トークン i が j を強く注視するなら、V_j からより多くを受け取る。
    out = attn_weights @ V  # (B, H, T, head_dim)

    # --- Step 7: ヘッドを連結して出力射影 ---
    # 全ヘッドを統合: (B, 4, T, 16) → (B, T, 4, 16) → (B, T, 64)
    # その後 Wo で射影して、ヘッド間の情報を混ぜる。
    out = out.transpose(1, 2).contiguous().view(B, T, D)  # (B, T, D)
    out = out @ Wo
    return out


def feed_forward(x, W1, b1, W2, b2):
    """ReLU 活性化付きの Position-wise Feed-Forward Network。

    各トークンを独立に 2 つの線形層で変換する:
        x → D_FF に拡張 → ReLU → D_MODEL に圧縮

    Self-Attention がトークン間で情報を混ぜるのに対し、
    FFN は各トークンの表現を 個別に 変換する。

    式: ReLU(x @ W1 + b1) @ W2 + b2

    Args:
        x:  入力テンソル、shape (B, T, D_MODEL)
        W1: 第 1 層重み、 shape (D_MODEL, D_FF) — 拡張: 64 → 128
        b1: 第 1 層バイアス、shape (D_FF,)
        W2: 第 2 層重み、 shape (D_FF, D_MODEL) — 圧縮: 128 → 64
        b2: 第 2 層バイアス、shape (D_MODEL,)

    Returns:
        出力テンソル、shape (B, T, D_MODEL)
    """
    # ReLU(x) = max(0, x) — 非線形性を入れることで、線形変換では学習できない
    # 複雑なパターンを表現できるようになる。
    return F.relu(x @ W1 + b1) @ W2 + b2


def transformer_block(x, layer):
    """Transformer ブロック 1 つ: Self-Attention + FFN、それぞれに残差接続と Layer Norm。

    構造 (Pre-LN、GPT-2 以降のスタイル):
        x → Layer Norm → Self-Attention → 残差加算 → Layer Norm → FFN → 残差加算

    残差接続 (x + sublayer_output) によって勾配が直接流れる経路ができ、
    深いモデルでも訓練可能になる。直感的には「元の情報を保ったまま、新しい情報を上に積む」。

    Args:
        x:     入力テンソル、shape (B, T, D_MODEL)
        layer: このブロックの全パラメータを格納した dict

    Returns:
        出力テンソル、shape (B, T, D_MODEL)
    """
    # サブブロック 1: Layer Norm → Multi-head Self-Attention → 残差
    normed = layer_norm(x, layer["ln1_g"], layer["ln1_b"])
    attn_out = self_attention(normed, layer["Wq"], layer["Wk"], layer["Wv"], layer["Wo"])
    x = x + attn_out

    # サブブロック 2: Layer Norm → Feed-Forward → 残差
    normed = layer_norm(x, layer["ln2_g"], layer["ln2_b"])
    ff_out = feed_forward(normed, layer["W1"], layer["b1"], layer["W2"], layer["b2"])
    x = x + ff_out
    return x


# ==============================================================================
# 訓練
# ==============================================================================

def make_training_data(text, vocab):
    """次単語予測用の入力 / ターゲットペアを作る。

    トークン列を SEQ_LEN サイズの窓でスライドさせる。
    各窓について、ターゲットは入力を 1 位置ずらしたもの:
        input:  [the, cat, sat, on,  the, mat, ...]
        target: [cat, sat, on,  the, mat, .,   ...]
    つまり全位置で「次の単語」を予測することになる。

    Args:
        text:  生のテキストコーパス
        vocab: word → id マッピング

    Returns:
        inputs:  整数テンソル、shape (num_samples, SEQ_LEN) 例: (28, 12)
        targets: 整数テンソル、shape (num_samples, SEQ_LEN) 例: (28, 12)
    """
    tokens = tokenize(text, vocab)  # コーパス全体を id 列に変換

    # トークン列に窓をスライドさせる
    inputs, targets = [], []
    for i in range(len(tokens) - SEQ_LEN):
        inputs.append(tokens[i : i + SEQ_LEN])       # 入力として 12 トークン
        targets.append(tokens[i + 1 : i + SEQ_LEN + 1])  # 1 つずらしたものをターゲット
    return torch.tensor(inputs), torch.tensor(targets)


def train(model, inputs, targets):
    """各位置で次単語を予測するようモデルを訓練する。

    訓練ループ:
        1. 順伝播:  inputs → Transformer → logits (予測スコア)
        2. 損失:    logits を targets と Cross-Entropy で比較
        3. 逆伝播:  損失をすべてのパラメータについて微分 (autograd)
        4. 更新:    損失を減らす方向にパラメータを動かす (Adam)

    Args:
        model:   TinyTransformer インスタンス
        inputs:  整数テンソル、shape (num_samples, SEQ_LEN)
        targets: 整数テンソル、shape (num_samples, SEQ_LEN)
    """
    # Adam オプティマイザ: 勾配履歴に基づいて学習率を
    # パラメータごとに自動調整する、改良された勾配降下法。
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        # --- 1. 順伝播: 予測を計算 ---
        logits = model.forward(inputs)  # (B, T, vocab_size) 例: (28, 12, 10)

        # --- 2. 損失を計算: 予測はどれだけ外れているか? ---
        # Cross-Entropy 損失 = -log(正解単語に割り当てた確率)。
        # 損失が低い = モデルが正しい答えに自信を持っている。
        # 損失関数のために (B, T) を 1 次元にフラット化する。
        loss = F.cross_entropy(
            logits.view(-1, model.vocab_size),  # (B*T, vocab_size) = (336, 10)
            targets.view(-1),                    # (B*T,) = (336,)
        )

        # --- 3. 逆伝播: 勾配を計算 ---
        optimizer.zero_grad()  # 前回の勾配をクリア
        loss.backward()        # 逆伝播: 全パラメータの d(loss)/d(param) を計算

        # --- 4. パラメータを更新 ---
        # 各パラメータが損失を減らす方向に少しずつ動く:
        #   param = param - learning_rate * gradient
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"epoch {epoch+1:4d}  loss={loss.item():.4f}")


# ==============================================================================
# テキスト生成
# ==============================================================================

def generate(model, prompt, vocab, id2word, max_tokens=20):
    """次の単語を繰り返し予測して文章を生成する。

    これがすべての LLM の核となる発想: 単語列が与えられたら、
    次に来る可能性が最も高い単語を予測し、それを末尾に追加して繰り返す。

    ここでは Greedy デコーディング (最高スコアの単語を常に選ぶ) を使う。
    本物の LLM では多様性のために temperature, top-k, top-p などの
    サンプリング手法を使う。

    Args:
        model:      訓練済みの TinyTransformer インスタンス
        prompt:     開始テキスト (例: "the cat sat on")
        vocab:      word → id マッピング
        id2word:    id → word マッピング
        max_tokens: 生成する新規トークン数

    Returns:
        生成された文字列 (prompt + 生成トークン)
    """
    tokens = tokenize(prompt, vocab)  # prompt を id 列に変換

    with torch.no_grad():  # 生成中は勾配追跡不要
        for _ in range(max_tokens):
            # 直近 SEQ_LEN トークンを文脈として取る (モデルの窓サイズ)
            context = tokens[-SEQ_LEN:]
            x = torch.tensor([context])           # (1, T) — バッチ 1

            # 順伝播: 各位置における全単語のスコアを取得
            logits = model.forward(x)             # (1, T, vocab_size)

            # 必要なのは 最後の位置の予測 だけ:
            # ここまでの全トークンを見た上で「次に来る単語」を予測している
            next_logit = logits[0, -1, :]         # (vocab_size,) — 次単語のスコア

            # 最高スコアの単語を取る (Greedy デコーディング)
            next_id = torch.argmax(next_logit).item()
            tokens.append(next_id)

    # 全 id を単語に戻して結合
    return " ".join(id2word[t] for t in tokens)


# ==============================================================================
# Main — すべてを組み合わせる
# ==============================================================================

if __name__ == "__main__":
    # ちょうど Transformer を見せるのに十分な、小さなコーパス
    corpus = (
        "the cat sat on the mat . the dog sat on the log . "
        "the cat saw the dog . the dog saw the cat . "
        "the cat sat on the log . the dog sat on the mat ."
    )

    # Step 1: 語彙を作る (word ↔ id のマッピング)
    vocab, id2word = build_vocab(corpus)
    print(f"vocab size: {len(vocab)}")
    print(f"vocab: {vocab}\n")

    # Step 2: 訓練データを作る (入力 / ターゲットのペア)
    inputs, targets = make_training_data(corpus, vocab)
    print(f"training samples: {inputs.shape[0]}, seq_len: {inputs.shape[1]}\n")

    # Step 3: モデルを作って訓練する
    model = TinyTransformer(len(vocab))
    train(model, inputs, targets)

    # Step 4: プロンプトから文を生成する
    print("\n--- Generation ---")
    prompts = ["the cat sat on", "the dog saw"]
    for prompt in prompts:
        print(f'prompt: "{prompt}"')
        print(f"output: {generate(model, prompt, vocab, id2word)}")
        print()
