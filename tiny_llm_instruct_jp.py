"""tiny-LLM 第5章: Alpaca 形式の Instruction Tuning。

tiny_llm_jp.py の続編。TinyTransformer を自然言語コーパスで事前学習 (1〜4 章)
した後、Alpaca 形式の指示 / 応答ペアで instruction tuning する:

    ### Instruction: <タスク>
    ### Input: <省略可能な入力>
    ### Response: <期待される応答>

ここで見せたい要点:
  - 同じ Transformer アーキテクチャ、同じ Cross-Entropy 損失
  - 違うのは「訓練データの形」と「どの位置で loss を取るか」だけ
  - Loss は応答トークンの位置でだけ計算する (Response マスキング)
  - 推論時は "### Instruction: ... ### Response:" を与え、続きを生成させる

モデルと訓練ユーティリティは tiny_llm_jp.py から import する。
"""

import torch
import torch.nn.functional as F

from tiny_llm_jp import (
    TinyTransformer,
    build_vocab,
    tokenize,
    make_training_data,
    train,
    SEQ_LEN,
    LR,
)


# ==============================================================================
# Alpaca 形式
# ==============================================================================
# 3 つのスロットからなる:
#   - Instruction: タスクの説明 (必ず存在)
#   - Input:       省略可能な文脈
#   - Response:    モデルが生成すべき答え
#
# 本プロジェクトの空白分割トークナイザは "###", "Instruction:", "Input:",
# "Response:" をそれぞれ 1 トークン (1 単語) として扱う。

def format_alpaca(instruction, input_text, response):
    """1 例を Alpaca テンプレートに整形する。

    Input ありの場合:
        ### Instruction: <ins> ### Input: <inp> ### Response: <resp>
    Input なしの場合:
        ### Instruction: <ins> ### Response: <resp>

    `response` が空文字のときは prompt 部分 (推論時に与える文字列、および
    訓練時に prefix 長を計算するための文字列) になる。
    """
    if input_text:
        return (
            f"### Instruction: {instruction} "
            f"### Input: {input_text} "
            f"### Response: {response}"
        ).strip()
    return (
        f"### Instruction: {instruction} "
        f"### Response: {response}"
    ).strip()


# ==============================================================================
# Response マスキング付きの訓練テンソルを作る
# ==============================================================================
# 標準的な次トークン予測:
#   input  = full_ids[0..T-1]
#   target = full_ids[1..T]
#
# Response マスクは、target が応答トークンに対応する位置で 1、それ以外で 0。
# mask=0 の位置では per-token 損失に 0 が掛かり、勾配に貢献しない。
# こうしてモデルは「指示テキスト自体を予測する」という無駄な学習をしない。

def build_example(instruction, input_text, response, vocab, seq_len):
    """1 つの Alpaca 例から (input_ids, target_ids, mask) を作る。

    Args:
        instruction: タスクの説明文
        input_text:  省略可能な入力文 (不要なら "")
        response:    モデルが出力すべき応答文
        vocab:       word → id マッピング
        seq_len:     出力テンソルの長さ

    Returns:
        inp:  トークン id のリスト、長さ seq_len
        tgt:  トークン id のリスト、長さ seq_len (inp を 1 つずらしたもの)
        mask: 0/1 のリスト、長さ seq_len (target が応答トークンの位置で 1)
    """
    pad = vocab["<pad>"]

    # prefix ("### Response:" まで) のトークン数
    prefix_text = format_alpaca(instruction, input_text, "")
    P = len(tokenize(prefix_text, vocab))

    # 全文 (prefix + 応答)
    full_text = format_alpaca(instruction, input_text, response)
    full_ids = tokenize(full_text, vocab)
    F_len = len(full_ids)

    # seq_len + 1 に揃える (1 つ余分は shift のため)
    if F_len > seq_len + 1:
        full_ids = full_ids[:seq_len + 1]
        F_len = seq_len + 1
    full_ids = full_ids + [pad] * (seq_len + 1 - len(full_ids))

    inp = full_ids[:seq_len]            # 入力 位置 0..T-1
    tgt = full_ids[1:seq_len + 1]       # ターゲット 位置 1..T

    # 応答トークンは full_ids の index [P, F_len - 1] にある。
    # さらに「応答末尾の直後の <pad> 1 つ」も loss に含める。
    # これがモデルへの「停止の合図」になる。これが無いと、推論時に応答後に止まれない。
    # target indexing では (1 つ shift して) [P - 1, F_len - 1]。
    mask = [1 if (P - 1) <= i <= (F_len - 1) else 0 for i in range(seq_len)]

    return inp, tgt, mask


def make_instruction_batch(examples, vocab):
    """(instruction, input, response) のリストをバッチテンソルにまとめる。"""
    inputs, targets, masks = [], [], []
    for ins, inp, resp in examples:
        i, t, m = build_example(ins, inp, resp, vocab, SEQ_LEN)
        inputs.append(i)
        targets.append(t)
        masks.append(m)
    return (
        torch.tensor(inputs),
        torch.tensor(targets),
        torch.tensor(masks, dtype=torch.float),
    )


# ==============================================================================
# Instruction Tuning ループ
# ==============================================================================

def train_instruct(model, examples, vocab, epochs=300, lr=LR):
    """事前学習済みモデルを Alpaca 形式の例で fine-tune する。

    事前学習ループと違うのは 2 点だけ:
      - 訓練データが生のコーパスではなく format_alpaca() の出力。
      - 各位置の Cross-Entropy 損失に応答マスクを掛けるので、
        応答トークン位置だけが loss に貢献する。
    """
    inputs, targets, mask = make_instruction_batch(examples, vocab)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        logits = model.forward(inputs)                       # (B, T, V)

        # 位置ごとの Cross-Entropy (まだ reduce しない)
        per_tok = F.cross_entropy(
            logits.view(-1, model.vocab_size),
            targets.view(-1),
            reduction="none",
        ).view(targets.shape)                                # (B, T)

        # マスクを適用 — 応答位置だけが loss に貢献する
        loss = (per_tok * mask).sum() / mask.sum().clamp(min=1.0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 30 == 0:
            print(f"epoch {epoch+1:4d}  loss={loss.item():.4f}")


# ==============================================================================
# 1 つの指示に対して応答を生成する
# ==============================================================================

def respond(model, instruction, input_text, vocab, id2word, max_tokens=8):
    """応答を生成する。

    "### Instruction: ... ### Response:" を与え、
    max_tokens 個生成するか <pad> を選ぶまで続きを生成する。

    Returns:
        生成された応答テキスト ("### Response:" 以降の部分)。
    """
    prompt = format_alpaca(instruction, input_text, "")
    prefix_len = len(tokenize(prompt, vocab))

    tokens = tokenize(prompt, vocab)
    with torch.no_grad():
        for _ in range(max_tokens):
            context = tokens[-SEQ_LEN:]
            x = torch.tensor([context])
            logits = model.forward(x)
            next_id = torch.argmax(logits[0, -1, :]).item()
            if next_id == vocab["<pad>"]:
                break
            tokens.append(next_id)

    return " ".join(id2word[t] for t in tokens[prefix_len:])


# ==============================================================================
# Main — Stage 1 事前学習、Stage 2 Instruction Tuning、Stage 3 応答
# ==============================================================================

if __name__ == "__main__":
    # コーパスで以下の語彙を仕込む:
    #   - 言語の単語 (Stage 1 の事前学習に使う)
    #   - Alpaca 形式のマーカーとタスク用単語 "who" (Stage 2 に使う)
    corpus = (
        "the cat sat on the mat . the dog sat on the log . "
        "the cat saw the dog . the dog saw the cat . "
        "### Instruction: Input: Response: who"
    )
    vocab, id2word = build_vocab(corpus)
    print(f"vocab size: {len(vocab)}")
    print(f"vocab: {vocab}\n")

    # --- Stage 1: 事前学習 (第 1〜3 章) ---
    print("--- Stage 1: Pretraining ---")
    pretrain_text = (
        "the cat sat on the mat . the dog sat on the log . "
        "the cat saw the dog . the dog saw the cat . "
        "the cat sat on the log . the dog sat on the mat ."
    )
    inputs, targets = make_training_data(pretrain_text, vocab)
    model = TinyTransformer(len(vocab))
    train(model, inputs, targets)

    # --- Stage 2: Instruction Tuning ---
    print("\n--- Stage 2: Instruction tuning ---")
    examples = [
        ("who sat on the mat", "", "the cat"),
        ("who sat on the log", "", "the dog"),
        ("who saw the dog",    "", "the cat"),
        ("who saw the cat",    "", "the dog"),
    ]
    train_instruct(model, examples, vocab)

    # --- Stage 3: 指示に応答する ---
    print("\n--- Stage 3: Responding ---")
    for ins in [
        "who sat on the mat",
        "who saw the dog",
        "who sat on the log",
    ]:
        print(f"### Instruction: {ins}")
        print(f"### Response: {respond(model, ins, '', vocab, id2word)}\n")
