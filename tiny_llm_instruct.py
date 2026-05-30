"""tiny-LLM Chapter 5: Instruction tuning with Alpaca format.

Continues from tiny_llm.py. Pretrains a TinyTransformer on the natural-language
corpus (Stages 1-4 of the project), then instruction-tunes it on a tiny set of
Alpaca-formatted instruction/response pairs:

    ### Instruction: <task>
    ### Input: <optional context, omitted when not needed>
    ### Response: <expected answer>

Key ideas demonstrated:
  - Same Transformer architecture, same Cross-Entropy loss
  - Just a different shape of training data
  - Loss is computed ONLY on response tokens (response masking)
  - At inference, feed "### Instruction: ... ### Response:" and let the model
    generate what comes next.

Model and training utilities are imported from tiny_llm.py.
"""

import torch
import torch.nn.functional as F

from tiny_llm import (
    TinyTransformer,
    build_vocab,
    tokenize,
    make_training_data,
    train,
    SEQ_LEN,
    LR,
)


# ==============================================================================
# Alpaca format
# ==============================================================================
# Three slots:
#   - Instruction: the task description (always present)
#   - Input:       optional context (omitted when not needed)
#   - Response:    expected answer (the part the model must learn to produce)
#
# Our whitespace tokenizer treats "###", "Instruction:", "Input:", "Response:"
# as one token each (one "word" per whitespace-separated piece).

def format_alpaca(instruction, input_text, response):
    """Format one example using the Alpaca template.

    With input:
        ### Instruction: <ins> ### Input: <inp> ### Response: <resp>
    Without input:
        ### Instruction: <ins> ### Response: <resp>

    When `response` is empty, the result is the prompt (used at inference time
    and for computing the prefix length during training).
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
# Build training tensors with response masking
# ==============================================================================
# Standard next-token prediction:
#   input  = full_ids[0..T-1]
#   target = full_ids[1..T]
#
# Response mask is 1 at target positions that correspond to response tokens,
# and 0 elsewhere. At mask=0 positions the per-token loss is multiplied by 0
# and contributes nothing to the gradient, so the model never wastes capacity
# learning to predict the instruction text itself.

def build_example(instruction, input_text, response, vocab, seq_len):
    """Build (input_ids, target_ids, mask) for one Alpaca example.

    Args:
        instruction: task description string
        input_text:  optional input context (use "" if not needed)
        response:    the answer string the model should produce
        vocab:       word -> id mapping
        seq_len:     output tensor length

    Returns:
        inp:  list of token ids, length seq_len
        tgt:  list of token ids, length seq_len (inp shifted by 1)
        mask: list of 0/1, length seq_len (1 where target is a response token)
    """
    pad = vocab["<pad>"]

    # Token count of the prefix (everything up to and including "### Response:")
    prefix_text = format_alpaca(instruction, input_text, "")
    P = len(tokenize(prefix_text, vocab))

    # Full sequence (prefix + response)
    full_text = format_alpaca(instruction, input_text, response)
    full_ids = tokenize(full_text, vocab)
    F_len = len(full_ids)

    # Pad / truncate to seq_len + 1 (one extra slot for the shift)
    if F_len > seq_len + 1:
        full_ids = full_ids[:seq_len + 1]
        F_len = seq_len + 1
    full_ids = full_ids + [pad] * (seq_len + 1 - len(full_ids))

    inp = full_ids[:seq_len]            # input  positions 0..T-1
    tgt = full_ids[1:seq_len + 1]       # target positions 1..T

    # Response tokens live at full_ids indices [P, F_len - 1] inclusive.
    # We also include ONE position past the response so the model learns to
    # emit <pad> there — that's the "stop" signal for inference. Without it,
    # the model never knows when to stop after the response.
    # In target indexing (shifted by 1), the loss range is [P - 1, F_len - 1].
    mask = [1 if (P - 1) <= i <= (F_len - 1) else 0 for i in range(seq_len)]

    return inp, tgt, mask


def make_instruction_batch(examples, vocab):
    """Stack a list of (instruction, input, response) into batched tensors."""
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
# Instruction tuning loop
# ==============================================================================

def train_instruct(model, examples, vocab, epochs=300, lr=LR):
    """Fine-tune the pretrained model on Alpaca-formatted instruction examples.

    Identical to the pretraining loop except:
      - The training data comes from format_alpaca() output, not the raw corpus.
      - The per-position cross-entropy is multiplied by a response mask, so
        only response-token positions contribute to the loss.
    """
    inputs, targets, mask = make_instruction_batch(examples, vocab)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        logits = model.forward(inputs)                       # (B, T, V)

        # Per-position cross-entropy (no reduction yet)
        per_tok = F.cross_entropy(
            logits.view(-1, model.vocab_size),
            targets.view(-1),
            reduction="none",
        ).view(targets.shape)                                # (B, T)

        # Apply mask — only response positions contribute
        loss = (per_tok * mask).sum() / mask.sum().clamp(min=1.0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 30 == 0:
            print(f"epoch {epoch+1:4d}  loss={loss.item():.4f}")


# ==============================================================================
# Generate a response for one instruction
# ==============================================================================

def respond(model, instruction, input_text, vocab, id2word, max_tokens=8):
    """Generate a response.

    Feeds "### Instruction: ... ### Response:" and continues generation
    until either max_tokens new tokens are produced or a <pad> is sampled.

    Returns:
        The generated response text (the part AFTER "### Response:").
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
# Main — Stage 1 pretrain, Stage 2 instruction-tune, Stage 3 respond
# ==============================================================================

if __name__ == "__main__":
    # The corpus seeds the vocabulary with:
    #   - language words (used for Stage 1 pretraining)
    #   - Alpaca format tokens and the task word "who" (used for Stage 2)
    corpus = (
        "the cat sat on the mat . the dog sat on the log . "
        "the cat saw the dog . the dog saw the cat . "
        "### Instruction: Input: Response: who"
    )
    vocab, id2word = build_vocab(corpus)
    print(f"vocab size: {len(vocab)}")
    print(f"vocab: {vocab}\n")

    # --- Stage 1: Pretraining (Chapters 1-3) ---
    print("--- Stage 1: Pretraining ---")
    pretrain_text = (
        "the cat sat on the mat . the dog sat on the log . "
        "the cat saw the dog . the dog saw the cat . "
        "the cat sat on the log . the dog sat on the mat ."
    )
    inputs, targets = make_training_data(pretrain_text, vocab)
    model = TinyTransformer(len(vocab))
    train(model, inputs, targets)

    # --- Stage 2: Instruction tuning ---
    print("\n--- Stage 2: Instruction tuning ---")
    examples = [
        ("who sat on the mat", "", "the cat"),
        ("who sat on the log", "", "the dog"),
        ("who saw the dog",    "", "the cat"),
        ("who saw the cat",    "", "the dog"),
    ]
    train_instruct(model, examples, vocab)

    # --- Stage 3: Respond to instructions ---
    print("\n--- Stage 3: Responding ---")
    for ins in [
        "who sat on the mat",
        "who saw the dog",
        "who sat on the log",
    ]:
        print(f"### Instruction: {ins}")
        print(f"### Response: {respond(model, ins, '', vocab, id2word)}\n")
