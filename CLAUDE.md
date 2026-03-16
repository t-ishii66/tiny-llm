# tiny-LLM

## Goal
最小限のPythonコードでTransformerの核心を理解するための学習プロジェクト。
性能は度外視し、LLMのコアアルゴリズムを最も簡潔に実装する。

## Design Principles
- **最短コード**: 可読性を保ちつつ、最小限のコードで実装する
- **トークン = 単語**: サブワードトークナイザは使わない。空白分割のみ
- **英語のみ**: 多言語対応は不要
- **性能度外視**: 精度・速度より構造の理解を優先
- **依存最小**: PyTorch使用（順伝播は手書き、逆伝播はtorch.autogradに任せる）

## Core Concepts to Implement
1. **Embedding**: 単語 -> ベクトル変換
2. **Positional Encoding**: 位置情報の付加
3. **Self-Attention**: Query, Key, Value による注意機構
4. **Multi-Head Attention**: 複数のAttentionヘッド
5. **Feed-Forward Network**: 全結合層
6. **Transformer Block**: 上記を組み合わせたブロック
7. **Text Generation**: 学習済みモデルからのテキスト生成

## Project Structure
```
tiny-LLM/
  CLAUDE.md        # This file
  tiny_llm.py      # All implementation in one file
```

## Language
- Code comments: English
- Documentation: Japanese OK
