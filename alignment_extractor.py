"""
Word alignment extraction from NLLB attention matrices.

This module provides:
- Token-to-word mapping using SentencePiece boundaries
- Attention aggregation from token-level to word-level
- Alignment extraction with configurable thresholds
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


# Special tokens to filter out
SPECIAL_TOKENS = {
    "<pad>",
    "<s>",
    "</s>",
    "<unk>",
}


@dataclass
class WordMapping:
    """Mapping between tokens and words."""

    tokens: List[str]
    words: List[str]
    tok2word: List[int]  # token index -> word index (-1 for special tokens)
    word2toks: List[List[int]]  # word index -> list of token indices


@dataclass
class AlignmentResult:
    """Result of word alignment extraction for a single sentence pair."""

    src_words: List[str]
    tgt_words: List[str]
    pharaoh: str  # "0-0 1-2 2-1" format
    alignments: List[dict]  # Detailed alignment info


def is_word_start(token: str) -> bool:
    """Check if token starts a new word (SentencePiece boundary)."""
    return token.startswith("\u2581") or token.startswith("<")


def is_special_token(token: str) -> bool:
    """Check if token is a special token to filter out."""
    if token in SPECIAL_TOKENS:
        return True
    # Filter language tokens like swh_Latn, eng_Latn, etc.
    if "_Latn" in token or "_Ethi" in token or "_Arab" in token or "_Deva" in token or "_Beng" in token:
        return True
    return token.startswith("<")


def map_tokens_to_words(tokens: List[str]) -> WordMapping:
    """
    Build mapping from subword tokens to word indices.

    SentencePiece uses "\u2581" (LOWER ONE EIGHTH BLOCK) to mark word boundaries.
    E.g., ["\u2581Hab", "ari", "\u2581ya", "ko"] -> ["Habari", "yako"]

    Args:
        tokens: List of subword tokens from tokenizer

    Returns:
        WordMapping with bidirectional token-word mappings
    """
    words = []
    tok2word = []
    word2toks = []
    current_word_tokens = []
    current_word_text = ""

    for i, token in enumerate(tokens):
        if is_special_token(token):
            tok2word.append(-1)
            continue

        # Remove the "\u2581" prefix for text reconstruction
        clean_token = token.lstrip("\u2581")

        if is_word_start(token) and current_word_text:
            # Save previous word
            words.append(current_word_text)
            word2toks.append(current_word_tokens)
            current_word_tokens = []
            current_word_text = ""

        # Add to current word
        current_word_tokens.append(i)
        current_word_text += clean_token
        tok2word.append(len(words))  # Current word index

    # Save last word
    if current_word_text:
        words.append(current_word_text)
        word2toks.append(current_word_tokens)

    return WordMapping(
        tokens=tokens,
        words=words,
        tok2word=tok2word,
        word2toks=word2toks,
    )


def aggregate_attention_to_words(
    attention_matrix: np.ndarray,
    src_mapping: WordMapping,
    tgt_mapping: WordMapping,
    use_shift_att: bool = False,
) -> np.ndarray:
    """
    Aggregate token-level attention to word-level scores.

    Args:
        attention_matrix: (tgt_tokens, src_tokens) attention weights
        src_mapping: Source token-to-word mapping
        tgt_mapping: Target token-to-word mapping
        use_shift_att: If True, use Shift-Att method from Zenkel et al. (EMNLP 2020).
            With proper teacher forcing, attention[t] captures the attention when
            predicting token t. Shift-Att uses attention[t+1] instead, i.e., one
            position beyond when the token is being predicted.

            - use_shift_att=False (Standard): attention when token is being predicted
            - use_shift_att=True (Shift-Att): attention from position t+1

    Returns:
        (tgt_words, src_words) aggregated attention matrix
    """
    n_tgt_words = len(tgt_mapping.words)
    n_src_words = len(src_mapping.words)

    if n_tgt_words == 0 or n_src_words == 0:
        return np.zeros((max(1, n_tgt_words), max(1, n_src_words)))

    word_attention = np.zeros((n_tgt_words, n_src_words))

    for tgt_word_idx, tgt_tok_indices in enumerate(tgt_mapping.word2toks):
        for src_word_idx, src_tok_indices in enumerate(src_mapping.word2toks):
            # Sum attention from all target tokens to all source tokens in this word pair
            total_attn = 0.0
            count = 0
            for tgt_tok in tgt_tok_indices:
                # Shift-Att: use attention from position t+1 (when token t is input)
                # instead of position t (when token t is output)
                effective_tgt_tok = tgt_tok + 1 if use_shift_att else tgt_tok
                for src_tok in src_tok_indices:
                    if effective_tgt_tok < attention_matrix.shape[0] and src_tok < attention_matrix.shape[1]:
                        total_attn += attention_matrix[effective_tgt_tok, src_tok]
                        count += 1
            if count > 0:
                word_attention[tgt_word_idx, src_word_idx] = total_attn / count

    return word_attention


def extract_alignments(
    word_attention: np.ndarray,
    src_words: List[str],
    tgt_words: List[str],
    threshold_abs: float = 0.01,
    threshold_rel: float = 0.1,
) -> AlignmentResult:
    """
    Extract word alignments from attention matrix using thresholds.

    An alignment is kept if:
    - Attention score >= threshold_abs (absolute threshold), AND
    - Attention score >= threshold_rel * max_attention_for_that_target_word

    Args:
        word_attention: (tgt_words, src_words) attention matrix
        src_words: Source word list
        tgt_words: Target word list
        threshold_abs: Minimum absolute attention score
        threshold_rel: Minimum relative attention score (fraction of max)

    Returns:
        AlignmentResult with Pharaoh format and detailed alignments
    """
    alignments = []
    pharaoh_pairs = []

    for tgt_idx in range(len(tgt_words)):
        if tgt_idx >= word_attention.shape[0]:
            continue

        row = word_attention[tgt_idx]
        max_attn = row.max() if len(row) > 0 else 0

        for src_idx in range(len(src_words)):
            if src_idx >= word_attention.shape[1]:
                continue

            score = row[src_idx]

            # Apply thresholds
            if score >= threshold_abs and score >= threshold_rel * max_attn:
                alignments.append({
                    "src_idx": src_idx,
                    "tgt_idx": tgt_idx,
                    "score": float(score),
                    "src_word": src_words[src_idx],
                    "tgt_word": tgt_words[tgt_idx],
                })
                pharaoh_pairs.append(f"{src_idx}-{tgt_idx}")

    # Sort Pharaoh format by source index, then target index
    pharaoh_pairs.sort(key=lambda x: (int(x.split("-")[0]), int(x.split("-")[1])))
    pharaoh_str = " ".join(pharaoh_pairs)

    return AlignmentResult(
        src_words=src_words,
        tgt_words=tgt_words,
        pharaoh=pharaoh_str,
        alignments=alignments,
    )


def compute_bidirectional_agreement(word_attention: np.ndarray) -> float:
    """
    Compute bidirectional agreement score.

    For each target word, find its top-aligned source word.
    Then check if that source word's top alignment points back to the same target word.

    Args:
        word_attention: (tgt_words, src_words) attention matrix

    Returns:
        Agreement score between 0 and 1
    """
    if word_attention.size == 0:
        return 0.0

    agreements = []
    for tgt_idx in range(word_attention.shape[0]):
        row = word_attention[tgt_idx]
        if len(row) == 0 or np.sum(row) == 0:
            continue
        top_src_idx = np.argmax(row)
        # Check reverse
        if top_src_idx < word_attention.shape[1]:
            col = word_attention[:, top_src_idx]
            if len(col) > 0 and np.sum(col) > 0:
                reverse_top = np.argmax(col)
                agreements.append(1.0 if reverse_top == tgt_idx else 0.0)

    return np.mean(agreements) if agreements else 0.0


def compute_concentration(word_attention: np.ndarray) -> float:
    """
    Compute concentration score (entropy-based).

    Lower entropy = higher concentration = more focused attention.

    Args:
        word_attention: (tgt_words, src_words) attention matrix

    Returns:
        Concentration score between 0 and 1
    """
    if word_attention.size == 0:
        return 0.0

    concentrations = []
    for row in word_attention:
        if len(row) == 0 or np.sum(row) == 0:
            continue
        normalized = row / np.sum(row)
        normalized = normalized[normalized > 0]
        entropy = -np.sum(normalized * np.log2(normalized + 1e-10))
        max_entropy = np.log2(len(row)) if len(row) > 1 else 1
        concentration = 1 - (entropy / max_entropy) if max_entropy > 0 else 1
        concentrations.append(concentration)

    return np.mean(concentrations) if concentrations else 0.0
