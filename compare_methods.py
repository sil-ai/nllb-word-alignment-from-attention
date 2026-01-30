#!/usr/bin/env python3
"""
Compare standard vs Shift-Att alignment extraction methods.

This script runs both methods on sentence pairs and compares:
1. Intrinsic quality metrics (concentration, bidirectionality, etc.)
2. Layer-by-layer comparison across all 24 NLLB layers

Usage:
    python compare_methods.py \
        --model sil-ai/nllb-finetuned-eng-nih \
        --source data/eng-source.txt \
        --target data/nih-target.txt \
        --sample-size 100

Output:
    Markdown-formatted comparison table showing metrics for each layer.
"""

import argparse
import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from alignment_extractor import (
    map_tokens_to_words,
    aggregate_attention_to_words,
    compute_bidirectional_agreement,
    compute_concentration,
)


def load_sentence_pairs(
    source_path: str,
    target_path: str,
    sample_size: int = 100,
    seed: int = 42
) -> List[Dict]:
    """Load and sample sentence pairs from text files."""
    with open(source_path, 'r', encoding='utf-8') as f:
        source_lines = [line.strip() for line in f]
    with open(target_path, 'r', encoding='utf-8') as f:
        target_lines = [line.strip() for line in f]

    # Filter to non-empty pairs
    pairs = []
    for i, (src, tgt) in enumerate(zip(source_lines, target_lines)):
        if src and tgt:
            pairs.append({
                "idx": i,
                "src": src,
                "tgt": tgt,
            })

    # Sample
    random.seed(seed)
    if len(pairs) > sample_size:
        pairs = random.sample(pairs, sample_size)

    return pairs


def extract_attention_all_layers(
    model,
    tokenizer,
    src_text: str,
    tgt_text: str,
    src_lang: str = "eng_Latn",
    tgt_lang: str = "nih_Latn",
) -> Dict:
    """
    Extract cross-attention matrices from all layers using teacher forcing.

    Args:
        model: HuggingFace NLLB model
        tokenizer: Corresponding tokenizer
        src_text: Source language text
        tgt_text: Target language text (for teacher forcing)
        src_lang: Source language code
        tgt_lang: Target language code

    Returns:
        Dict with src_tokens, tgt_tokens, and attention_matrices by layer
    """
    device = next(model.parameters()).device

    # Tokenize source
    tokenizer.src_lang = src_lang
    src_inputs = tokenizer(src_text, return_tensors="pt").to(device)

    # Tokenize target with target language token prepended
    tokenizer.tgt_lang = tgt_lang
    with tokenizer.as_target_tokenizer():
        tgt_inputs = tokenizer(tgt_text, return_tensors="pt").to(device)

    # Run model with teacher forcing
    with torch.no_grad():
        outputs = model(
            input_ids=src_inputs["input_ids"],
            attention_mask=src_inputs["attention_mask"],
            decoder_input_ids=tgt_inputs["input_ids"],
            output_attentions=True,
            return_dict=True,
        )

    # Extract cross-attention from all layers
    # Shape of each: (batch, heads, tgt_len, src_len)
    cross_attentions = outputs.cross_attentions

    attention_matrices = {}
    for layer_idx, layer_attn in enumerate(cross_attentions):
        # Average across attention heads
        avg_attn = layer_attn[0].mean(dim=0).cpu().numpy()  # (tgt_len, src_len)
        attention_matrices[layer_idx] = avg_attn

    # Get tokens for mapping
    src_tokens = tokenizer.convert_ids_to_tokens(src_inputs["input_ids"][0])
    tgt_tokens = tokenizer.convert_ids_to_tokens(tgt_inputs["input_ids"][0])

    return {
        "src_tokens": src_tokens,
        "tgt_tokens": tgt_tokens,
        "attention_matrices": attention_matrices,
    }


def compute_metrics_for_layer(
    attention_results: List[Dict],
    layer: int,
    use_shift_att: bool = False,
) -> Dict[str, float]:
    """
    Compute quality metrics for a specific layer.

    Args:
        attention_results: List of attention extraction results
        layer: Layer index to evaluate
        use_shift_att: Whether to use Shift-Att method

    Returns:
        Dict of metric name -> value
    """
    bidirectional_scores = []
    concentration_scores = []
    sparsity_scores = []
    confidence_scores = []
    word_alignments: Dict[str, List[str]] = defaultdict(list)

    for result in attention_results:
        attn_matrices = result.get("attention_matrices", {})
        src_tokens = result.get("src_tokens", [])
        tgt_tokens = result.get("tgt_tokens", [])

        if layer not in attn_matrices or not src_tokens or not tgt_tokens:
            continue

        attn = attn_matrices[layer]

        # Map tokens to words
        src_mapping = map_tokens_to_words(src_tokens)
        tgt_mapping = map_tokens_to_words(tgt_tokens)

        if not src_mapping.words or not tgt_mapping.words:
            continue

        # Aggregate to word level
        word_attn = aggregate_attention_to_words(
            attn, src_mapping, tgt_mapping, use_shift_att=use_shift_att
        )

        # Bidirectional agreement
        bidirectional_scores.append(compute_bidirectional_agreement(word_attn))

        # Concentration
        concentration_scores.append(compute_concentration(word_attn))

        # Sparsity
        threshold = 0.05
        for row in word_attn:
            if len(row) == 0:
                continue
            row_sum = np.sum(row)
            if row_sum == 0:
                continue
            normalized = row / row_sum
            n_aligned = np.sum(normalized >= threshold)
            sparsity = 1.0 - (n_aligned / len(row))
            sparsity_scores.append(sparsity)

        # Confidence
        for row in word_attn:
            if len(row) == 0:
                continue
            max_attn = np.max(row)
            confidence_scores.append(1.0 if max_attn >= 0.1 else 0.0)

        # Collect for rare word consistency
        for tgt_idx, tgt_word in enumerate(tgt_mapping.words):
            if tgt_idx >= word_attn.shape[0]:
                continue
            row = word_attn[tgt_idx]
            if len(row) == 0 or np.sum(row) == 0:
                continue
            top_src_idx = np.argmax(row)
            if top_src_idx < len(src_mapping.words):
                src_word = src_mapping.words[top_src_idx]
                word_alignments[tgt_word.lower()].append(src_word.lower())

    # Compute rare word consistency
    consistencies = []
    for tgt_word, src_words in word_alignments.items():
        if 2 <= len(src_words) <= 5:
            counts = Counter(src_words)
            most_common_count = counts.most_common(1)[0][1]
            consistency = most_common_count / len(src_words)
            consistencies.append(consistency)

    return {
        "bidirectional": np.mean(bidirectional_scores) if bidirectional_scores else 0.5,
        "concentration": np.mean(concentration_scores) if concentration_scores else 0.5,
        "sparsity": np.mean(sparsity_scores) if sparsity_scores else 0.5,
        "confidence": np.mean(confidence_scores) if confidence_scores else 0.5,
        "rare_word_consistency": np.mean(consistencies) if consistencies else 0.5,
    }


def run_comparison(
    model_id: str,
    source_path: str,
    target_path: str,
    sample_size: int = 100,
    src_lang: str = "eng_Latn",
    tgt_lang: str = "nih_Latn",
    output_file: str = None,
):
    """
    Run full comparison between Standard and Shift-Att methods.

    Args:
        model_id: HuggingFace model identifier
        source_path: Path to source text file
        target_path: Path to target text file
        sample_size: Number of sentence pairs to evaluate
        src_lang: Source language code for NLLB
        tgt_lang: Target language code for NLLB
        output_file: Optional output file for results
    """
    print(f"\n{'='*70}")
    print("NLLB Alignment Method Comparison: Standard vs Shift-Att")
    print(f"{'='*70}")
    print(f"Model: {model_id}")
    print(f"Source: {source_path}")
    print(f"Target: {target_path}")
    print(f"Sample size: {sample_size}")

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")

    # Load sentence pairs
    print("\nLoading sentence pairs...")
    pairs = load_sentence_pairs(source_path, target_path, sample_size)
    print(f"Loaded {len(pairs)} non-empty sentence pairs")

    # Extract attention from all pairs
    print("\nExtracting attention matrices...")
    all_results = []
    for i, pair in enumerate(pairs):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(pairs)}")

        try:
            result = extract_attention_all_layers(
                model, tokenizer,
                pair["src"], pair["tgt"],
                src_lang=src_lang, tgt_lang=tgt_lang
            )
            all_results.append(result)
        except Exception as e:
            print(f"  Warning: Failed on pair {i}: {e}")

    print(f"Successfully extracted {len(all_results)} attention matrices")

    # Compare across all layers
    print("\n" + "="*70)
    print("LAYER-BY-LAYER COMPARISON")
    print("="*70)

    num_layers = len(all_results[0]["attention_matrices"]) if all_results else 24

    results_table = []
    results_table.append("| Layer | Std Bidir | Shift Bidir | Delta | Std Conc | Shift Conc | Delta |")
    results_table.append("|-------|-----------|-------------|-------|----------|------------|-------|")

    best_std_layer = 0
    best_std_score = 0
    best_shift_layer = 0
    best_shift_score = 0

    for layer in range(num_layers):
        std_metrics = compute_metrics_for_layer(all_results, layer, use_shift_att=False)
        shift_metrics = compute_metrics_for_layer(all_results, layer, use_shift_att=True)

        bidir_delta = shift_metrics["bidirectional"] - std_metrics["bidirectional"]
        conc_delta = shift_metrics["concentration"] - std_metrics["concentration"]

        results_table.append(
            f"| {layer:5d} | {std_metrics['bidirectional']:.4f}    | {shift_metrics['bidirectional']:.4f}      | {bidir_delta:+.4f} | {std_metrics['concentration']:.4f}   | {shift_metrics['concentration']:.4f}     | {conc_delta:+.4f} |"
        )

        if std_metrics["bidirectional"] > best_std_score:
            best_std_score = std_metrics["bidirectional"]
            best_std_layer = layer

        if shift_metrics["bidirectional"] > best_shift_score:
            best_shift_score = shift_metrics["bidirectional"]
            best_shift_layer = layer

    # Print results
    print("\n".join(results_table))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nBest Standard layer: {best_std_layer} (bidirectional={best_std_score:.4f})")
    print(f"Best Shift-Att layer: {best_shift_layer} (bidirectional={best_shift_score:.4f})")

    if best_std_score > best_shift_score:
        print(f"\nStandard method outperforms Shift-Att by {best_std_score - best_shift_score:.4f}")
        print("Recommendation: Use use_shift_att=False (default)")
    else:
        print(f"\nShift-Att outperforms Standard by {best_shift_score - best_std_score:.4f}")
        print("Recommendation: Use use_shift_att=True")

    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write(f"# Alignment Method Comparison: {model_id}\n\n")
            f.write(f"- Source: {source_path}\n")
            f.write(f"- Target: {target_path}\n")
            f.write(f"- Sample size: {len(all_results)}\n\n")
            f.write("## Results by Layer\n\n")
            f.write("\n".join(results_table))
            f.write(f"\n\n## Summary\n\n")
            f.write(f"- Best Standard layer: {best_std_layer} (bidirectional={best_std_score:.4f})\n")
            f.write(f"- Best Shift-Att layer: {best_shift_layer} (bidirectional={best_shift_score:.4f})\n")
            if best_std_score > best_shift_score:
                f.write(f"\n**Standard method outperforms Shift-Att by {best_std_score - best_shift_score:.4f}**\n")
            else:
                f.write(f"\n**Shift-Att outperforms Standard by {best_shift_score - best_std_score:.4f}**\n")
        print(f"\nResults saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare Standard vs Shift-Att alignment extraction methods"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model ID (e.g., sil-ai/nllb-finetuned-eng-nih)"
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to source text file (one sentence per line)"
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Path to target text file (one sentence per line, aligned with source)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of sentence pairs to sample (default: 100)"
    )
    parser.add_argument(
        "--src-lang",
        default="eng_Latn",
        help="Source language code for NLLB (default: eng_Latn)"
    )
    parser.add_argument(
        "--tgt-lang",
        default="nih_Latn",
        help="Target language code for NLLB (default: nih_Latn)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file for results (markdown format)"
    )

    args = parser.parse_args()

    run_comparison(
        model_id=args.model,
        source_path=args.source,
        target_path=args.target,
        sample_size=args.sample_size,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
