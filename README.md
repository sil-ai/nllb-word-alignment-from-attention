# NLLB Word Alignment from Attention

Extract word alignments from NLLB (No Language Left Behind) model attention weights, comparing Standard vs Shift-Att methods.

## Background

When using neural machine translation models like NLLB for word alignment extraction, attention weights from cross-attention layers can indicate which source words a target word "attended to" during translation. This provides unsupervised word alignment without requiring parallel alignment data.

### Shift-Att Method

The **Shift-Att** method was proposed by [Zenkel et al. (EMNLP 2020)](https://aclanthology.org/2020.emnlp-main.187/) as a way to extract higher-quality alignments from transformer attention.

**The idea**: In autoregressive decoding, when the model outputs token `t`, the decoder input is token `t-1`. Shift-Att extracts alignments from position `t+1` (where token `t` is the input) instead of position `t` (where token `t` is the output). The hypothesis is that attention is more informative when the token is being "read" as input rather than "written" as output.

### Our Hypothesis

We hypothesized that Shift-Att would improve alignment quality for finetuned NLLB models used in low-resource language settings.

### Findings

**Empirical testing on two finetuned NLLB models showed that Shift-Att produces WORSE results than the standard method across all layers.**

| Model | Best Layer | Standard Bidirectional | Shift-Att Bidirectional | Difference |
|-------|------------|------------------------|-------------------------|------------|
| eng-nih | Layer 5 | 0.6234 | 0.5891 | -5.5% |
| ben-mjx | Layer 5 | 0.5987 | 0.5612 | -6.3% |

The standard method (extracting attention when the token is output) consistently outperformed Shift-Att for bidirectional agreement - a key indicator of alignment quality.

**Recommendation**: Use the standard method (`use_shift_att=False`) for finetuned NLLB models.

## Why This Matters

This finding is important for:
1. **Low-resource language work**: NLLB finetuned models are often used for languages without existing alignment tools
2. **Reproducibility**: Others may attempt to implement Shift-Att based on the original paper without testing on their specific models
3. **Method selection**: The choice of alignment extraction method significantly impacts downstream tasks

## Repository Contents

```
nllb-word-alignment-from-attention/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── alignment_extractor.py       # Core alignment extraction code
├── compare_methods.py           # Comparison script (local, no Modal)
├── results/
│   ├── eng-nih-results.md       # Full results for English-Nih model
│   └── ben-mjx-results.md       # Full results for Bengali-Mahle model
└── data/
    └── README.md                # Instructions for obtaining test data
```

## Installation

```bash
git clone https://github.com/sil-ai/nllb-word-alignment-from-attention
cd nllb-word-alignment-from-attention
pip install -r requirements.txt

# Configure Modal (for GPU inference)
modal token new
```

**Note**: GPU inference runs remotely on [Modal](https://modal.com/). You need access to the `agent-critique` Modal app with the `AlignmentExtractor` class.

## Usage

### Running the Comparison

```bash
python compare_methods.py \
    --model sil-ai/nllb-finetuned-eng-nih \
    --source data/eng-source.txt \
    --target data/nih-target.txt \
    --sample-size 100
```

### Output

The script outputs:
1. Metrics comparison table across all 24 layers
2. Summary showing which method performs better
3. Markdown-formatted results for documentation

### Using the Alignment Extractor

```python
from alignment_extractor import (
    map_tokens_to_words,
    aggregate_attention_to_words,
    extract_alignments
)

# Get attention matrix from model (shape: [tgt_tokens, src_tokens])
attention_matrix = model_output.cross_attentions[layer]

# Map tokens to words
src_mapping = map_tokens_to_words(src_tokens)
tgt_mapping = map_tokens_to_words(tgt_tokens)

# Aggregate to word level
word_attention = aggregate_attention_to_words(
    attention_matrix,
    src_mapping,
    tgt_mapping,
    use_shift_att=False  # Use standard method (recommended)
)

# Extract alignments
result = extract_alignments(word_attention, src_mapping.words, tgt_mapping.words)
print(result.pharaoh)  # "0-0 1-2 2-1 ..." format
```

## Metrics Explained

- **Bidirectional Agreement**: For each target word's top-aligned source word, check if that source word's top alignment points back to the original target word. Higher is better.
- **Concentration**: Entropy-based measure of how focused attention is (vs. spread across all words). Higher means more decisive alignments.
- **Sparsity**: Fraction of attention weights below threshold. Higher means cleaner alignments.
- **Confidence**: Fraction of alignments with attention weight >= 0.1.
- **Rare Word Consistency**: For words appearing 2-5 times, how often they align to the same source word.

## References

- Zenkel, T., Wuebker, J., & DeNero, J. (2020). [End-to-End Neural Word Alignment Outperforms GIZA++](https://aclanthology.org/2020.emnlp-main.187/). EMNLP 2020.
- Costa-jussà, M. R., et al. (2022). [No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672). arXiv.

## License

MIT License - See LICENSE file for details.
