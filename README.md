# NLLB Word Alignment from Attention

Extract word alignments from NLLB (No Language Left Behind) model attention weights, comparing Standard vs Shift-Att methods.

## Background

When using neural machine translation models like NLLB for word alignment extraction, attention weights from cross-attention layers can indicate which source words a target word "attended to" during translation. This provides unsupervised word alignment without requiring parallel alignment data.

### Shift-Att Method

The **Shift-Att** method was proposed by [Zenkel et al. (EMNLP 2020)](https://aclanthology.org/2020.emnlp-main.187/) as a way to extract higher-quality alignments from transformer attention.

**The idea**: In autoregressive decoding, when the model outputs token `t`, the decoder input is token `t-1`. Shift-Att extracts alignments from position `t+1` (where token `t` is the input) instead of position `t` (where token `t` is the output). The hypothesis is that attention is more informative when the token is being "read" as input rather than "written" as output.

### Our Hypothesis

We hypothesized that Shift-Att would improve alignment quality for finetuned NLLB models used in low-resource language settings.

## Findings

**Empirical testing on 7 finetuned NLLB models showed that Shift-Att produces WORSE results than the standard method.**

**Metric**: Bidirectional agreement score — measures how consistently the attention-based alignments agree when computed in both directions (source→target and target→source). Higher scores indicate better alignment quality.

| Language Pair | Model | Best Layer | Standard | Shift-Att | Delta |
|---------------|-------|:----------:|:--------:|:---------:|:-----:|
| eng → nih | sil-ai/nllb-finetuned-eng-nih | 3 | 0.3484 | 0.3143 | -0.0341 |
| ben → mjx | sil-ai/nllb-finetuned-ben-mjx | 3 | 0.5620 | 0.4522 | -0.1098 |
| eng → npi | sil-ai/nllb-finetuned-eng-npi | 2 | 0.3964 | 0.3305 | -0.0659 |
| eng → pcm | sil-ai/nllb-finetuned-eng-pcm | 3 | 0.5322 | 0.4806 | -0.0516 |
| eng → qup | sil-ai/nllb-finetuned-eng-qup | 3 | 0.3752 | 0.3236 | -0.0516 |
| eng → spa | sil-ai/nllb-finetuned-eng-spa | 2 | 0.4272 | 0.4003 | -0.0269 |
| spa → qup | sil-ai/nllb-finetuned-spa-qup | 3 | 0.3058 | 0.2636 | -0.0422 |

### Key Findings

1. **Standard method outperforms Shift-Att in all 7 language pairs tested**
2. **Early layers (2-3) are optimal** across all models — not middle layers as sometimes suggested
3. **Delta ranges from -0.0269 to -0.1098**, representing 6-20% relative improvement for Standard
4. **Consistent across language families**: Latin scripts (eng, spa, pcm), Devanagari (npi, mjx), Bengali script (ben), and Quechua
5. **Alignment quality varies by language pair**: scores range from 0.31 (spa-qup) to 0.56 (ben-mjx), likely reflecting model quality and language complexity

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
├── compare_methods.py           # Comparison script
├── results/
│   ├── README.md                # Summary of all results
│   ├── eng-nih.md               # Full layer-by-layer results
│   ├── ben-mjx.md
│   ├── eng-npi.md
│   ├── eng-pcm.md
│   ├── eng-qup.md
│   ├── eng-spa.md
│   └── spa-qup.md
└── data/
    └── README.md                # Instructions for obtaining test data
```

## Installation

```bash
git clone https://github.com/sil-ai/nllb-word-alignment-from-attention
cd nllb-word-alignment-from-attention
python -m venv .venv
source .venv/bin/activate
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
    --source data/eng-BSB.txt \
    --target data/nih-NIH.txt \
    --sample-size 100 \
    --layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 \
    --output results/eng-nih.md
```

### Output

The script outputs:
1. Metrics comparison table across specified layers
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

## Methodology

- Sample size: 100 sentence pairs per language pair (randomly sampled, both source and target non-empty)
- Layers tested: All 24 encoder layers (0-23)
- All models are SIL-AI finetuned NLLB variants
- Data: Vref-aligned Bible texts

## References

- Zenkel, T., Wuebker, J., & DeNero, J. (2020). [End-to-End Neural Word Alignment Outperforms GIZA++](https://aclanthology.org/2020.emnlp-main.187/). EMNLP 2020.
- Costa-jussà, M. R., et al. (2022). [No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672). arXiv.

## License

MIT License - See LICENSE file for details.
