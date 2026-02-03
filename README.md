# NLLB Word Alignment from Attention

Extract word alignments from NLLB (No Language Left Behind) model attention weights, comparing Standard vs Shift-Att methods.

## Background

When using neural machine translation models like NLLB for word alignment extraction, attention weights from cross-attention layers can indicate which source words a target word "attended to" during translation. This provides unsupervised word alignment without requiring parallel alignment data.

### Teacher Forcing for Attention Extraction

To extract attention weights, we use **teacher forcing**: provide the target tokens as decoder input and extract cross-attention from a forward pass. The key is proper alignment between:

- **decoder_input_ids**: What the model sees as input (right-shifted with START token prepended)
- **labels**: What the model is predicting at each position (the actual target tokens)

We shift labels right by prepending `decoder_start_token_id` and removing the last token, ensuring `attention[t]` corresponds to predicting `labels[t]`, not inputting it.

### Shift-Att Method

The **Shift-Att** method was proposed by [Zenkel et al. (EMNLP 2020)](https://aclanthology.org/2020.emnlp-main.187/) as a way to extract higher-quality alignments from transformer attention.

**The idea**: With proper teacher forcing, `attention[t]` is the attention when predicting token `t`. Shift-Att uses `attention[t+1]` instead — one position beyond the prediction point.

### Our Hypothesis

We hypothesized that Shift-Att would improve alignment quality for finetuned NLLB models used in low-resource language settings.

## Findings

**Empirical testing on 7 finetuned NLLB models showed that Shift-Att produces BETTER results than the standard method.**

**Metric**: Bidirectional agreement score — measures how consistently the attention-based alignments agree when computed in both directions (source→target and target→source). Higher scores indicate better alignment quality.

| Language Pair | Model | Best Layer | Standard | Shift-Att | Delta |
|---------------|-------|:----------:|:--------:|:---------:|:-----:|
| eng → nih | sil-ai/nllb-finetuned-eng-nih | 6 (Shift) / 10 (Std) | 0.5467 | 0.5623 | +0.0156 |
| ben → mjx | sil-ai/nllb-finetuned-ben-mjx | 5 (Shift) / 16 (Std) | 0.7653 | 0.7868 | +0.0215 |
| eng → npi | sil-ai/nllb-finetuned-eng-npi | 5 | 0.5725 | 0.6831 | +0.1106 |
| eng → pcm | sil-ai/nllb-finetuned-eng-pcm | 3 | 0.5564 | 0.6108 | +0.0544 |
| eng → qup | sil-ai/nllb-finetuned-eng-qup | 5 (Shift) / 14 (Std) | 0.4532 | 0.4969 | +0.0437 |
| eng → spa | sil-ai/nllb-finetuned-eng-spa | 5 (Shift) / 12 (Std) | 0.6219 | 0.6826 | +0.0608 |
| spa → qup | sil-ai/nllb-finetuned-spa-qup | 5 | 0.4151 | 0.4641 | +0.0490 |

### Key Findings

1. **Shift-Att outperforms Standard in all 7 language pairs tested**
2. **Layer 5 is optimal for Shift-Att** across most models
3. **Delta ranges from +0.0156 to +0.1106**, representing 3-19% relative improvement for Shift-Att
4. **Consistent across language families**: Latin scripts (eng, spa, pcm), Devanagari (npi, mjx), Bengali script (ben), and Quechua
5. **Alignment quality varies by language pair**: scores range from 0.46 (spa-qup) to 0.79 (ben-mjx), likely reflecting model quality and language complexity

**Recommendation**: Use the Shift-Att method (`use_shift_att=True`) with layer 5 for finetuned NLLB models.

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
├── modal_app.py                 # Modal app for GPU inference (deploy this first)
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

# Deploy the Modal app (creates 'nllb-alignment' app with GPU support)
modal deploy modal_app.py
```

**Note**: GPU inference can run either:
- **Modal cloud** (default): Requires `modal deploy modal_app.py` first. Model cached in persistent volume.
- **Local GPU** (`--local` flag): Requires CUDA and additional dependencies:
  ```bash
  pip install torch transformers sentencepiece accelerate
  ```

## Usage

### Running the Comparison

```bash
# Run on Modal cloud (default)
python compare_methods.py \
    --model sil-ai/nllb-finetuned-eng-nih \
    --source data/eng-BSB.txt \
    --target data/nih-NIH.txt \
    --sample-size 100 \
    --layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 \
    --output results/eng-nih.md

# Or run on local GPU (requires CUDA)
python compare_methods.py \
    --model sil-ai/nllb-finetuned-eng-nih \
    --source data/eng-BSB.txt \
    --target data/nih-NIH.txt \
    --sample-size 100 \
    --local
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
    use_shift_att=True  # Use Shift-Att method (recommended)
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
- Layers tested: All 24 decoder cross-attention layers (0-23)
- All models are SIL-AI finetuned NLLB variants
- Data: Vref-aligned Bible texts
- Teacher forcing: decoder sees `[START, tok1, tok2, ...]` while predicting `[tok1, tok2, ..., EOS]`

## References

- Zenkel, T., Wuebker, J., & DeNero, J. (2020). [End-to-End Neural Word Alignment Outperforms GIZA++](https://aclanthology.org/2020.emnlp-main.187/). EMNLP 2020.
- Costa-jussà, M. R., et al. (2022). [No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672). arXiv.

## License

MIT License - See LICENSE file for details.
