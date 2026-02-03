# Alignment Method Comparison Summary

Comparison of **Standard** vs **Shift-Att** methods for extracting word alignments from NLLB cross-attention.

With proper teacher forcing:
- **Standard**: `attention[t]` when predicting token `t`
- **Shift-Att**: `attention[t+1]`, one position beyond the prediction point

**Metric**: Bidirectional agreement score — measures how consistently the attention-based alignments agree when computed in both directions (source→target and target→source). Higher scores indicate better alignment quality.

## Results

| Language Pair | Model | Best Layer | Standard | Shift-Att | Delta |
|---------------|-------|:----------:|:--------:|:---------:|:-----:|
| eng → nih | sil-ai/nllb-finetuned-eng-nih | 6 (Shift) / 10 (Std) | 0.5467 | 0.5623 | +0.0156 |
| ben → mjx | sil-ai/nllb-finetuned-ben-mjx | 5 (Shift) / 16 (Std) | 0.7653 | 0.7868 | +0.0215 |
| eng → npi | sil-ai/nllb-finetuned-eng-npi | 5 | 0.5725 | 0.6831 | +0.1106 |
| eng → pcm | sil-ai/nllb-finetuned-eng-pcm | 3 | 0.5564 | 0.6108 | +0.0544 |
| eng → qup | sil-ai/nllb-finetuned-eng-qup | 5 (Shift) / 14 (Std) | 0.4532 | 0.4969 | +0.0437 |
| eng → spa | sil-ai/nllb-finetuned-eng-spa | 5 (Shift) / 12 (Std) | 0.6219 | 0.6826 | +0.0608 |
| spa → qup | sil-ai/nllb-finetuned-spa-qup | 5 | 0.4151 | 0.4641 | +0.0490 |

*Scores shown at the best-performing layer for each method.*

## Key Findings

1. **Shift-Att outperforms Standard in all 7 language pairs tested**
2. **Layer 5 is optimal for Shift-Att** across most models
3. **Delta ranges from +0.0156 to +0.1106**, representing 3-19% relative improvement
4. **Consistent across language families**: Latin scripts (eng, spa, pcm), Devanagari (npi, mjx), Bengali script (ben), and Quechua

## Recommendation

Use `use_shift_att=True` for finetuned NLLB models, with layer 5 as a good default.

## Methodology

- Sample size: 100 sentence pairs per language pair
- Layers tested: All 24 decoder layers (0-23)
- All models are SIL-AI finetuned NLLB variants
- Teacher forcing implemented correctly: decoder sees `[START, tok1, tok2, ...]` while predicting `[tok1, tok2, ..., EOS]`
