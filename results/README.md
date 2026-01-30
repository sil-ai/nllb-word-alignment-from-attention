# Alignment Method Comparison Summary

Comparison of **Standard** vs **Shift-Att** methods for extracting word alignments from NLLB cross-attention.

- **Standard**: Extract attention when target token is the decoder *output*
- **Shift-Att**: Extract attention when target token is the decoder *input* (shifted by +1)

**Metric**: Bidirectional agreement score — measures how consistently the attention-based alignments agree when computed in both directions (source→target and target→source). Higher scores indicate better alignment quality.

## Results

| Language Pair | Model | Best Layer | Standard | Shift-Att | Delta |
|---------------|-------|:----------:|:--------:|:---------:|:-----:|
| eng → nih | sil-ai/nllb-finetuned-eng-nih | 3 | 0.3484 | 0.3143 | -0.0341 |
| ben → mjx | sil-ai/nllb-finetuned-ben-mjx | 3 | 0.5620 | 0.4522 | -0.1098 |
| eng → npi | sil-ai/nllb-finetuned-eng-npi | 2 | 0.3964 | 0.3305 | -0.0659 |
| eng → pcm | sil-ai/nllb-finetuned-eng-pcm | 3 | 0.5322 | 0.4806 | -0.0516 |
| eng → qup | sil-ai/nllb-finetuned-eng-qup | 3 | 0.3752 | 0.3236 | -0.0516 |
| eng → spa | sil-ai/nllb-finetuned-eng-spa | 2 | 0.4272 | 0.4003 | -0.0269 |
| spa → qup | sil-ai/nllb-finetuned-spa-qup | 3 | 0.3058 | 0.2636 | -0.0422 |

*Scores shown at the best-performing layer for the Standard method.*

## Key Findings

1. **Standard method outperforms Shift-Att in all 7 language pairs tested**
2. **Early layers (2-3) are optimal** across all models
3. **Delta ranges from -0.0269 to -0.1098**, representing 6-20% relative improvement
4. **Consistent across language families**: Latin scripts (eng, spa, pcm), Devanagari (npi, mjx), Bengali script (ben), and Quechua

## Recommendation

Use `use_shift_att=False` (the default) for finetuned NLLB models.

## Methodology

- Sample size: 100 sentence pairs per language pair
- Layers tested: All 24 encoder layers (0-23)
- All models are SIL-AI finetuned NLLB variants
