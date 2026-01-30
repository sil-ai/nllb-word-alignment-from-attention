# Alignment Method Comparison Summary

Comparison of **Standard** vs **Shift-Att** methods for extracting word alignments from NLLB cross-attention.

- **Standard**: Extract attention when target token is the decoder *output*
- **Shift-Att**: Extract attention when target token is the decoder *input* (shifted by +1)

## Results

| Language Pair | Model | Best Layer | Standard | Shift-Att | Delta |
|---------------|-------|:----------:|:--------:|:---------:|:-----:|
| eng → nih | sil-ai/nllb-finetuned-eng-nih | 3 | 0.3484 | 0.3143 | -0.0341 |
| ben → mjx | sil-ai/nllb-finetuned-ben-mjx | 3 | 0.5620 | 0.4481 | -0.1139 |
| eng → npi | sil-ai/nllb-finetuned-eng-npi | 3 | 0.3412 | 0.2767 | -0.0645 |
| eng → pcm | sil-ai/nllb-finetuned-eng-pcm | 3 | 0.5322 | 0.4806 | -0.0516 |
| eng → qup | sil-ai/nllb-finetuned-eng-qup | 3 | 0.3752 | 0.3236 | -0.0516 |
| eng → spa | sil-ai/nllb-finetuned-eng-spa | 3 | 0.3706 | 0.3471 | -0.0235 |
| spa → qup | sil-ai/nllb-finetuned-spa-qup | 3 | 0.3058 | 0.2636 | -0.0422 |

*Values shown are bidirectional agreement scores at the best-performing layer for the Standard method.*

## Key Findings

1. **Standard method outperforms Shift-Att in all 7 language pairs tested**
2. **Layer 3 is optimal** across all models (of layers 3, 5, 7, 9, 11 tested)
3. **Delta ranges from -0.0235 to -0.1139**, representing 6-20% relative improvement
4. **Consistent across language families**: Latin scripts (eng, spa, pcm), Devanagari (npi, mjx), Bengali script (ben), and Quechua

## Recommendation

Use `use_shift_att=False` (the default) for finetuned NLLB models.

## Methodology

- Sample size: 100 sentence pairs per language pair
- Layers tested: 3, 5, 7, 9, 11
- Metric: Bidirectional agreement (alignment consistency in both directions)
- All models are SIL-AI finetuned NLLB variants
