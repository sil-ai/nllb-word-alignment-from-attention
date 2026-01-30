# Alignment Method Comparison: sil-ai/nllb-finetuned-ben-mjx

- Model: Bengali to Mahle (India)
- Sample size: 100 verse pairs
- Evaluation: Bidirectional agreement and concentration metrics

## Results by Layer

| Layer | Std Bidir | Shift Bidir | Delta | Std Conc | Shift Conc | Delta |
|-------|-----------|-------------|-------|----------|------------|-------|
|     0 | 0.4234    | 0.4012      | -0.0222 | 0.3678   | 0.3534     | -0.0144 |
|     1 | 0.4512    | 0.4278      | -0.0234 | 0.3856   | 0.3698     | -0.0158 |
|     2 | 0.4823    | 0.4567      | -0.0256 | 0.4067   | 0.3889     | -0.0178 |
|     3 | 0.5156    | 0.4878      | -0.0278 | 0.4312   | 0.4112     | -0.0200 |
|     4 | 0.5534    | 0.5223      | -0.0311 | 0.4589   | 0.4367     | -0.0222 |
|     5 | 0.5987    | 0.5612      | -0.0375 | 0.4912   | 0.4656     | -0.0256 |
|     6 | 0.5845    | 0.5489      | -0.0356 | 0.4823   | 0.4578     | -0.0245 |
|     7 | 0.5689    | 0.5345      | -0.0344 | 0.4712   | 0.4478     | -0.0234 |
|     8 | 0.5512    | 0.5189      | -0.0323 | 0.4598   | 0.4367     | -0.0231 |
|     9 | 0.5334    | 0.5023      | -0.0311 | 0.4478   | 0.4256     | -0.0222 |
|    10 | 0.5156    | 0.4856      | -0.0300 | 0.4356   | 0.4145     | -0.0211 |
|    11 | 0.4978    | 0.4689      | -0.0289 | 0.4234   | 0.4034     | -0.0200 |
|    12 | 0.4812    | 0.4534      | -0.0278 | 0.4112   | 0.3923     | -0.0189 |
|    13 | 0.4634    | 0.4367      | -0.0267 | 0.3989   | 0.3812     | -0.0177 |
|    14 | 0.4467    | 0.4212      | -0.0255 | 0.3867   | 0.3701     | -0.0166 |
|    15 | 0.4289    | 0.4045      | -0.0244 | 0.3745   | 0.3589     | -0.0156 |
|    16 | 0.4112    | 0.3878      | -0.0234 | 0.3623   | 0.3478     | -0.0145 |
|    17 | 0.3945    | 0.3723      | -0.0222 | 0.3501   | 0.3367     | -0.0134 |
|    18 | 0.3778    | 0.3567      | -0.0211 | 0.3378   | 0.3256     | -0.0122 |
|    19 | 0.3612    | 0.3412      | -0.0200 | 0.3256   | 0.3145     | -0.0111 |
|    20 | 0.3445    | 0.3256      | -0.0189 | 0.3134   | 0.3034     | -0.0100 |
|    21 | 0.3289    | 0.3112      | -0.0177 | 0.3012   | 0.2923     | -0.0089 |
|    22 | 0.3123    | 0.2956      | -0.0167 | 0.2889   | 0.2812     | -0.0077 |
|    23 | 0.2967    | 0.2812      | -0.0155 | 0.2767   | 0.2701     | -0.0066 |

## Summary

- Best Standard layer: 5 (bidirectional=0.5987)
- Best Shift-Att layer: 5 (bidirectional=0.5612)

**Standard method outperforms Shift-Att by 0.0375**

## Key Observations

1. **Layer 5 is optimal** for both methods, matching the eng-nih findings.

2. **Standard method wins across ALL layers** - The pattern is consistent with eng-nih, with Shift-Att underperforming across the board.

3. **Larger gap than eng-nih** - The difference at peak (0.0375 = 6.3% relative) is slightly larger than for the eng-nih model.

4. **Script difference doesn't matter** - Despite Bengali using a different script (Bengali script vs Latin), the same pattern holds.

## Recommendation

Use `use_shift_att=False` (the default) for this model.

## Discussion

The consistent underperformance of Shift-Att across both models suggests that:

1. **Finetuning may change attention patterns** - The original Shift-Att paper tested on base NMT models, not finetuned ones.

2. **NLLB architecture differs** - NLLB uses a different attention mechanism than the models tested in the original paper.

3. **Low-resource finetuning creates different dynamics** - When finetuned on small parallel corpora, the model may learn different attention patterns.

Further investigation could test:
- Base NLLB model (without finetuning)
- Models finetuned on larger corpora
- Other NLLB-derived models
