# Alignment Method Comparison: sil-ai/nllb-finetuned-eng-nih

- Model: English to Nih (Nigeria)
- Sample size: 100 verse pairs
- Evaluation: Bidirectional agreement and concentration metrics

## Results by Layer

| Layer | Std Bidir | Shift Bidir | Delta | Std Conc | Shift Conc | Delta |
|-------|-----------|-------------|-------|----------|------------|-------|
|     0 | 0.4521    | 0.4312      | -0.0209 | 0.3845   | 0.3721     | -0.0124 |
|     1 | 0.4823    | 0.4598      | -0.0225 | 0.4012   | 0.3876     | -0.0136 |
|     2 | 0.5134    | 0.4891      | -0.0243 | 0.4234   | 0.4089     | -0.0145 |
|     3 | 0.5467    | 0.5198      | -0.0269 | 0.4498   | 0.4321     | -0.0177 |
|     4 | 0.5812    | 0.5523      | -0.0289 | 0.4789   | 0.4587     | -0.0202 |
|     5 | 0.6234    | 0.5891      | -0.0343 | 0.5123   | 0.4876     | -0.0247 |
|     6 | 0.6098    | 0.5756      | -0.0342 | 0.5034   | 0.4798     | -0.0236 |
|     7 | 0.5934    | 0.5612      | -0.0322 | 0.4923   | 0.4689     | -0.0234 |
|     8 | 0.5756    | 0.5445      | -0.0311 | 0.4812   | 0.4578     | -0.0234 |
|     9 | 0.5589    | 0.5287      | -0.0302 | 0.4698   | 0.4456     | -0.0242 |
|    10 | 0.5412    | 0.5123      | -0.0289 | 0.4576   | 0.4334     | -0.0242 |
|    11 | 0.5234    | 0.4956      | -0.0278 | 0.4445   | 0.4212     | -0.0233 |
|    12 | 0.5056    | 0.4789      | -0.0267 | 0.4312   | 0.4089     | -0.0223 |
|    13 | 0.4878    | 0.4623      | -0.0255 | 0.4178   | 0.3967     | -0.0211 |
|    14 | 0.4712    | 0.4467      | -0.0245 | 0.4045   | 0.3845     | -0.0200 |
|    15 | 0.4534    | 0.4298      | -0.0236 | 0.3912   | 0.3723     | -0.0189 |
|    16 | 0.4356    | 0.4134      | -0.0222 | 0.3778   | 0.3598     | -0.0180 |
|    17 | 0.4189    | 0.3978      | -0.0211 | 0.3645   | 0.3476     | -0.0169 |
|    18 | 0.4012    | 0.3812      | -0.0200 | 0.3512   | 0.3354     | -0.0158 |
|    19 | 0.3845    | 0.3656      | -0.0189 | 0.3378   | 0.3231     | -0.0147 |
|    20 | 0.3678    | 0.3498      | -0.0180 | 0.3245   | 0.3109     | -0.0136 |
|    21 | 0.3512    | 0.3345      | -0.0167 | 0.3112   | 0.2987     | -0.0125 |
|    22 | 0.3345    | 0.3189      | -0.0156 | 0.2978   | 0.2865     | -0.0113 |
|    23 | 0.3189    | 0.3045      | -0.0144 | 0.2845   | 0.2743     | -0.0102 |

## Summary

- Best Standard layer: 5 (bidirectional=0.6234)
- Best Shift-Att layer: 5 (bidirectional=0.5891)

**Standard method outperforms Shift-Att by 0.0343**

## Key Observations

1. **Layer 5 is optimal** for both methods, consistent with findings that middle encoder layers capture alignment information best.

2. **Standard method wins across ALL layers** - Shift-Att never outperforms the standard method, with deltas ranging from -0.0144 to -0.0343.

3. **Peak performance gap** occurs at layer 5, where the difference is most pronounced (-5.5% relative).

4. **Concentration follows same pattern** - Standard method also achieves higher attention concentration across all layers.

## Recommendation

Use `use_shift_att=False` (the default) for this model.
