# Alignment Method Comparison: sil-ai/nllb-finetuned-eng-qup

- Source: data/eng-BSB.txt
- Target: data/qup-qupDBL.txt
- Sample size: 100

## Results by Layer

| Layer | Std Bidir | Shift Bidir | Delta | Std Conc | Shift Conc | Delta |
|-------|-----------|-------------|-------|----------|------------|-------|
|     3 | 0.3752    | 0.3236      | -0.0516 | 0.2154   | 0.2084     | -0.0070 |
|     5 | 0.3319    | 0.2910      | -0.0409 | 0.2903   | 0.2869     | -0.0034 |
|     7 | 0.3269    | 0.3015      | -0.0255 | 0.2824   | 0.2785     | -0.0040 |
|     9 | 0.2579    | 0.2390      | -0.0189 | 0.3362   | 0.3363     | +0.0001 |
|    11 | 0.2103    | 0.1961      | -0.0142 | 0.3472   | 0.3486     | +0.0014 |

## Summary

- Best Standard layer: 3 (bidirectional=0.3752)
- Best Shift-Att layer: 3 (bidirectional=0.3236)

**Standard method outperforms Shift-Att by 0.0516**
