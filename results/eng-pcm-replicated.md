# Alignment Method Comparison: sil-ai/nllb-finetuned-eng-pcm

- Source: data/eng-BSB.txt
- Target: data/pcm-pcm.txt
- Sample size: 100

## Results by Layer

| Layer | Std Bidir | Shift Bidir | Delta | Std Conc | Shift Conc | Delta |
|-------|-----------|-------------|-------|----------|------------|-------|
|     3 | 0.5322    | 0.4806      | -0.0516 | 0.2662   | 0.2683     | +0.0021 |
|     5 | 0.4419    | 0.4022      | -0.0396 | 0.3196   | 0.3317     | +0.0120 |
|     7 | 0.3443    | 0.3267      | -0.0176 | 0.3093   | 0.3147     | +0.0055 |
|     9 | 0.2724    | 0.2697      | -0.0028 | 0.3990   | 0.4136     | +0.0146 |
|    11 | 0.2590    | 0.2611      | +0.0021 | 0.3771   | 0.3825     | +0.0053 |

## Summary

- Best Standard layer: 3 (bidirectional=0.5322)
- Best Shift-Att layer: 3 (bidirectional=0.4806)

**Standard method outperforms Shift-Att by 0.0516**
