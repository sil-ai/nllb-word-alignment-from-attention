# Alignment Method Comparison: sil-ai/nllb-finetuned-eng-spa

- Source: data/eng-BSB.txt
- Target: data/spa-spablm.txt
- Sample size: 100

## Results by Layer

| Layer | Std Bidir | Shift Bidir | Delta | Std Conc | Shift Conc | Delta |
|-------|-----------|-------------|-------|----------|------------|-------|
|     3 | 0.3706    | 0.3471      | -0.0235 | 0.3135   | 0.3229     | +0.0094 |
|     5 | 0.2136    | 0.2021      | -0.0115 | 0.3394   | 0.3561     | +0.0167 |
|     7 | 0.3159    | 0.3057      | -0.0102 | 0.2375   | 0.2494     | +0.0119 |
|     9 | 0.2915    | 0.3049      | +0.0134 | 0.2992   | 0.3026     | +0.0034 |
|    11 | 0.2227    | 0.2186      | -0.0041 | 0.3353   | 0.3363     | +0.0010 |

## Summary

- Best Standard layer: 3 (bidirectional=0.3706)
- Best Shift-Att layer: 3 (bidirectional=0.3471)

**Standard method outperforms Shift-Att by 0.0235**
