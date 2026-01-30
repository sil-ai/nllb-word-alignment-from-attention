# Alignment Method Comparison: sil-ai/nllb-finetuned-ben-mjx

- Source: data/bn-SBCL.txt
- Target: data/mjx-mjx.txt
- Sample size: 100

## Results by Layer

| Layer | Std Bidir | Shift Bidir | Delta | Std Conc | Shift Conc | Delta |
|-------|-----------|-------------|-------|----------|------------|-------|
|     3 | 0.5620    | 0.4481      | -0.1139 | 0.2742   | 0.2622     | -0.0120 |
|     5 | 0.4475    | 0.3656      | -0.0819 | 0.3433   | 0.3390     | -0.0043 |
|     7 | 0.4072    | 0.3651      | -0.0422 | 0.3118   | 0.3107     | -0.0011 |
|     9 | 0.3164    | 0.2951      | -0.0213 | 0.4119   | 0.4082     | -0.0037 |
|    11 | 0.2147    | 0.2135      | -0.0012 | 0.4595   | 0.4633     | +0.0038 |

## Summary

- Best Standard layer: 3 (bidirectional=0.5620)
- Best Shift-Att layer: 3 (bidirectional=0.4481)

**Standard method outperforms Shift-Att by 0.1139**
