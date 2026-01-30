# Alignment Method Comparison: sil-ai/nllb-finetuned-eng-nih

- Source: data/eng-BSB.txt
- Target: data/nih-NIH.txt
- Sample size: 100

## Results by Layer

| Layer | Std Bidir | Shift Bidir | Delta | Std Conc | Shift Conc | Delta |
|-------|-----------|-------------|-------|----------|------------|-------|
|     3 | 0.3484    | 0.3143      | -0.0341 | 0.1797   | 0.1786     | -0.0011 |
|     5 | 0.2591    | 0.2437      | -0.0155 | 0.2364   | 0.2402     | +0.0038 |
|     7 | 0.3326    | 0.3025      | -0.0301 | 0.2103   | 0.2101     | -0.0003 |
|     9 | 0.2824    | 0.2700      | -0.0124 | 0.3187   | 0.3229     | +0.0043 |
|    11 | 0.2375    | 0.2224      | -0.0152 | 0.3132   | 0.3159     | +0.0027 |

## Summary

- Best Standard layer: 3 (bidirectional=0.3484)
- Best Shift-Att layer: 3 (bidirectional=0.3143)

**Standard method outperforms Shift-Att by 0.0341**
