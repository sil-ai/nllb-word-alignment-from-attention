# Alignment Method Comparison: sil-ai/nllb-finetuned-spa-qup

- Source: data/spa-spablm.txt
- Target: data/qup-qupDBL.txt
- Sample size: 100

## Results by Layer

| Layer | Std Bidir | Shift Bidir | Delta | Std Conc | Shift Conc | Delta |
|-------|-----------|-------------|-------|----------|------------|-------|
|     3 | 0.3058    | 0.2616      | -0.0442 | 0.2207   | 0.2152     | -0.0055 |
|     5 | 0.2654    | 0.2379      | -0.0275 | 0.2943   | 0.2911     | -0.0032 |
|     7 | 0.2797    | 0.2636      | -0.0161 | 0.2676   | 0.2626     | -0.0050 |
|     9 | 0.2285    | 0.2186      | -0.0099 | 0.3669   | 0.3681     | +0.0011 |
|    11 | 0.2049    | 0.1929      | -0.0120 | 0.3495   | 0.3529     | +0.0034 |

## Summary

- Best Standard layer: 3 (bidirectional=0.3058)
- Best Shift-Att layer: 7 (bidirectional=0.2636)

**Standard method outperforms Shift-Att by 0.0422**
