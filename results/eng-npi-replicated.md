# Alignment Method Comparison: sil-ai/nllb-finetuned-eng-npi

- Source: data/eng-BSB.txt
- Target: data/npi-NNRV.txt
- Sample size: 100

## Results by Layer

| Layer | Std Bidir | Shift Bidir | Delta | Std Conc | Shift Conc | Delta |
|-------|-----------|-------------|-------|----------|------------|-------|
|     3 | 0.3412    | 0.2767      | -0.0645 | 0.2324   | 0.2372     | +0.0048 |
|     5 | 0.1855    | 0.1760      | -0.0095 | 0.2890   | 0.2957     | +0.0067 |
|     7 | 0.2217    | 0.2126      | -0.0092 | 0.1987   | 0.1991     | +0.0004 |
|     9 | 0.2106    | 0.2062      | -0.0044 | 0.2682   | 0.2681     | -0.0001 |
|    11 | 0.1807    | 0.1814      | +0.0007 | 0.2570   | 0.2576     | +0.0007 |

## Summary

- Best Standard layer: 3 (bidirectional=0.3412)
- Best Shift-Att layer: 3 (bidirectional=0.2767)

**Standard method outperforms Shift-Att by 0.0645**
