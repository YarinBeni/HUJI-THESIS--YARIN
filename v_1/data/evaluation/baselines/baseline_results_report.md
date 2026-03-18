# LLM Baseline Evaluation Report

**Generated**: 2026-01-28 18:49:31

**Total texts**: 4,976

**Models evaluated**: 1

## Summary: Period Prediction Accuracy

| Model | Accuracy | F1 (Macro) | F1 (Weighted) | Valid Predictions |
|-------|----------|------------|---------------|-------------------|
| gpt-oss-20b | 22.2% | 0.121 | 0.364 | 9 / 10 |

---

## Model: gpt-oss-20b

### Period Classification

- **Accuracy**: 22.2%
- **F1 (Macro)**: 0.121
- **F1 (Weighted)**: 0.364
- **Precision (Macro)**: 0.333
- **Recall (Macro)**: 0.074

#### Per-Period Breakdown

| Period | Total | Correct | Accuracy |
|--------|-------|---------|----------|
| Old Babylonian | 10 | 2 | 20.0% |

#### Per-Group Breakdown

| Group | Accuracy | F1 (Macro) | N |
|-------|----------|------------|---|
| Group 1 | 22.2% | 0.121 | 9 |

### Domain Classification

- **Accuracy**: 0.0%
- **F1 (Macro)**: 0.000

### Token Usage

- **Total input tokens**: 9,767
- **Total output tokens**: 14,722
- **Total tokens**: 24,489

### Confidence Distribution

- medium: 8
- low: 2

---

## Confusion Matrices


### gpt-oss-20b - Period

```
                 Old Baby  Neo-Assy  Late Bab
Old Babylonian          2         6         1
Neo-Assyrian            0         0         0
Late Babylonian         0         0         0
```