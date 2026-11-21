# Results

Model: `openai/gpt-oss-120b`, confidence threshold 0.6

## Metrics

| Metric | Value |
|---|---|
| Test cases | 41 |
| Precision | 100.0% |
| Recall | 100.0% |
| False-positive rate | 0.0% |
| Target-item accuracy (on caught decisions) | 100.0% |
| Overall accuracy (label + target) | 100.0% |
| Pre-filter skip rate | 17.1% (7/41) |
| Avg latency, all messages | 2794 ms |
| Avg latency, LLM calls only | 3369 ms |
| LLM calls made | 34 |
| LLM errors | 0 |
| Hard rule downgraded to suggestion | 7 |
| Confusion (TP / FP / FN / TN) | 18 / 0 / 0 / 23 |

## Accuracy by category

| Category | Correct |
|---|---|
| decision | 13/13 |
| false_alarm | 13/13 |
| messy_sarcasm | 3/3 |
| messy_partial | 4/4 |
| messy_hinglish | 5/5 |
| messy_not_in_plan | 3/3 |

## Wrong cases (0)

None.
