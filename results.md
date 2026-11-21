# Results

Confidence threshold: 0.6

## Summary

| Model | Dataset | Cases | Precision | Recall | FP rate | Accuracy | TP/FP/FN/TN | Avg latency (LLM calls) | Pre-filter skip | Tokens in / out per call | Est. cost per call | LLM errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `openai/gpt-oss-120b` | test_cases | 41 | 100% | 100% | 0% | 100% | 18/0/0/23 | 3305 ms | 17% (7/41) | 528 / 224 | $0.00025 | 0 |
| `openai/gpt-oss-120b` | holdout | 20 | 80% | 100% | 17% | 85% | 8/2/0/10 | 4595 ms | 0% (0/20) | 533 / 264 | $0.00028 | 0 |
| `openai/gpt-oss-20b` | test_cases | 41 | 100% | 100% | 0% | 100% | 18/0/0/23 | 3230 ms | 17% (7/41) | 528 / 217 | $0.00016 | 0 |
| `openai/gpt-oss-20b` | holdout | 20 | 70% | 88% | 25% | 80% | 7/3/1/9 | 5838 ms | 0% (0/20) | 533 / 367 | $0.00024 | 0 |

Accuracy counts a case as right only if both the label and the target item are right. Cost uses the token counts Groq returned and the prices in `eval.py`.


## Wrong cases: `openai/gpt-oss-120b` on test_cases (0)

None.

## Wrong cases: `openai/gpt-oss-120b` on holdout (3)

| # | Source / category | Newest message | Expected | Got | Model's reason |
|---|---|---|---|---|---|
| H12 | synthetic_adversarial / false_alarm | Nisha: sure lol | no decision | decision on day_3 (conf 0.98) | Nisha's reply 'sure lol' agrees to Dev's proposal to replace the houseboat with a bus trip to Kanyakumari |
| H18 | synthetic_adversarial / conflict | Ravi: fine | no decision | decision on day_2 (conf 0.7) | Ravi's "fine" accepts the latest suggestion to change day 2 |
| H19 | synthetic_adversarial / conflict | Arjun: both fine by me | decision on day_3 or day_4 | decision on a new item (conf 0.98) | Arjun agrees to both proposed changes |

## Wrong cases: `openai/gpt-oss-20b` on test_cases (0)

None.

## Wrong cases: `openai/gpt-oss-20b` on holdout (4)

| # | Source / category | Newest message | Expected | Got | Model's reason |
|---|---|---|---|---|---|
| H12 | synthetic_adversarial / false_alarm | Nisha: sure lol | no decision | decision on day_3 (conf 0.95) | Nisha agrees to Dev's change of day 3 itinerary |
| H13 | synthetic_adversarial / false_alarm | Meera: yes let's do that, book a table for 8 | no decision | decision on a new item (conf 0.95) | Agrees to add a new dinner activity |
| H14 | synthetic_adversarial / false_alarm | Nisha: ok | no decision | decision on day_4 (conf 0.9) | Nisha confirms the change to Marari |
| H20 | real / real | Ameesh: Fine | decision on day_2 | no decision (conf 0.9) | Unresolved conflict, no agreement. |
