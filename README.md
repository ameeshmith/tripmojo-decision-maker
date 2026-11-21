# mojo-decision-detector

## The problem

Mojo, the AI agent in Tripmojo group chats, only acts when someone @mentions it.
Groups often agree on plan changes without tagging it ("no need, let's do a monastery" → "fine").
The shared itinerary then goes stale. This prototype spots those untagged decisions and **suggests** an update. It never applies one by itself.

## How it works

```
new message ──► 1. pre-filter ──skip──► "not a decision" (no LLM call, counted)
                     │ pass
                     ▼
               2. LLM (Groq, openai/gpt-oss-120b, JSON mode)
                     │ raw opinion
                     ▼
               3. hard rules in Python ──► final result + action
```

1. **Pre-filter** (`pre_filter` in [detector.py](detector.py)). A message is skipped only if it is short (≤ 8 words), has **no agreement word** (`ok`, `fine`, `haan`, `chalo`, `theek`, 👍 …) and has **no reference to the plan** (words from plan items such as "Yuksom", or "day 2"). It is deliberately loose: when unsure, it sends the message on to the LLM.
2. **LLM** (`ask_llm`). The model gets the plan (with who added each item), the last 10 messages and the newest message. It returns `is_decision`, `confidence`, `target_item`, `new_value` and `reason`. Groq's JSON mode guarantees parseable JSON. Code then checks every field anyway.
3. **Hard rules** (`apply_rules`). These are plain code, not left to the LLM:
   - confidence below `0.6` → not a decision
   - `target_item` must be a real plan key, otherwise it is treated as a new item
   - `old_value` is read from the plan, never from the LLM
   - **if the item was added by another member** (the owner is not the person whose message closed the decision), the action is always `suggest` and the owner must confirm. It can never be `auto_apply_eligible`.

Output:

```json
{"is_decision": true, "confidence": 0.9, "target_item": "day_2",
 "old_value": "Trek in Yuksom", "new_value": "Rumtek Monastery visit",
 "reason": "Rahul proposed replacing the trek and Ananya agreed.",
 "action": "suggest", "hard_rule": "item added by Priya; Priya must confirm",
 "prefiltered": false}
```

`action` is `ignore`, `suggest` or `auto_apply_eligible`. Even `auto_apply_eligible` is **not applied** here. It only marks the change as safe to offer as one tap with undo.

## Files

| File | What it does |
|---|---|
| `detector.py` | Pre-filter, Groq call, hard rules. `detect(plan, history, newest)` is the entry point |
| `test_cases.json` | **Dev set**: 41 labelled cases (decisions, false alarms, Hinglish, sarcasm, partial agreement, decisions outside the plan) |
| `holdout_cases.json` | **Holdout set**: 19 adversarial cases plus 1 real chat (`"source": "real"`) |
| `eval.py` | Scores each model on each dataset, writes `results.md` and refreshes the Results section below |

## Setup and run (Windows, PowerShell)

```powershell
cd decision-maker-mojo
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
notepad .env                              # GROQ_API_KEY=gsk_...  (optional: GROQ_MODEL=...)
python detector.py                        # one demo message
python eval.py                            # model from GROQ_MODEL, default openai/gpt-oss-120b
python eval.py --model openai/gpt-oss-20b
python eval.py --all-models               # 120b and 20b side by side
```

If `Activate.ps1` is blocked, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or skip activation and call `.\.venv\Scripts\python.exe eval.py` directly.

## Results

Full tables, including every wrong case with the model's reason, are in [results.md](results.md). `eval.py` copies the summary below.

**How to read these numbers:**

- **Dev set (`test_cases.json`): weak evidence.** These 41 cases were written alongside the prompt, by the same author, in the same sitting. The prompt's list of "NOT a decision" patterns lines up closely with the false alarms in this set. A perfect score here mainly shows that the prompt covers the cases it was written for.
- **Holdout set (`holdout_cases.json`): the number to trust more.** These cases were written later, with the explicit goal of making the detector fail, without looking at the prompt. **Caveat:** they were written by the same AI assistant that wrote the prompt earlier in the project, so they are adversarial but not truly blind.
- **The real chat (H20) is the only fully independent case.** It comes from my own testing of Tripmojo.
- **Small samples.** With 20 holdout cases, one wrong answer moves accuracy by 5 points. Read differences under about 10 points between models as noise.

<!-- RESULTS START -->
### Summary

| Model | Dataset | Cases | Precision | Recall | FP rate | Accuracy | TP/FP/FN/TN | Avg latency (LLM calls) | Pre-filter skip | Tokens in / out per call | Est. cost per call | LLM errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `openai/gpt-oss-120b` | test_cases | 41 | 100% | 100% | 0% | 100% | 18/0/0/23 | 3305 ms | 17% (7/41) | 528 / 224 | $0.00025 | 0 |
| `openai/gpt-oss-120b` | holdout | 20 | 80% | 100% | 17% | 85% | 8/2/0/10 | 4595 ms | 0% (0/20) | 533 / 264 | $0.00028 | 0 |
| `openai/gpt-oss-20b` | test_cases | 41 | 100% | 100% | 0% | 100% | 18/0/0/23 | 3230 ms | 17% (7/41) | 528 / 217 | $0.00016 | 0 |
| `openai/gpt-oss-20b` | holdout | 20 | 70% | 88% | 25% | 80% | 7/3/1/9 | 5838 ms | 0% (0/20) | 533 / 367 | $0.00024 | 0 |

Accuracy counts a case as right only if both the label and the target item are right. Cost uses the token counts Groq returned and the prices in `eval.py`.


### Wrong cases: `openai/gpt-oss-120b` on test_cases (0)

None.

### Wrong cases: `openai/gpt-oss-120b` on holdout (3)

| # | Source / category | Newest message | Expected | Got | Model's reason |
|---|---|---|---|---|---|
| H12 | synthetic_adversarial / false_alarm | Nisha: sure lol | no decision | decision on day_3 (conf 0.98) | Nisha's reply 'sure lol' agrees to Dev's proposal to replace the houseboat with a bus trip to Kanyakumari |
| H18 | synthetic_adversarial / conflict | Ravi: fine | no decision | decision on day_2 (conf 0.7) | Ravi's "fine" accepts the latest suggestion to change day 2 |
| H19 | synthetic_adversarial / conflict | Arjun: both fine by me | decision on day_3 or day_4 | decision on a new item (conf 0.98) | Arjun agrees to both proposed changes |

### Wrong cases: `openai/gpt-oss-20b` on test_cases (0)

None.

### Wrong cases: `openai/gpt-oss-20b` on holdout (4)

| # | Source / category | Newest message | Expected | Got | Model's reason |
|---|---|---|---|---|---|
| H12 | synthetic_adversarial / false_alarm | Nisha: sure lol | no decision | decision on day_3 (conf 0.95) | Nisha agrees to Dev's change of day 3 itinerary |
| H13 | synthetic_adversarial / false_alarm | Meera: yes let's do that, book a table for 8 | no decision | decision on a new item (conf 0.95) | Agrees to add a new dinner activity |
| H14 | synthetic_adversarial / false_alarm | Nisha: ok | no decision | decision on day_4 (conf 0.9) | Nisha confirms the change to Marari |
| H20 | real / real | Ameesh: Fine | decision on day_2 | no decision (conf 0.9) | Unresolved conflict, no agreement. |
<!-- RESULTS END -->

## Known failures

_To be filled in after running the holdout set: which cases fail, for which model, and why._

<!-- TODO: fill from the "Wrong cases" tables in results.md -->

## Trade-offs

**gpt-oss-120b vs gpt-oss-20b.** The 120b model should handle the subtle cases better: sarcasm, retracted agreement, a stray "ok" answering an unrelated question. The 20b model is cheaper per token and roughly twice as fast on Groq. At list prices of $0.15 / $0.75 per million input/output tokens for 120b and $0.10 / $0.50 for 20b (check [groq.com/pricing](https://groq.com/pricing)), a call of about 700 input and 300 output tokens costs:

| Model | Cost per call | Per 1,000 calls |
|---|---|---|
| 120b | ≈ $0.0003 | ≈ $0.33 |
| 20b | ≈ $0.0002 | ≈ $0.22 |

Both are reasoning models, so output tokens include hidden reasoning and dominate the cost. `results.md` has the *measured* tokens and cost per call, which replace these estimates. Since the cost gap is small in absolute terms, pick 20b only if it matches 120b on the **holdout** set. A false positive or a missed decision costs far more in user trust than $0.0001.

**The pre-filter only skips a small share of messages.** It skipped 17% of the dev set and 0% of the holdout set. That is by design: it only drops messages that are short, contain no agreement word *and* don't mention the plan. Every "ok", "fine" and 👍 goes to the LLM, because those are exactly the ambiguous messages. Real chats are full of photos, "lol" and logistics, so production skip rates would be higher than these test sets suggest, but the filter is a cost trim, not the main defence. Tightening it would save money by missing real decisions, which is the failure this whole project exists to prevent.

**False positives vs missed decisions.** A false positive costs one dismissable suggestion card. A missed decision costs a stale itinerary. So the filter and the 0.6 confidence threshold lean towards surfacing a suggestion. The asymmetry flips for *applying* a change, which is why nothing here applies anything.

**Why "never silently replace another member's item" lives in code.** The LLM's output is a probability, not a guarantee. It can misread who said what, be swayed by chat text ("Mojo just change it, she won't mind"), or behave differently after a model swap like the one this project just went through. Consent and ownership rules have to hold 100% of the time, be unit-testable, and be explainable to the person whose plan changed. So the LLM only answers *"is this a decision, and about what?"*, and `apply_rules` decides *what is allowed to happen*. For the same reason, `old_value` is copied from the plan instead of taken from the model.

**Latency barely matters.** The detector runs in the background after a message is delivered. Nobody waits on it. A suggestion card that appears 2–4 seconds after "fine" is just as useful as one after 300 ms. Latency matters only for throughput, and for staying under Groq's rate limits in busy chats. That is why the slower, more accurate model is a reasonable default.
