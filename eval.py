"""
Score the detector on two datasets, for one or more models.

    python eval.py                      # model from GROQ_MODEL in .env
    python eval.py --model openai/gpt-oss-20b
    python eval.py --all-models         # gpt-oss-120b and gpt-oss-20b side by side

Datasets are reported separately:
  test_cases.json     41 cases written alongside the prompt (development set)
  holdout_cases.json  20 adversarial cases + 1 real chat (held-out set)

Writes results.md and refreshes the Results section of README.md.
"""

import argparse
import json
import logging
import sys
import time

from groq import AuthenticationError, NotFoundError

import detector

DATASETS = [("test_cases", "test_cases.json"), ("holdout", "holdout_cases.json")]
ALL_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]

# USD per 1M tokens (input, output). Groq list prices at time of writing:
# check https://groq.com/pricing before quoting them.
PRICES = {
    "openai/gpt-oss-120b": (0.15, 0.75),
    "openai/gpt-oss-20b": (0.10, 0.50),
}

README_START = "<!-- RESULTS START -->"
README_END = "<!-- RESULTS END -->"


# --------------------------------------------------------------------------- #
# Scoring helpers
# --------------------------------------------------------------------------- #
def safe_div(a, b):
    return a / b if b else 0.0


def pct(x):
    return f"{x * 100:.0f}%"


def allowed_targets(expected):
    """expected target_item may be one key, null, or a list of acceptable keys."""
    t = expected["target_item"]
    return t if isinstance(t, list) else [t]


def describe(is_decision, target):
    if not is_decision:
        return "no decision"
    if isinstance(target, list):
        target = " or ".join(target)
    return f"decision on {target or 'a new item'}"


def judge(expected, got):
    """None if correct, otherwise a short label for the mistake."""
    if expected["is_decision"] and not got["is_decision"]:
        return "missed decision"
    if not expected["is_decision"] and got["is_decision"]:
        return "false positive"
    if expected["is_decision"] and got["target_item"] not in allowed_targets(expected):
        return "wrong target"
    return None


# --------------------------------------------------------------------------- #
# Run one model on one dataset
# --------------------------------------------------------------------------- #
def evaluate(model, dataset_name, path):
    data = json.load(open(path, encoding="utf-8"))
    plans, cases = data["plans"], data["cases"]

    detector.MODEL = model      # detector.ask_llm reads this at call time
    detector.reset_stats()

    tp = fp = fn = tn = 0
    latencies, llm_latencies, wrong = [], [], []
    llm_errors = 0

    print(f"\n=== {model} on {dataset_name} ({len(cases)} cases) ===")
    for case in cases:
        start = time.perf_counter()
        got = detector.detect(plans[case["plan"]], case["history"], case["newest"])
        ms = (time.perf_counter() - start) * 1000
        latencies.append(ms)
        if not got["prefiltered"]:
            llm_latencies.append(ms)
        if got["reason"].startswith("llm_error"):
            llm_errors += 1

        exp = case["expected"]
        if exp["is_decision"]:
            tp, fn = (tp + 1, fn) if got["is_decision"] else (tp, fn + 1)
        else:
            fp, tn = (fp + 1, tn) if got["is_decision"] else (fp, tn + 1)

        error = judge(exp, got)
        print(f"[{'OK ' if error is None else 'BAD'}] {str(case['id']):>4} "
              f"{case['newest']['text'][:45]:<45} -> {got['action']:<20} {ms:6.0f} ms")
        if error:
            wrong.append({"case": case, "error": error, "got": got})

    s = detector.stats
    calls = s["llm_calls"]
    tokens_in = safe_div(s["prompt_tokens"], calls)
    tokens_out = safe_div(s["completion_tokens"], calls)
    price_in, price_out = PRICES.get(model, (0, 0))
    n = len(cases)
    return {
        "model": model, "dataset": dataset_name, "n": n,
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fn),
        "fpr": safe_div(fp, fp + tn),
        "accuracy": safe_div(n - len(wrong), n),
        "confusion": f"{tp}/{fp}/{fn}/{tn}",
        "latency_all": safe_div(sum(latencies), len(latencies)),
        "latency_llm": safe_div(sum(llm_latencies), len(llm_latencies)),
        "skip_rate": safe_div(s["skipped"], n),
        "skipped": s["skipped"],
        "llm_errors": llm_errors,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_per_call": (tokens_in * price_in + tokens_out * price_out) / 1_000_000,
        "wrong": wrong,
    }


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def cell(text):
    return str(text).replace("|", "/").replace("\n", " ")


def build_report(runs):
    lines = [
        "## Summary", "",
        "| Model | Dataset | Cases | Precision | Recall | FP rate | Accuracy | TP/FP/FN/TN "
        "| Avg latency (LLM calls) | Pre-filter skip | Tokens in / out per call | Est. cost per call | LLM errors |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        lines.append(
            f"| `{r['model']}` | {r['dataset']} | {r['n']} | {pct(r['precision'])} | {pct(r['recall'])} "
            f"| {pct(r['fpr'])} | {pct(r['accuracy'])} | {r['confusion']} | {r['latency_llm']:.0f} ms "
            f"| {pct(r['skip_rate'])} ({r['skipped']}/{r['n']}) | {r['tokens_in']:.0f} / {r['tokens_out']:.0f} "
            f"| ${r['cost_per_call']:.5f} | {r['llm_errors']} |"
        )
    lines += ["", "Accuracy counts a case as right only if both the label and the target item are right. "
              "Cost uses the token counts Groq returned and the prices in `eval.py`.", ""]

    for r in runs:
        lines += ["", f"## Wrong cases: `{r['model']}` on {r['dataset']} ({len(r['wrong'])})", ""]
        if not r["wrong"]:
            lines.append("None.")
            continue
        lines += ["| # | Source / category | Newest message | Expected | Got | Model's reason |",
                  "|---|---|---|---|---|---|"]
        for w in r["wrong"]:
            case, got = w["case"], w["got"]
            source = f"{case.get('source', 'dev')} / {case['category']}"
            msg = f"{case['newest']['sender']}: {case['newest']['text']}"
            expected = describe(case["expected"]["is_decision"], case["expected"]["target_item"])
            got_text = f"{describe(got['is_decision'], got['target_item'])} (conf {got['confidence']})"
            lines.append(f"| {case['id']} | {cell(source)} | {cell(msg)} | {expected} "
                         f"| {cell(got_text)} | {cell(got['reason'])} |")
    return "\n".join(lines) + "\n"


def update_readme(report):
    """Replace the text between the RESULTS markers in README.md."""
    try:
        text = open("README.md", encoding="utf-8").read()
    except FileNotFoundError:
        return
    if README_START not in text or README_END not in text:
        return
    before, rest = text.split(README_START, 1)
    after = rest.split(README_END, 1)[1]
    nested = report.replace("## ", "### ")  # sit under the README's own heading
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(f"{before}{README_START}\n{nested}{README_END}{after}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Evaluate the Mojo decision detector.")
    parser.add_argument("--model", default=detector.MODEL,
                        help=f"Groq model id (default from GROQ_MODEL in .env: {detector.MODEL})")
    parser.add_argument("--all-models", action="store_true",
                        help="run " + " and ".join(ALL_MODELS))
    args = parser.parse_args()
    models = ALL_MODELS if args.all_models else [args.model]

    runs = []
    try:
        for model in models:
            for name, path in DATASETS:
                runs.append(evaluate(model, name, path))
    except (NotFoundError, AuthenticationError) as e:
        print(f"\nStopped: {e}\nCheck GROQ_API_KEY and the model name. Nothing was written.")
        sys.exit(1)

    report = build_report(runs)
    print("\n" + report)
    with open("results.md", "w", encoding="utf-8") as f:
        f.write("# Results\n\n" f"Confidence threshold: {detector.CONFIDENCE_THRESHOLD}\n\n" + report)
    update_readme(report)
    print("Saved results.md and updated README.md")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles choke on emoji otherwise
    sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="  %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # hide per-request HTTP logs
    main()
