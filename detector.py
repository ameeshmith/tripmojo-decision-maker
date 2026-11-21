"""
mojo-decision-detector

Spots likely plan decisions in group-chat messages that did NOT @mention Mojo,
and turns them into a *suggested* itinerary update. Nothing is ever applied here.

Pipeline for each new message:
    1. pre_filter()  - cheap keyword check, skips obvious chatter (no LLM call)
    2. ask_llm()     - Groq llama-3.3-70b in JSON mode classifies the message
    3. apply_rules() - hard rules in plain Python (confidence threshold,
                       plan validation, "someone else's item" rule)
"""

import json
import logging
import os
import re

from dotenv import load_dotenv
from groq import AuthenticationError, Groq, NotFoundError

load_dotenv()
log = logging.getLogger("detector")

# Override in .env with GROQ_MODEL=... if this model is retired on your account.
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
CONFIDENCE_THRESHOLD = 0.6      # below this we stay silent
AUTO_APPLY_CONFIDENCE = 0.85    # needed before a change is even *eligible* for auto-apply
MAX_HISTORY = 10                # how many recent messages the LLM sees
SHORT_MESSAGE_WORDS = 8         # pre-filter only skips messages this short or shorter

# Words that signal agreement, in English and Hinglish.
AGREEMENT_WORDS = {
    "ok", "okay", "okk", "k", "fine", "yes", "yeah", "yep", "yup", "sure", "done",
    "agreed", "agree", "deal", "perfect", "cool", "works", "sounds", "let's", "lets",
    "final", "fix", "booked", "confirmed",
    "haan", "han", "ha", "theek", "thik", "chalo", "chal", "sahi", "pakka", "bilkul",
}
AGREEMENT_EMOJI = {"👍", "👌", "✅", "🙌"}

# Common words that should not count as "this message mentions the plan".
STOPWORDS = {"from", "with", "the", "and", "trip", "walk", "tour", "day", "at", "in", "to"}

# Counters so we can report how much work the pre-filter saved and what the LLM cost.
stats = {}


def reset_stats():
    stats.update(seen=0, skipped=0, llm_calls=0, prompt_tokens=0, completion_tokens=0)


reset_stats()


# --------------------------------------------------------------------------- #
# Plan helpers
# --------------------------------------------------------------------------- #
def item_value(item):
    """Plan items can be a plain string or {"value": ..., "added_by": ...}."""
    return item["value"] if isinstance(item, dict) else item


def item_owner(item):
    return item.get("added_by") if isinstance(item, dict) else None


def words(text):
    return re.findall(r"[a-z']+", text.lower().replace("’", "'"))


# --------------------------------------------------------------------------- #
# Step 1: pre-filter
# --------------------------------------------------------------------------- #
def plan_keywords(plan):
    """Words that, if they show up in a message, mean it is probably about the plan."""
    keywords = set()
    for key, item in plan.items():
        keywords.update(w for w in words(item_value(item)) if len(w) >= 4 and w not in STOPWORDS)
        # "day_2" -> also match "day 2" / "day2"
        keywords.add(key.replace("_", " "))
        keywords.add(key.replace("_", ""))
    return keywords


def pre_filter(newest_text, plan):
    """
    Return (should_call_llm, reason).
    Only skips SHORT messages that have NO agreement word and NO plan reference.
    Anything uncertain goes to the LLM: a missed decision is worse than a wasted call.
    """
    tokens = words(newest_text)
    lowered = " ".join(tokens)

    has_agreement = any(t in AGREEMENT_WORDS for t in tokens) or any(e in newest_text for e in AGREEMENT_EMOJI)
    mentions_plan = any(k in lowered if " " in k else k in tokens for k in plan_keywords(plan))
    is_short = len(tokens) <= SHORT_MESSAGE_WORDS

    if is_short and not has_agreement and not mentions_plan:
        return False, "pre-filter: short, no agreement word, no plan reference"
    return True, "pre-filter: passed"


# --------------------------------------------------------------------------- #
# Step 2: LLM
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """You watch a travel group chat. Decide if the NEWEST message completes a
decision to change the shared trip itinerary (the plan).

A decision means: someone proposed a change to the itinerary, and the newest message
agrees to it or confirms it (e.g. "fine", "ok let's do that", "haan theek hai", "chalo", "👍").
Messages may be in English, Hindi or Hinglish.

NOT a decision:
- "fine"/"ok" about something unrelated to the itinerary ("fine, I'll check the weather", "ok I'm boarding")
- sarcasm or mock agreement ("yeah sure, can't wait to wake up at 4 🙄")
- a new proposal or a question that nobody has agreed to yet
- agreement while someone, especially whoever added the item, has objected and it is unresolved
- conditional or hesitant replies ("ok but let me check", "hmm not sure")
- decisions about money, packing, logistics or anything that is not an itinerary item
- excitement about the existing plan without any change

target_item: the plan key being changed. If the decision ADDS a new activity that is not
in the plan, use null. new_value: short description of the new itinerary item.

Reply with JSON only, exactly these keys:
{"is_decision": true|false, "confidence": 0.0-1.0, "target_item": "<plan key>"|null,
 "new_value": "<string>"|null, "reason": "<one short sentence>"}"""

_client = None


def get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is missing. Put it in a .env file.")
        _client = Groq(api_key=api_key, max_retries=5)  # retries handle free-tier rate limits
    return _client


def build_user_prompt(plan, history, newest):
    plan_lines = []
    for key, item in plan.items():
        owner = item_owner(item)
        plan_lines.append(f"- {key}: {item_value(item)}" + (f" (added by {owner})" if owner else ""))
    chat_lines = [f"{m['sender']}: {m['text']}" for m in history[-MAX_HISTORY:]]
    return (
        "PLAN:\n" + "\n".join(plan_lines)
        + "\n\nRECENT CHAT (oldest first):\n" + ("\n".join(chat_lines) or "(none)")
        + f"\n\nNEWEST MESSAGE:\n{newest['sender']}: {newest['text']}"
    )


def ask_llm(plan, history, newest):
    response = get_client().chat.completions.create(
        model=MODEL,
        temperature=0,
        response_format={"type": "json_object"},   # Groq JSON mode: output is valid JSON
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(plan, history, newest)},
        ],
    )
    if response.usage:  # token counts let eval.py compute real cost per call
        stats["llm_calls"] += 1
        stats["prompt_tokens"] += response.usage.prompt_tokens
        stats["completion_tokens"] += response.usage.completion_tokens
    return json.loads(response.choices[0].message.content)


# --------------------------------------------------------------------------- #
# Step 3: hard rules (code, not the LLM)
# --------------------------------------------------------------------------- #
def apply_rules(raw, plan, newest):
    """Turn the LLM's raw opinion into the final, validated result."""
    try:
        confidence = max(0.0, min(1.0, float(raw.get("confidence", 0))))
    except (TypeError, ValueError):
        confidence = 0.0
    target = raw.get("target_item")
    if target not in plan:          # LLM invented a key -> treat as "new item"
        target = None

    result = {
        "is_decision": bool(raw.get("is_decision")) and confidence >= CONFIDENCE_THRESHOLD,
        "confidence": round(confidence, 2),
        "target_item": target,
        # old_value comes from the plan itself, never from the LLM
        "old_value": item_value(plan[target]) if target else None,
        "new_value": raw.get("new_value"),
        "reason": str(raw.get("reason", "")),
        "action": "ignore",
        "hard_rule": None,
    }
    if not result["is_decision"]:
        result.update(target_item=None, old_value=None, new_value=None)
        return result

    # HARD RULE: replacing an item that another member added is ALWAYS a suggestion.
    # "Another member" = the owner is not the person whose message closed the decision.
    owner = item_owner(plan[target]) if target else None
    if owner and owner != newest["sender"]:
        result["action"] = "suggest"
        result["hard_rule"] = f"item added by {owner}; {owner} must confirm"
    elif target and confidence >= AUTO_APPLY_CONFIDENCE:
        # Owner themself agreed with high confidence. Still NOT applied by this
        # prototype - it is only flagged as safe to offer one-tap apply.
        result["action"] = "auto_apply_eligible"
    else:
        result["action"] = "suggest"
    return result


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def not_a_decision(reason, prefiltered=False):
    return {
        "is_decision": False, "confidence": 0.0, "target_item": None,
        "old_value": None, "new_value": None, "reason": reason,
        "action": "ignore", "hard_rule": None, "prefiltered": prefiltered,
    }


def detect(plan, history, newest):
    """
    plan:    {"day_2": "Trek in Yuksom"} or {"day_2": {"value": ..., "added_by": ...}}
    history: [{"sender": "Rahul", "text": "..."}, ...]  (oldest first)
    newest:  {"sender": "Priya", "text": "fine"}
    """
    stats["seen"] += 1

    should_call, why = pre_filter(newest["text"], plan)
    if not should_call:
        stats["skipped"] += 1
        log.info("skipped %r (%d/%d skipped so far)", newest["text"], stats["skipped"], stats["seen"])
        return not_a_decision(why, prefiltered=True)

    try:
        raw = ask_llm(plan, history, newest)
    except (NotFoundError, AuthenticationError):
        raise  # wrong model name or bad key: a setup problem, so stop instead of guessing
    except Exception as e:  # network error, bad JSON, rate limit after retries...
        log.warning("LLM call failed: %s", e)
        return not_a_decision(f"llm_error: {e}")

    result = apply_rules(raw, plan, newest)
    result["prefiltered"] = False
    return result


if __name__ == "__main__":
    # Quick demo: python detector.py
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    demo_plan = {"day_2": {"value": "Trek in Yuksom", "added_by": "Priya"}}
    demo_history = [
        {"sender": "Rahul", "text": "trek after that long drive sounds brutal"},
        {"sender": "Rahul", "text": "no need, let's do a monastery instead. Rumtek?"},
    ]
    print(json.dumps(detect(demo_plan, demo_history, {"sender": "Ananya", "text": "fine"}), indent=2))
