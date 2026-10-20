"""Evidence-first dark-pattern signals for the DarkLens Streamlit app.

These phrase rules flag possible indicators only. They do not establish intent,
prove that a claim is false, or replace human review.
"""
from __future__ import annotations

import re
from typing import Dict, List

PATTERNS = {
    "Scarcity claims": {
        "description": "The text says that stock or availability may be running out.",
        "patterns": [
            r"\bonly\s+\d+\s+left\b", r"\bonly\s+a\s+few\s+left\b",
            r"\blimited\s+stock\b", r"\blow\s+stock\b", r"\bfew\s+left\b",
            r"\balmost\s+sold\s+out\b", r"\brunning\s+out\b", r"\blast\s+\d+\s+items?\b",
        ],
    },
    "Urgency pressure": {
        "description": "The text encourages the shopper to act quickly.",
        "patterns": [
            r"\bhurry\b", r"\bact now\b", r"\bbuy now\b", r"\border now\b",
            r"\blimited[- ]time offer\b", r"\bends? in\s+\d+\s*(?:minutes?|hours?|days?)\b",
            r"\bending soon\b", r"\btoday only\b", r"\bdon['’]?t miss out\b",
            r"\bexpires? (?:today|soon|in\b)", r"\bcountdown\b",
        ],
    },
    "Social-proof claims": {
        "description": "The text uses popularity or other shoppers to influence a decision.",
        "patterns": [
            r"\b\d+\s+people\s+(?:are\s+)?viewing\b", r"\b\d+\s+people bought\b",
            r"\bpopular choice\b", r"\bbestseller\b", r"\bbest seller\b",
            r"\btrending now\b", r"\bcustomers also bought\b", r"\bpeople are viewing\b",
        ],
    },
    "Discount pressure": {
        "description": "The text highlights a deal or saving.",
        "patterns": [
            r"\b\d{1,3}\s*%\s*off\b", r"\bdiscount\b", r"\bflash sale\b",
            r"\bspecial offer\b", r"\blimited offer\b", r"\bsave\s+₹?\s*\d+\b",
            r"\bcoupon code\b", r"\bdeal ends\b", r"\bwas\s+₹?\s*\d+\s*(?:now|today)\b",
        ],
    },
    "Possible extra-cost clues": {
        "description": "The text mentions an extra fee or charge.",
        "patterns": [
            r"\bhandling fee\b", r"\bplatform fee\b", r"\bconvenience fee\b",
            r"\bservice fee\b", r"\badditional charges?\b", r"\bextra charges?\b",
            r"\bshipping calculated at checkout\b", r"\bfees may apply\b",
        ],
    },
    "Possible add-on / sneaking clues": {
        "description": "The text mentions an extra product or service.",
        "patterns": [
            r"\badded to your cart\b", r"\badd protection plan\b", r"\bextended warranty\b",
            r"\badd-on\b", r"\badd a donation\b", r"\bprotect your purchase\b",
            r"\bpre[- ]selected\b", r"\bautomatically added\b",
        ],
    },
    "Subscription / renewal clues": {
        "description": "The text mentions a subscription, trial, or renewal.",
        "patterns": [
            r"\bauto(?:matic)? renewal\b", r"\brecurring payment\b", r"\bmonthly subscription\b",
            r"\bfree trial\b", r"\bcharged every month\b", r"\brenews? automatically\b",
            r"\bcancel anytime\b", r"\bcancellation fee\b", r"\bsubscription\b",
        ],
    },
}

COMPILED = {
    category: [re.compile(p, re.IGNORECASE) for p in details["patterns"]]
    for category, details in PATTERNS.items()
}


def detect_signals(text: str) -> Dict[str, dict]:
    """Return matching categories and short text snippets as supporting evidence."""
    text = text or ""
    results: Dict[str, dict] = {}
    for category, regexes in COMPILED.items():
        matches: List[str] = []
        seen = set()
        for regex in regexes:
            for match in regex.finditer(text):
                start, end = match.span()
                left = max(0, text.rfind(".", 0, start) + 1, text.rfind("\n", 0, start) + 1)
                stops = [x for x in (text.find(".", end), text.find("\n", end), text.find("!", end), text.find("?", end)) if x >= 0]
                right = min(stops) + 1 if stops else min(len(text), end + 100)
                snippet = re.sub(r"\s+", " ", text[left:right]).strip()
                if not snippet:
                    snippet = text[max(0, start - 35):min(len(text), end + 65)].strip()
                key = snippet.casefold()
                if key not in seen:
                    seen.add(key)
                    matches.append(snippet[:240])
                if len(matches) >= 5:
                    break
            if len(matches) >= 5:
                break
        if matches:
            results[category] = {
                "description": PATTERNS[category]["description"],
                "evidence": matches,
            }
    return results


def classify_text(text: str, model, vectorizer):
    """Return the saved model's predicted label, or None if unavailable."""
    if model is None or vectorizer is None or not (text or "").strip():
        return None
    try:
        return str(model.predict(vectorizer.transform([text]))[0])
    except Exception:
        return None


def compare_observations(first_text: str, second_text: str) -> dict:
    """Compare two user-captured page observations for changes in pressure wording.

    This is a transparent text comparison, not proof that a timer reset or a claim
    is false. The user must capture both observations from the same page.
    """
    import difflib

    first = (first_text or "").strip()
    second = (second_text or "").strip()
    if not first or not second:
        return {"status": "incomplete", "changes": [], "countdown_changes": [], "similarity": None}

    first_norm = re.sub(r"\s+", " ", first).strip()
    second_norm = re.sub(r"\s+", " ", second).strip()
    similarity = round(difflib.SequenceMatcher(None, first_norm.casefold(), second_norm.casefold()).ratio() * 100, 1)

    countdown_re = re.compile(
        r"(?:ends?|ending|expires?|remaining|left|in)\s*[:：-]?\s*"
        r"(\d{1,2}\s*(?:seconds?|secs?|minutes?|mins?|hours?|hrs?|days?))",
        re.IGNORECASE,
    )
    first_times = [m.group(1).lower().replace(" ", "") for m in countdown_re.finditer(first)]
    second_times = [m.group(1).lower().replace(" ", "") for m in countdown_re.finditer(second)]

    changes = []
    if first_norm.casefold() != second_norm.casefold():
        matcher = difflib.SequenceMatcher(None, first_norm, second_norm)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "equal":
                old = first_norm[i1:i2].strip()
                new = second_norm[j1:j2].strip()
                if old or new:
                    changes.append({"type": tag, "before": old[:180], "after": new[:180]})
            if len(changes) >= 8:
                break

    countdown_changed = bool(first_times and second_times and first_times != second_times)
    return {
        "status": "compared",
        "changes": changes,
        "countdown_before": first_times,
        "countdown_after": second_times,
        "countdown_changed": countdown_changed,
        "similarity": similarity,
        "identical": first_norm.casefold() == second_norm.casefold(),
    }
