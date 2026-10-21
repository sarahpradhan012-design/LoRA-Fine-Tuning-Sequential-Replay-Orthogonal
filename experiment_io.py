"""Canonical CLINC150 prompt construction and stage-aware label evaluation."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "data" / "manifest.json"
PROMPT_TEMPLATE = (
    "Classify this user request. Reply with one intent label only.\n"
    "Request: {utterance}"
)

# Keep the domain order in one explicit place so task index has an unambiguous
# meaning. The mapping itself is loaded from the data manifest.
TASK_ORDER = (
    "auto_and_commute",
    "banking",
    "credit_cards",
    "home",
    "kitchen_and_dining",
    "meta",
    "small_talk",
    "travel",
    "utility",
    "work",
)


def _load_intent_to_domain() -> dict[str, str]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    recorded_order = tuple(manifest["domain_order_for_first_run"])
    if recorded_order != TASK_ORDER:
        raise ValueError("Task order differs from data/manifest.json")
    mapping = manifest["intent_to_domain"]
    if len(mapping) != 150 or any(domain not in TASK_ORDER for domain in mapping.values()):
        raise ValueError("Manifest must map all 150 intents to the fixed ten domains")
    return mapping


INTENT_TO_DOMAIN = _load_intent_to_domain()
CANONICAL_INTENTS = frozenset(INTENT_TO_DOMAIN)


def format_prompt(utterance: str) -> str:
    """Return the one shared prompt; no task/domain/label list is inserted."""
    if not isinstance(utterance, str):
        raise TypeError("utterance must be a string")
    return PROMPT_TEMPLATE.format(utterance=utterance)


def format_model_prompt(tokenizer, utterance: str) -> str:
    """Wrap the fixed prompt body in the model tokenizer's user-turn template."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": format_prompt(utterance)}],
        tokenize=False,
        add_generation_prompt=True,
    )


def format_target(intent: str) -> str:
    """Return the canonical intent name used as the training target."""
    if intent not in CANONICAL_INTENTS:
        raise ValueError(f"Unknown canonical intent: {intent!r}")
    return intent


def available_intents(task_index: int) -> frozenset[str]:
    """Return labels learned through task_index (1-based, inclusive)."""
    if isinstance(task_index, bool) or not isinstance(task_index, int):
        raise TypeError("task_index must be an integer")
    if not 1 <= task_index <= len(TASK_ORDER):
        raise ValueError(f"task_index must be between 1 and {len(TASK_ORDER)}")
    seen_domains = set(TASK_ORDER[:task_index])
    return frozenset(
        intent for intent, domain in INTENT_TO_DOMAIN.items() if domain in seen_domains
    )


def normalize_generated_output(output: str) -> str | None:
    """Trim whitespace and trailing Unicode punctuation, preserving underscores."""
    if not isinstance(output, str):
        return None
    candidate = output.strip()
    while candidate:
        # Underscores are Unicode connector punctuation and are part of the
        # canonical intent names, so preserve category Pc while removing other
        # punctuation at the end of a generated answer.
        category = unicodedata.category(candidate[-1])
        if not category.startswith("P") or candidate[-1] == "_":
            break
        candidate = candidate[:-1].rstrip()
    return candidate or None


def accepted_prediction(output: str, task_index: int) -> str | None:
    """Return an exact, stage-available intent; invalid output returns None."""
    candidate = normalize_generated_output(output)
    if candidate is None or candidate not in available_intents(task_index):
        return None
    return candidate


def is_correct_prediction(output: str, target_intent: str, task_index: int) -> bool:
    """Count only exact canonical matches available at this continual stage."""
    allowed = available_intents(task_index)
    prediction = normalize_generated_output(output)
    return (
        prediction is not None
        and target_intent in allowed
        and prediction in allowed
        and prediction == target_intent
    )
