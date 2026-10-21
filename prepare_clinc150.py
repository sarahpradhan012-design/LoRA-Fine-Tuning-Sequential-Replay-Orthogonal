"""Prepare the official UCI CLINC150 full release for the in-scope experiment."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "source" / "clinc150_uci" / "data_full.json"
ARCHIVE = ROOT / "data" / "raw" / "clinc150-uci.zip"
OUTPUT = ROOT / "data" / "prepared"
MANIFEST_PATH = ROOT / "data" / "manifest.json"
RANDOM_SEED = 42
DOMAIN_ORDER = [
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
]

# Official CLINC OOS evaluation repository taxonomy, kept inline so the
# experiment manifest and prepared corpus do not depend on a live URL.
DOMAIN_TO_INTENTS = {
    "auto_and_commute": [
        "current_location", "oil_change_when", "oil_change_how", "uber", "traffic",
        "tire_pressure", "schedule_maintenance", "gas", "mpg", "distance",
        "directions", "last_maintenance", "gas_type", "tire_change", "jump_start",
    ],
    "banking": [
        "freeze_account", "routing", "pin_change", "bill_due", "pay_bill",
        "account_blocked", "interest_rate", "min_payment", "bill_balance", "transfer",
        "order_checks", "balance", "spending_history", "transactions", "report_fraud",
    ],
    "credit_cards": [
        "replacement_card_duration", "expiration_date", "damaged_card", "improve_credit_score",
        "report_lost_card", "card_declined", "credit_limit_change", "apr", "redeem_rewards",
        "credit_limit", "rewards_balance", "application_status", "credit_score", "new_card",
        "international_fees",
    ],
    "home": [
        "what_song", "play_music", "todo_list_update", "reminder", "reminder_update",
        "calendar_update", "order_status", "update_playlist", "shopping_list", "calendar",
        "next_song", "order", "todo_list", "shopping_list_update", "smart_home",
    ],
    "kitchen_and_dining": [
        "food_last", "confirm_reservation", "how_busy", "ingredients_list", "calories",
        "nutrition_info", "recipe", "restaurant_reviews", "restaurant_reservation",
        "meal_suggestion", "restaurant_suggestion", "cancel_reservation",
        "ingredient_substitution", "cook_time", "accept_reservations",
    ],
    "meta": [
        "change_speed", "user_name", "whisper_mode", "yes", "change_volume", "no",
        "change_language", "repeat", "change_accent", "cancel", "sync_device",
        "change_user_name", "change_ai_name", "reset_settings", "maybe",
    ],
    "small_talk": [
        "who_made_you", "meaning_of_life", "who_do_you_work_for", "do_you_have_pets",
        "what_are_your_hobbies", "fun_fact", "what_is_your_name", "where_are_you_from",
        "goodbye", "thank_you", "greeting", "tell_joke", "are_you_a_bot", "how_old_are_you",
        "what_can_i_ask_you",
    ],
    "travel": [
        "plug_type", "travel_notification", "translate", "flight_status", "international_visa",
        "timezone", "exchange_rate", "travel_suggestion", "travel_alert", "vaccines",
        "lost_luggage", "book_flight", "book_hotel", "carry_on", "car_rental",
    ],
    "utility": [
        "weather", "alarm", "date", "find_phone", "share_location", "timer", "make_call",
        "calculator", "definition", "measurement_conversion", "flip_coin", "spelling", "time",
        "roll_dice", "text",
    ],
    "work": [
        "pto_request_status", "next_holiday", "insurance_change", "insurance", "meeting_schedule",
        "payday", "taxes", "income", "rollover_401k", "pto_balance", "pto_request", "w2",
        "schedule_meeting", "direct_deposit", "pto_used",
    ],
}

SPLIT_KEYS = {"train": "train", "validation": "val", "test": "test"}
EXPECTED_PER_INTENT = {"train": 100, "validation": 20, "test": 30}
OUTPUT_FILES = {
    "train": "train.jsonl",
    "validation": "validation.jsonl",
    "test": "test.jsonl",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(value: object, split: str) -> list[tuple[str, str]]:
    if not isinstance(value, list):
        raise ValueError(f"Expected '{split}' to be a list")
    rows: list[tuple[str, str]] = []
    for index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError(f"Unexpected {split} row at index {index}: {row!r}")
        text, intent = row
        if not isinstance(text, str) or not isinstance(intent, str):
            raise ValueError(f"Non-string field in {split} row {index}")
        rows.append((text, intent))
    return rows


def validate_domain_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for domain, intents in DOMAIN_TO_INTENTS.items():
        if len(intents) != 15 or len(set(intents)) != 15:
            raise ValueError(f"Domain {domain} must contain 15 unique intents")
        for intent in intents:
            if intent in mapping:
                raise ValueError(f"Intent {intent} is assigned to multiple domains")
            mapping[intent] = domain
    if len(mapping) != 150:
        raise ValueError(f"Expected 150 mapped intents, found {len(mapping)}")
    if set(DOMAIN_TO_INTENTS) != set(DOMAIN_ORDER):
        raise ValueError("Domain list does not match the fixed experiment order")
    return mapping


def validate_replay(train_ids: set[tuple[str, str]], validation_ids: set[tuple[str, str]],
                    test_ids: set[tuple[str, str]]) -> None:
    # Step 2 does not create a replay buffer. If one is later created in this
    # standard location, enforce that every stored example is train-only.
    replay = OUTPUT / "replay.jsonl"
    if not replay.exists():
        return
    replay_ids: set[tuple[str, str]] = set()
    with replay.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            record = json.loads(line)
            item = (record["text"], record["intent"])
            if item in replay_ids:
                raise ValueError(f"Duplicate replay record on line {line_number}")
            replay_ids.add(item)
    if not replay_ids <= train_ids:
        raise ValueError("Replay contains an example not drawn from official training data")
    if replay_ids & validation_ids or replay_ids & test_ids:
        raise ValueError("Validation/test data leaked into replay")


def main() -> None:
    if not SOURCE.is_file() or not ARCHIVE.is_file():
        raise FileNotFoundError("Extract data_full.json and keep the UCI zip in data/raw")
    if not SOURCE.resolve().is_relative_to(ROOT) or not ARCHIVE.resolve().is_relative_to(ROOT):
        raise ValueError("Input paths must remain inside this project")

    intent_to_domain = validate_domain_map()
    label_mapping = {intent: label_id for label_id, intent in enumerate(sorted(intent_to_domain))}
    source_obj = json.loads(SOURCE.read_text(encoding="utf-8"))

    split_rows: dict[str, list[tuple[str, str]]] = {
        output_split: load_rows(source_obj[source_key], output_split)
        for output_split, source_key in SPLIT_KEYS.items()
    }
    in_scope_intents = set(intent_to_domain)
    for split, rows in split_rows.items():
        unknown = sorted({intent for _, intent in rows} - in_scope_intents)
        if unknown:
            raise ValueError(f"Unexpected labels in {split}: {unknown}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, object]] = {}
    identities: dict[str, set[tuple[str, str]]] = {}
    text_identities: dict[str, set[str]] = {}
    intent_counts: dict[str, Counter[str]] = {}
    domain_counts: dict[str, Counter[str]] = {}

    for split, rows in split_rows.items():
        destination = OUTPUT / OUTPUT_FILES[split]
        intent_counts[split] = Counter(intent for _, intent in rows)
        domain_counts[split] = Counter(intent_to_domain[intent] for _, intent in rows)
        identities[split] = set(rows)
        text_identities[split] = {text for text, _ in rows}
        with destination.open("w", encoding="utf-8", newline="\n") as stream:
            for text, intent in rows:
                record = {
                    "text": text,
                    "intent": intent,
                    "domain": intent_to_domain[intent],
                    "label_id": label_mapping[intent],
                    "split": split,
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

        bad_counts = {intent: intent_counts[split].get(intent, 0)
                      for intent in intent_to_domain
                      if intent_counts[split].get(intent, 0) != EXPECTED_PER_INTENT[split]}
        if bad_counts:
            raise ValueError(f"Wrong {split} counts per intent: {bad_counts}")
        if len(rows) != EXPECTED_PER_INTENT[split] * 150:
            raise ValueError(f"Wrong total {split} count: {len(rows)}")
        if len(domain_counts[split]) != 10:
            raise ValueError(f"Expected 10 domains in {split}")
        expected_domain_count = EXPECTED_PER_INTENT[split] * 15
        bad_domain_counts = {
            domain: domain_counts[split].get(domain, 0)
            for domain in DOMAIN_ORDER
            if domain_counts[split].get(domain, 0) != expected_domain_count
        }
        if bad_domain_counts:
            raise ValueError(f"Wrong {split} counts per domain: {bad_domain_counts}")
        summaries[split] = {
            "file": f"data/prepared/{OUTPUT_FILES[split]}",
            "examples": len(rows),
            "examples_per_intent": EXPECTED_PER_INTENT[split],
            "examples_per_domain": {
                domain: domain_counts[split][domain] for domain in DOMAIN_ORDER
            },
        }

    pairs = [("train", "validation"), ("train", "test"), ("validation", "test")]
    cross_split_overlaps: list[dict[str, object]] = []
    for left, right in pairs:
        pair_overlap = identities[left] & identities[right]
        text_overlap = text_identities[left] & text_identities[right]
        if pair_overlap or text_overlap:
            rows_by_text: dict[str, dict[str, list[str]]] = defaultdict(dict)
            for split in (left, right):
                labels_by_text: dict[str, set[str]] = defaultdict(set)
                for text, intent in split_rows[split]:
                    if text in text_overlap:
                        labels_by_text[text].add(intent)
                rows_by_text[split] = {text: sorted(labels) for text, labels in labels_by_text.items()}
            cross_split_overlaps.append({
                "left_split": left,
                "right_split": right,
                "identical_text_and_intent_count": len(pair_overlap),
                "identical_text_count": len(text_overlap),
                "texts_and_labels": {
                    text: {left: rows_by_text[left].get(text, []), right: rows_by_text[right].get(text, [])}
                    for text in sorted(text_overlap)
                },
            })
    validate_replay(identities["train"], identities["validation"], identities["test"])

    manifest = {
        "preparation_status": "prepared_with_source_split_overlap",
        "dataset": "CLINC150 full",
        "uci_dataset_id": 570,
        "uci_download_url": "https://archive.ics.uci.edu/static/public/570/clinc150.zip",
        "version_or_release": "UCI full release; clinc150_uci/data_full.json",
        "domain_mapping_source": "https://raw.githubusercontent.com/clinc/oos-eval/master/data/domains.json",
        "downloaded_date": date.today().isoformat(),
        "downloaded_archive": "data/raw/clinc150-uci.zip",
        "downloaded_archive_sha256": sha256(ARCHIVE),
        "source_split_file": "data/source/clinc150_uci/data_full.json",
        "source_split_keys": {"train": "train", "validation": "val", "test": "test"},
        "processed_split_files": {key: value["file"] for key, value in summaries.items()},
        "oos_handling": "OOS rows are excluded from all processed splits and from replay.",
        "source_overlap_handling": (
            "Official partitions are preserved unchanged. Cross-split duplicate text is reported "
            "below; no rows were silently removed, moved, or relabeled."
        ),
        "target_label_field": "intent (intent name string)",
        "intent_to_label_id": label_mapping,
        "intent_to_domain": {intent: intent_to_domain[intent]
                              for intent in sorted(intent_to_domain)},
        "domain_order_for_first_run": DOMAIN_ORDER,
        "random_seed": RANDOM_SEED,
        "sampling_or_shuffling_performed": False,
        "split_summary": summaries,
        "validation": {
            "intent_count": len(intent_to_domain),
            "each_intent_has_exactly_one_domain": True,
            "intents_per_domain": {domain: len(DOMAIN_TO_INTENTS[domain])
                                    for domain in DOMAIN_ORDER},
            "cross_split_overlaps": cross_split_overlaps,
            "no_exact_example_overlap_between_splits": not any(
                item["identical_text_and_intent_count"] > 0
                for item in cross_split_overlaps
            ),
            "no_cross_split_text_overlap": not any(
                item["identical_text_count"] > 0 for item in cross_split_overlaps
            ),
            "source_overlap_report": cross_split_overlaps,
            "replay_buffer_created": False,
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(MANIFEST_PATH), "splits": summaries,
                      "intent_count": len(intent_to_domain),
                      "domains": len(DOMAIN_TO_INTENTS),
                      "cross_split_overlaps": cross_split_overlaps,
                      "replay_buffer_created": False},
                     indent=2))


if __name__ == "__main__":
    main()
