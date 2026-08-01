"""Build a fine-tune dataset (OpenAI chat JSONL) from processed sessions.

Every clean lap of every driver becomes training examples across the three
tasks, with the deterministic rule-based ground truth from f1coach.features as
the assistant target. Output format is portable: works with OpenAI-style
hosted fine-tuning, Unsloth/HF SFT (via `standardize_sharegpt`-style mapping),
Together, etc. Eval-set laps are excluded to avoid train/eval leakage.

Usage: uv run scripts/build_finetune_dataset.py
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from f1coach.config import EVAL_DIR, FINETUNE_DIR, PROCESSED_DIR
from f1coach.features import coaching_ground_truth, label_risks
from f1coach.prompts import COACH_SYSTEM, RISK_SYSTEM, coach_user, risk_user

rng = random.Random(44)  # deterministic sampling; 44 for the champ


def clean(laps):
    return [l for l in laps if l.get("lap_time_s") and l.get("is_accurate")
            and not l.get("is_pit_in_lap") and not l.get("is_pit_out_lap")]


def eval_lap_keys() -> set[tuple]:
    """(year, event, driver, lap_number) used by the eval set — exclude from training."""
    keys = set()
    path = EVAL_DIR / "eval_set.jsonl"
    if not path.exists():
        return keys
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(json.loads(line)["user"])
        lap = payload.get("current_lap", {})
        keys.add((lap.get("year"), lap.get("event"), lap.get("driver"), lap.get("lap_number")))
    return keys


def examples_for_session(session: dict, excluded: set[tuple]) -> list[dict]:
    laps = session["laps"]
    out = []
    drivers = sorted({l["driver"] for l in laps})

    session_best = None
    for lap in clean(laps):
        if session_best is None or lap["lap_time_s"] < session_best["lap_time_s"]:
            session_best = lap

    for drv in drivers:
        d_laps = sorted((l for l in laps if l["driver"] == drv), key=lambda l: l["lap_number"])
        d_clean = clean(d_laps)
        if len(d_clean) < 6:
            continue

        # risk: sample up to 4 laps spread through the race
        for cur in rng.sample(d_clean[3:], min(4, max(0, len(d_clean) - 3))):
            key = (cur["year"], cur["event"], cur["driver"], cur["lap_number"])
            if key in excluded:
                continue
            history = [l for l in d_laps if l["lap_number"] < cur["lap_number"]]
            out.append({"messages": [
                {"role": "system", "content": RISK_SYSTEM},
                {"role": "user", "content": risk_user(cur, history)},
                {"role": "assistant", "content": json.dumps(label_risks(cur, history))},
            ]})

        # coaching: chase vs session best (or maintain if this driver IS the best)
        cur = d_clean[len(d_clean) // 2]
        key = (cur["year"], cur["event"], cur["driver"], cur["lap_number"])
        if key in excluded or session_best is None:
            continue
        if session_best["driver"] == drv:
            first = d_clean[0]
            gt = coaching_ground_truth(cur, first, "maintain")
            user = coach_user(cur, first, "maintain")
        else:
            gt = coaching_ground_truth(cur, session_best, "chase")
            user = coach_user(cur, session_best, "chase")
        out.append({"messages": [
            {"role": "system", "content": COACH_SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(gt)},
        ]})
    return out


def main() -> None:
    files = sorted(PROCESSED_DIR.glob("*.json"))
    if not files:
        sys.exit(f"No processed sessions in {PROCESSED_DIR} — run scripts/fetch_data.py first.")

    excluded = eval_lap_keys()
    examples = []
    for f in files:
        examples.extend(examples_for_session(json.loads(f.read_text(encoding="utf-8")), excluded))

    rng.shuffle(examples)
    n_val = max(1, len(examples) // 10)
    FINETUNE_DIR.mkdir(parents=True, exist_ok=True)
    for name, subset in (("train", examples[n_val:]), ("val", examples[:n_val])):
        path = FINETUNE_DIR / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for ex in subset:
                fh.write(json.dumps(ex) + "\n")
        print(f"Wrote {len(subset)} examples to {path}")
    print(f"(excluded {len(excluded)} eval laps from training)")


if __name__ == "__main__":
    main()
