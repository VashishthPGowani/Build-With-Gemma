"""Build a task-shaped eval set from processed session JSON.

Samples real laps into ~30 eval items across the three tasks:
  - risk:     mid/late-stint HAM laps (so the rule labeler has trend context)
  - chase:    HAM lap vs the session best performer's benchmark lap
  - maintain: leader's lap vs their own first clean lap
Each item stores the exact messages to send plus rule-based ground truth.

Usage: uv run scripts/build_eval.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from f1coach.config import EVAL_DIR, PROCESSED_DIR
from f1coach.features import coaching_ground_truth, label_risks
from f1coach.prompts import COACH_SYSTEM, RISK_SYSTEM, coach_user, risk_user

PER_SESSION = {"risk": 2, "chase": 1, "maintain": 1}  # x9 sessions ~= 36 items


def driver_laps(laps: list[dict], driver: str) -> list[dict]:
    return sorted((l for l in laps if l["driver"] == driver), key=lambda l: l["lap_number"])


def clean(laps: list[dict]) -> list[dict]:
    return [l for l in laps if l.get("lap_time_s") and l.get("is_accurate")
            and not l.get("is_pit_in_lap") and not l.get("is_pit_out_lap")]


def best_performer(laps: list[dict]) -> tuple[str, dict] | None:
    """(driver, benchmark_lap) with the fastest clean lap of the session."""
    best = None
    for lap in clean(laps):
        if best is None or lap["lap_time_s"] < best["lap_time_s"]:
            best = lap
    return (best["driver"], best) if best else None


def session_winner(laps: list[dict]) -> str | None:
    finals = {}
    for lap in laps:
        if lap.get("position"):
            finals[lap["driver"]] = (lap["lap_number"], lap["position"])
    if not finals:
        return None
    return min(finals.items(), key=lambda kv: (kv[1][1], -kv[1][0]))[0]


def build_items(session: dict) -> list[dict]:
    laps, items = session["laps"], []
    tag = f"{session['year']}_{session['event'].lower().replace(' ', '_')}"

    ham = driver_laps(laps, "HAM")
    ham_clean = clean(ham)
    # risk: pick laps at 1/2 and 3/4 through HAM's race for varied tyre states
    picks = []
    if len(ham_clean) >= 10:
        picks = [ham_clean[len(ham_clean) // 2], ham_clean[3 * len(ham_clean) // 4]]
    for n, cur in enumerate(picks[:PER_SESSION["risk"]]):
        history = [l for l in ham if l["lap_number"] < cur["lap_number"]]
        items.append({
            "id": f"{tag}_risk_{n}", "task": "risk",
            "system": RISK_SYSTEM, "user": risk_user(cur, history),
            "ground_truth": label_risks(cur, history),
        })

    bp = best_performer(laps)
    if bp and ham_clean:
        who, bench = bp
        cur = ham_clean[len(ham_clean) // 2]
        if who != "HAM":
            items.append({
                "id": f"{tag}_chase_0", "task": "chase",
                "system": COACH_SYSTEM, "user": coach_user(cur, bench, "chase"),
                "ground_truth": coaching_ground_truth(cur, bench, "chase"),
            })

    winner = session_winner(laps)
    if winner:
        w_clean = clean(driver_laps(laps, winner))
        if len(w_clean) >= 6:
            first, cur = w_clean[0], w_clean[2 * len(w_clean) // 3]
            items.append({
                "id": f"{tag}_maintain_0", "task": "maintain",
                "system": COACH_SYSTEM, "user": coach_user(cur, first, "maintain"),
                "ground_truth": coaching_ground_truth(cur, first, "maintain"),
            })
    return items


def main() -> None:
    files = sorted(PROCESSED_DIR.glob("*.json"))
    if not files:
        sys.exit(f"No processed sessions in {PROCESSED_DIR} — run scripts/fetch_data.py first.")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    all_items = []
    for f in files:
        all_items.extend(build_items(json.loads(f.read_text(encoding="utf-8"))))

    out = EVAL_DIR / "eval_set.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for item in all_items:
            fh.write(json.dumps(item) + "\n")

    by_task = {}
    for item in all_items:
        by_task[item["task"]] = by_task.get(item["task"], 0) + 1
    print(f"Wrote {len(all_items)} eval items to {out} — {by_task}")


if __name__ == "__main__":
    main()
