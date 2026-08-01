"""Analyze drivers' races with the Spur-hosted Gemma model, lap by lap.

For every analyzed lap: a risk analysis (task 1) and coaching (task 2 chase vs
the session's best performer, or task 3 maintain vs own first clean lap when
leading). Model output is stored alongside the deterministic rule-based labels
so the dashboard can show where Gemma agrees with, or improves on, the rules.

Usage:
    uv run scripts/analyze_race.py                                  # 2025 British GP, HAM, all clean laps
    uv run scripts/analyze_race.py --session 2024_hungarian_grand_prix --driver HAM
    uv run scripts/analyze_race.py --all                            # whole field, sampled laps
    uv run scripts/analyze_race.py --all --max-laps 0               # whole field, every clean lap
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from f1coach.config import DATA_DIR, PROCESSED_DIR, settings
from f1coach.features import coaching_ground_truth, label_risks
from f1coach.prompts import COACH_SYSTEM, RISK_SYSTEM, coach_user, risk_user
from f1coach.spur import chat, parse_json_loose

ANALYSIS_DIR = DATA_DIR / "analysis"


def clean(laps):
    return [l for l in laps if l.get("lap_time_s") and l.get("is_accurate")
            and not l.get("is_pit_in_lap") and not l.get("is_pit_out_lap")]


def model_json(system: str, user: str) -> dict:
    try:
        raw = chat(system, user)
    except Exception as e:
        return {"_error": f"{type(e).__name__}: {e}"}
    parsed = parse_json_loose(raw)
    return parsed if parsed is not None else {"_error": "unparseable", "_raw": raw[:500]}


def analyze_lap(cur: dict, d_laps: list[dict], first_clean: dict, benchmark: dict) -> dict:
    history = [l for l in d_laps if l["lap_number"] < cur["lap_number"]]
    maintain = (cur.get("position") == 1 and cur["lap_number"] > first_clean["lap_number"])
    mode = "maintain" if maintain else "chase"
    reference = first_clean if maintain else benchmark

    risk = model_json(RISK_SYSTEM, risk_user(cur, history))
    coaching = model_json(COACH_SYSTEM, coach_user(cur, reference, mode))
    return {
        "lap": cur,
        "risk": risk,
        "risk_rules": label_risks(cur, history),
        "coaching": coaching,
        "coaching_rules": coaching_ground_truth(cur, reference, mode),
        "mode": mode,
        "reference_driver": reference.get("driver"),
        "reference_lap_number": reference.get("lap_number"),
    }


def sample_evenly(items: list, n: int) -> list:
    if n <= 0 or n >= len(items):
        return items
    step = (len(items) - 1) / (n - 1)
    return [items[round(i * step)] for i in range(n)]


def classification(laps: list[dict]) -> list[dict]:
    finals = {}
    for lap in laps:
        if lap.get("position"):
            finals[lap["driver"]] = (lap["lap_number"], int(lap["position"]))
    return sorted(({"driver": d, "position": p, "laps": n} for d, (n, p) in finals.items()),
                  key=lambda r: r["position"])


def analyze_driver(session: dict, session_name: str, driver: str, benchmark: dict,
                   workers: int, max_laps: int) -> Path | None:
    laps = session["laps"]
    d_laps = sorted((l for l in laps if l["driver"] == driver),
                    key=lambda l: l["lap_number"])
    d_clean = clean(d_laps)
    if len(d_clean) < 6:
        print(f"  {driver}: skipped ({len(d_clean)} clean laps)")
        return None

    targets = sample_evenly(d_clean, max_laps)
    first_clean = d_clean[0]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        analyzed = list(pool.map(
            lambda cur: analyze_lap(cur, d_laps, first_clean, benchmark), targets))
    errors = sum(1 for a in analyzed
                 if "_error" in a["risk"] or "_error" in a["coaching"])
    modes = {m: sum(1 for a in analyzed if a["mode"] == m) for m in ("chase", "maintain")}

    out = {
        "session": session_name, "year": session["year"], "event": session["event"],
        "driver": driver, "model": settings.spur_model,
        "benchmark": {k: benchmark.get(k) for k in
                      ("driver", "lap_number", "lap_time_s", "compound")},
        "classification": classification(laps),
        "all_laps": [{k: l.get(k) for k in
                      ("lap_number", "lap_time_s", "position", "compound",
                       "tyre_life_laps", "stint", "is_pit_in_lap", "is_pit_out_lap",
                       "is_accurate", "team")} for l in d_laps],
        "analyzed_laps": analyzed,
    }
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ANALYSIS_DIR / f"{session_name}_{driver}.json"
    out_path.write_text(json.dumps(out), encoding="utf-8")
    print(f"  {driver}: {len(analyzed)} laps ({modes['chase']} chase / "
          f"{modes['maintain']} maintain), {errors} errors, {time.time() - t0:.0f}s")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="2025_british_grand_prix")
    ap.add_argument("--driver", default="HAM")
    ap.add_argument("--all", action="store_true", help="analyze every driver in the session")
    ap.add_argument("--max-laps", type=int, default=None,
                    help="laps analyzed per driver (0 = all clean laps; "
                         "default: all for single driver, 12 for --all)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--skip-existing", action="store_true", default=None,
                    help="skip drivers that already have an analysis file (default in --all mode)")
    args = ap.parse_args()

    path = PROCESSED_DIR / f"{args.session}.json"
    if not path.exists():
        sys.exit(f"{path} not found — run scripts/fetch_data.py first.")
    session = json.loads(path.read_text(encoding="utf-8"))
    laps = session["laps"]

    benchmark = min(clean(laps), key=lambda l: l["lap_time_s"])
    drivers = ([r["driver"] for r in classification(laps)] if args.all else [args.driver])
    max_laps = args.max_laps if args.max_laps is not None else (12 if args.all else 0)
    skip_existing = args.skip_existing if args.skip_existing is not None else args.all

    print(f"{args.session}: {len(drivers)} driver(s), up to "
          f"{max_laps or 'all'} laps each, model {settings.spur_model}; "
          f"benchmark {benchmark['driver']} lap {benchmark['lap_number']} "
          f"({benchmark['lap_time_s']}s)")
    for drv in drivers:
        out_path = ANALYSIS_DIR / f"{args.session}_{drv}.json"
        if skip_existing and out_path.exists():
            print(f"  {drv}: kept existing analysis (delete {out_path.name} to redo)")
            continue
        analyze_driver(session, args.session, drv, benchmark, args.workers, max_laps)
    print("\nDone.")


if __name__ == "__main__":
    main()
