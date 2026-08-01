"""Replay a processed race against the live server as if it were happening now.

Groups the session's laps by lap number and POSTs each lap's records on an
interval, featured driver last (so the evolving benchmark already includes
that lap's rivals). Default cadence: one race lap every 20 seconds.

Usage:
    uv run scripts/replay_race.py
    uv run scripts/replay_race.py --session 2025_british_grand_prix --interval 8
"""

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from f1coach.config import PROCESSED_DIR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="2025_british_grand_prix")
    ap.add_argument("--featured", default="HAM")
    ap.add_argument("--interval", type=float, default=20.0,
                    help="seconds between race laps")
    ap.add_argument("--analyze-every", type=int, default=3,
                    help="non-featured cars are analyzed every Nth lap")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    args = ap.parse_args()

    path = PROCESSED_DIR / f"{args.session}.json"
    if not path.exists():
        sys.exit(f"{path} not found — run scripts/fetch_data.py first.")
    session = json.loads(path.read_text(encoding="utf-8"))
    laps = session["laps"]

    by_lap: dict[int, list[dict]] = {}
    for lap in laps:
        by_lap.setdefault(lap["lap_number"], []).append(lap)
    total = max(by_lap)

    client = httpx.Client(base_url=args.base, timeout=30)
    r = client.post("/api/reset", json={"event": session["event"],
                                        "year": session["year"],
                                        "featured": args.featured,
                                        "total_laps": total,
                                        "analyze_every": args.analyze_every})
    r.raise_for_status()
    print(f"Replaying {session['year']} {session['event']} — {total} laps, "
          f"one every {args.interval:g}s, featured {args.featured}. Ctrl+C to stop.")

    for n in sorted(by_lap):
        records = sorted(by_lap[n], key=lambda l: l["driver"] == args.featured)
        for lap in records:
            client.post("/api/laps", json=lap).raise_for_status()
        feat = next((l for l in records if l["driver"] == args.featured), None)
        note = (f"{args.featured} {feat['lap_time_s']}s P{feat.get('position')}"
                if feat and feat.get("lap_time_s") else f"no {args.featured} lap")
        print(f"  lap {n:>2}/{total}: {len(records)} cars in — {note}")
        if n != total:
            time.sleep(args.interval)

    print("Race complete.")


if __name__ == "__main__":
    main()
