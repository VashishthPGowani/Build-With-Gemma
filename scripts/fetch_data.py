"""Fetch race telemetry with FastF1 and process to per-lap JSON records.

Keeps ALL drivers (comparisons in steps 2/3 need the rest of the field), with
Hamilton (HAM) as the featured car. Default scope: three GPs per season across
2023-2025 to stay download-friendly; pass --years/--events to widen.

Usage:
    uv run scripts/fetch_data.py
    uv run scripts/fetch_data.py --years 2024 2025 --events "Monaco Grand Prix"
"""

import argparse
import json
import sys
from pathlib import Path

import fastf1

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from f1coach.config import DATA_DIR, PROCESSED_DIR
from f1coach.features import lap_to_record

DEFAULT_YEARS = [2023, 2024, 2025]
DEFAULT_EVENTS = ["British Grand Prix", "Hungarian Grand Prix", "Italian Grand Prix"]


def fetch_session(year: int, event: str) -> dict | None:
    try:
        session = fastf1.get_session(year, event, "R")
        session.load(laps=True, telemetry=False, weather=True, messages=False)
    except Exception as e:
        print(f"  !! {year} {event}: load failed ({type(e).__name__}: {e})")
        return None

    laps = session.laps.reset_index(drop=True)
    if laps.empty:
        print(f"  !! {year} {event}: no laps")
        return None

    try:
        weather = laps.get_weather_data().reset_index(drop=True)
    except Exception:
        weather = None

    records = []
    for i, lap in laps.iterrows():
        wx = None
        if weather is not None and i < len(weather):
            w = weather.iloc[i]
            wx = {
                "air_temp_c": round(float(w["AirTemp"]), 1),
                "track_temp_c": round(float(w["TrackTemp"]), 1),
                "rainfall": bool(w["Rainfall"]),
            }
        records.append(lap_to_record(lap, year, event, weather=wx))

    drivers = sorted({r["driver"] for r in records})
    print(f"  ok {year} {event}: {len(records)} laps, {len(drivers)} drivers"
          f"{' (incl HAM)' if 'HAM' in drivers else ' (NO HAM!)'}")
    return {"year": year, "event": event, "session": "R", "laps": records}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS)
    ap.add_argument("--events", nargs="+", default=DEFAULT_EVENTS)
    args = ap.parse_args()

    cache_dir = DATA_DIR / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))

    written = 0
    for year in args.years:
        for event in args.events:
            data = fetch_session(year, event)
            if data is None:
                continue
            slug = event.lower().replace(" ", "_")
            out = PROCESSED_DIR / f"{year}_{slug}.json"
            out.write_text(json.dumps(data), encoding="utf-8")
            written += 1

    print(f"\nWrote {written} session files to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
