"""Per-lap feature extraction from FastF1 laps + rule-based ground truth.

The rule labeler is deliberately deterministic: it provides scoreable ground
truth for the baseline eval and target outputs for the fine-tune dataset.
Heuristics are derived from timing/tyre/speed-trap data only (FastF1 does not
expose engine thermals).
"""

import math

import pandas as pd

# Practical stint-length limits (laps) before degradation typically bites.
TYRE_WEAR_LIMIT = {"SOFT": 18, "MEDIUM": 28, "HARD": 38, "INTERMEDIATE": 25, "WET": 20}
SLICKS = {"SOFT", "MEDIUM", "HARD"}


def _sec(v) -> float | None:
    if v is None or pd.isna(v):
        return None
    if isinstance(v, pd.Timedelta):
        return round(v.total_seconds(), 3)
    return round(float(v), 3)


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else round(f, 1)


def is_clean(lap: dict) -> bool:
    """A lap usable for analysis: timed, accurate, not entering/leaving the pits."""
    return bool(lap.get("lap_time_s") and lap.get("is_accurate")
                and not lap.get("is_pit_in_lap") and not lap.get("is_pit_out_lap"))


def lap_to_record(lap: pd.Series, year: int, event: str, weather: dict | None = None) -> dict:
    """One FastF1 lap row -> JSON-friendly per-lap record."""
    rec = {
        "year": year,
        "event": event,
        "driver": str(lap.get("Driver", "")),
        "team": str(lap.get("Team", "")),
        "lap_number": int(lap["LapNumber"]),
        "stint": _num(lap.get("Stint")),
        "position": _num(lap.get("Position")),
        "lap_time_s": _sec(lap.get("LapTime")),
        "sector1_s": _sec(lap.get("Sector1Time")),
        "sector2_s": _sec(lap.get("Sector2Time")),
        "sector3_s": _sec(lap.get("Sector3Time")),
        "speed_i1_kmh": _num(lap.get("SpeedI1")),
        "speed_i2_kmh": _num(lap.get("SpeedI2")),
        "speed_fl_kmh": _num(lap.get("SpeedFL")),
        "speed_st_kmh": _num(lap.get("SpeedST")),
        "compound": str(lap.get("Compound", "UNKNOWN")),
        "tyre_life_laps": _num(lap.get("TyreLife")),
        "fresh_tyre": bool(lap.get("FreshTyre", False)),
        "track_status": str(lap.get("TrackStatus", "")),
        "is_pit_in_lap": not pd.isna(lap.get("PitInTime")),
        "is_pit_out_lap": not pd.isna(lap.get("PitOutTime")),
        "is_personal_best": bool(lap.get("IsPersonalBest", False)),
        "is_accurate": bool(lap.get("IsAccurate", False)),
    }
    if weather:
        rec.update(weather)
    return rec


def stint_trend_s_per_lap(history: list[dict], current: dict, window: int = 3) -> float | None:
    """Lap-time slope (s/lap) over the last `window` clean laps of the same stint."""
    same_stint = [
        h for h in history
        if h.get("stint") == current.get("stint")
        and h.get("lap_time_s") and h.get("is_accurate")
    ]
    pts = [(h["lap_number"], h["lap_time_s"]) for h in same_stint[-window:]]
    if current.get("lap_time_s") and current.get("is_accurate"):
        pts.append((current["lap_number"], current["lap_time_s"]))
    if len(pts) < 3:
        return None
    xs, ys = zip(*pts)
    n = len(pts)
    mx, my = sum(xs) / n, sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return None
    return round(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom, 3)


def label_risks(current: dict, history: list[dict]) -> dict:
    """Rule-based step-1 ground truth: issues + overall status for one lap."""
    issues: list[dict] = []
    compound = current.get("compound", "UNKNOWN")
    tyre_life = current.get("tyre_life_laps") or 0
    limit = TYRE_WEAR_LIMIT.get(compound, 30)
    trend = stint_trend_s_per_lap(history, current)
    raining = bool(current.get("rainfall"))

    if raining and compound in SLICKS:
        issues.append({
            "severity": "critical", "system": "weather",
            "description": f"Rain reported while on {compound} slick tyres.",
            "time_frame": "box this lap",
            "recommendation": "Pit immediately for intermediates before grip collapses.",
        })
    if tyre_life >= limit and (trend or 0) > 0.15:
        issues.append({
            "severity": "critical", "system": "tyres",
            "description": (f"{compound} tyres at {int(tyre_life)} laps (limit ~{limit}) "
                            f"and pace falling {trend:+.2f}s per lap."),
            "time_frame": "within 2 laps",
            "recommendation": "Pit for fresh tyres; degradation is past the cliff.",
        })
    elif tyre_life >= 0.75 * limit:
        issues.append({
            "severity": "warning", "system": "tyres",
            "description": f"{compound} tyres at {int(tyre_life)} of ~{limit} usable laps.",
            "time_frame": "within 5 laps",
            "recommendation": "Prepare a pit window and manage rear traction on exits.",
        })
    if trend is not None and trend > 0.25 and not any(i["system"] == "tyres" and i["severity"] == "critical" for i in issues):
        issues.append({
            "severity": "warning", "system": "pace",
            "description": f"Lap times drifting up {trend:+.2f}s per lap over the stint.",
            "time_frame": "within 3 laps",
            "recommendation": "Check tyre temps and fuel-corrected pace; adjust brake balance.",
        })

    stint_speeds = [h["speed_st_kmh"] for h in history
                    if h.get("stint") == current.get("stint") and h.get("speed_st_kmh")]
    if stint_speeds and current.get("speed_st_kmh"):
        drop = max(stint_speeds) - current["speed_st_kmh"]
        if drop > 8:
            issues.append({
                "severity": "warning", "system": "powertrain",
                "description": f"Speed-trap down {drop:.0f} km/h vs stint best.",
                "time_frame": "within 3 laps",
                "recommendation": "Check energy deployment and DRS usage; report any engine noise.",
            })

    best_sectors = _personal_best_sectors(history)
    for i, key in enumerate(("sector1_s", "sector2_s", "sector3_s"), start=1):
        cur, best = current.get(key), best_sectors.get(key)
        if cur and best and (cur - best) > 0.4 and current.get("is_accurate"):
            issues.append({
                "severity": "low", "system": "driving",
                "description": f"Sector {i} is {cur - best:+.2f}s off personal best.",
                "time_frame": "next lap",
                "recommendation": f"Tidy up sector {i}: braking reference and apex speed.",
            })

    prev_pos = next((h["position"] for h in reversed(history) if h.get("position")), None)
    if prev_pos and current.get("position") and current["position"] > prev_pos:
        issues.append({
            "severity": "low", "system": "strategy",
            "description": f"Lost track position: P{int(prev_pos)} -> P{int(current['position'])}.",
            "time_frame": "within 2 laps",
            "recommendation": "Use overtake mode into the DRS zone to regain the place.",
        })

    if not issues:
        issues.append({
            "severity": "low", "system": "general",
            "description": "No anomalies in timing, tyres, or speed traps this lap.",
            "time_frame": "ongoing",
            "recommendation": "Maintain current inputs and tyre management.",
        })

    severities = {i["severity"] for i in issues}
    overall = ("critical" if "critical" in severities
               else "warning" if "warning" in severities else "ok")
    return {"overall_status": overall, "issues": issues}


def _personal_best_sectors(history: list[dict]) -> dict:
    best = {}
    for key in ("sector1_s", "sector2_s", "sector3_s"):
        vals = [h[key] for h in history if h.get(key) and h.get("is_accurate")]
        if vals:
            best[key] = min(vals)
    return best


def coaching_ground_truth(current: dict, reference: dict, mode: str) -> dict:
    """Rule-based steps-2/3 ground truth.

    mode="chase": reference is the best performer's best lap.
    mode="maintain": reference is this car's own first clean lap.
    """
    deltas = {}
    for key in ("lap_time_s", "sector1_s", "sector2_s", "sector3_s", "speed_st_kmh", "speed_fl_kmh"):
        if current.get(key) is not None and reference.get(key) is not None:
            deltas[key] = round(current[key] - reference[key], 3)

    suggestions: list[str] = []
    sector_deltas = {i: deltas.get(f"sector{i}_s") for i in (1, 2, 3)}
    worst = max((d, i) for i, d in ((i, d) for i, d in sector_deltas.items() if d is not None)) \
        if any(d is not None for d in sector_deltas.values()) else None

    if mode == "chase":
        who = reference.get("driver", "the leader")
        gap = deltas.get("lap_time_s")
        gap_summary = (f"{abs(gap):.2f}s per lap {'slower' if gap and gap > 0 else 'faster'} than "
                       f"{who}'s benchmark lap." if gap is not None
                       else f"Compared against {who}'s benchmark lap.")
        if worst and worst[0] > 0.1:
            suggestions.append(f"Biggest loss is sector {worst[1]} ({worst[0]:+.2f}s): "
                               f"match {who}'s braking points and carry more apex speed there.")
        if (deltas.get("speed_st_kmh") or 0) < -5:
            suggestions.append(f"Down {abs(deltas['speed_st_kmh']):.0f} km/h at the speed trap: "
                               "deploy battery earlier on the main straight and trim wing if balance allows.")
        if (current.get("tyre_life_laps") or 0) > (reference.get("tyre_life_laps") or 0) + 8:
            suggestions.append("Your tyres are significantly older than the benchmark's; "
                               "consider an undercut at the next pit window.")
        suggestions.append("Copy the leader's strongest sector line in clear air before committing to an attack.")
    else:
        gap = deltas.get("lap_time_s")
        gap_summary = (f"{abs(gap):.2f}s per lap {'slower' if gap and gap > 0 else 'faster'} than "
                       "your own first clean lap of the session."
                       if gap is not None else "Compared against your own first clean lap.")
        if gap is not None and gap > 0.3:
            suggestions.append(f"Pace has drifted {gap:+.2f}s from your opening benchmark: "
                               "recheck tyre temps and fuel-corrected pace before rivals close in.")
        else:
            suggestions.append("Pace still matches your opening benchmark; keep the same braking "
                               "references and energy deployment pattern.")
        if worst and worst[0] > 0.15:
            suggestions.append(f"Sector {worst[1]} is where you're bleeding time ({worst[0]:+.2f}s); "
                               "protect it — that's where an attack will come.")
        suggestions.append("Manage tyres to keep a pit-stop buffer over P2; respond to their stops, not the gap.")

    return {
        "mode": mode,
        "compared_to": reference.get("driver", "") if mode == "chase" else "own first clean lap",
        "gap_summary": gap_summary,
        "suggestions": suggestions[:5],
        "key_metric_deltas": deltas,
    }
