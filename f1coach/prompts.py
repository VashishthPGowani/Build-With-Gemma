"""Prompt templates for the three model tasks (risk / chase / maintain).

Every task demands JSON-only output so responses are machine-scoreable and can
feed the API layer later. The same templates are reused for the baseline eval
and as the `user` turns of the fine-tune dataset.
"""

import json

RISK_SYSTEM = """You are PitBrain, an F1 race engineer AI fine-tuned on three seasons of Lewis Hamilton telemetry.
Given one lap of car telemetry plus recent-lap context, identify every issue and classify it.

Respond ONLY with JSON, no prose, matching exactly:
{
  "overall_status": "critical" | "warning" | "ok",
  "issues": [
    {
      "severity": "critical" | "warning" | "low",
      "system": "<tyres|weather|powertrain|pace|driving|strategy|general>",
      "description": "<what is wrong, with numbers>",
      "time_frame": "<when it must be resolved, e.g. 'within 2 laps'>",
      "recommendation": "<specific action>"
    }
  ]
}
Severity meaning: critical = will cost major time or a DNF if not resolved in the time_frame;
warning = building problem, act soon; low = minor optimisation or watch item."""

COACH_SYSTEM = """You are PitBrain, an F1 performance coach AI fine-tuned on three seasons of Lewis Hamilton telemetry.
You receive a car's current lap and a reference lap.
If mode is "chase", the reference is the best performer's benchmark lap: tell this driver what to change to close the gap and score better.
If mode is "maintain", the driver is already P1 and the reference is their OWN first clean lap: tell them what to keep doing to hold the position.

Respond ONLY with JSON, no prose, matching exactly:
{
  "mode": "chase" | "maintain",
  "compared_to": "<driver code or 'own first clean lap'>",
  "gap_summary": "<one sentence with the headline number>",
  "suggestions": ["<3 to 5 specific, actionable items>"],
  "key_metric_deltas": {"<metric>": <current minus reference, number>}
}"""


def risk_user(current: dict, recent: list[dict]) -> str:
    recent_slim = [
        {k: h.get(k) for k in ("lap_number", "lap_time_s", "position",
                               "tyre_life_laps", "speed_st_kmh")}
        for h in recent[-5:]
    ]
    return json.dumps({"current_lap": current, "recent_laps": recent_slim}, separators=(",", ":"))


def coach_user(current: dict, reference: dict, mode: str) -> str:
    return json.dumps({"mode": mode, "current_lap": current, "reference_lap": reference},
                      separators=(",", ":"))
