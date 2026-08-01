"""Live race server: ingest laps as they happen, analyze EVERY car with the
Spur-hosted Gemma model, and serve a live-updating dashboard.

The four-step flow runs per car, live:
  1. risk scan   — each analyzed lap gets critical/warning/low issues + time frames
  2. chase       — coaching vs the evolving field benchmark (best performer)
  3. maintain    — if the car is P1, coaching vs its OWN first clean lap instead
  4. voice       — /api/ask answers questions grounded in steps 1-3 for any car

The featured car is analyzed every clean lap; every other car on a staggered
cadence (default every 3rd lap) so the model API keeps up with the whole field.

Endpoints:
    POST /api/reset   start a new session (event meta, featured driver, cadence)
    POST /api/laps    one lap record for one car (the replayer or a real feed)
    GET  /api/state   full live state for the dashboard (poll every ~2s)
    POST /api/ask     voice assistant: question (+optional driver) -> radio answer
    GET  /            the live dashboard page
    GET  /health      liveness + config check

Run: uv run main.py   (then open http://127.0.0.1:8000)
"""

import json
import threading
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .config import PROJECT_ROOT, settings
from .features import is_clean, label_risks
from .prompts import COACH_SYSTEM, RISK_SYSTEM, coach_user, risk_user
from .spur import chat, parse_json_loose

LIVE_PAGE = PROJECT_ROOT / "dashboard" / "live.html"

app = FastAPI(title="PitBrain live")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

_lock = threading.Lock()
_analyzer = ThreadPoolExecutor(max_workers=6)


def _fresh_state() -> dict:
    return {"meta": {"event": None, "year": None, "featured": None,
                     "model": settings.spur_model, "total_laps": None,
                     "analyze_every": 3},
            "laps": {}, "analyses": {}, "pending": 0}


_state = _fresh_state()


class ResetPayload(BaseModel):
    event: str
    year: int
    featured: str = "HAM"
    total_laps: int | None = None
    analyze_every: int = 3  # non-featured cars: analyze every Nth lap


class LapPayload(BaseModel, extra="allow"):
    driver: str
    lap_number: int


class AskPayload(BaseModel):
    question: str
    driver: str | None = None  # defaults to the featured car
    history: list[dict] = []


class AnalyzePayload(BaseModel):
    driver: str
    lap_number: int


ASK_SYSTEM = """You are PitBrain, the race engineer on the radio for car {car}.
Answer the driver's question using ONLY the session context JSON below. Be concise
and calm — one to three short sentences, radio style, no markdown — the answer
will be spoken aloud. If the context doesn't cover the question, say what you do
know and never invent numbers.

SESSION CONTEXT:
{context}"""


def _model_json(system: str, user: str) -> dict:
    try:
        raw = chat(system, user)
    except Exception as e:
        return {"_error": f"{type(e).__name__}: {e}"}
    parsed = parse_json_loose(raw)
    return parsed if parsed is not None else {"_error": "unparseable", "_raw": raw[:300]}


def _benchmark() -> dict | None:
    best = None
    for laps in _state["laps"].values():
        for l in laps:
            if is_clean(l) and (best is None or l["lap_time_s"] < best["lap_time_s"]):
                best = l
    return best


def _analyze(lap: dict) -> None:
    drv = lap["driver"]
    with _lock:
        d_laps = list(_state["laps"].get(drv, []))
        history = [l for l in d_laps if l["lap_number"] < lap["lap_number"]]
        first_clean = next((l for l in d_laps if is_clean(l)), lap)
        benchmark = _benchmark() or lap
    # step 3 (maintain vs own first lap) when leading, else step 2 (chase best)
    maintain = (lap.get("position") == 1
                and lap["lap_number"] > first_clean["lap_number"])
    mode = "maintain" if maintain else "chase"
    reference = first_clean if maintain else benchmark

    risk = _model_json(RISK_SYSTEM, risk_user(lap, history))          # step 1
    coaching = _model_json(COACH_SYSTEM, coach_user(lap, reference, mode))
    rules = label_risks(lap, history)

    entry = {
        "n": lap["lap_number"],
        "st": risk.get("overall_status", "ok") if "_error" not in risk else "ok",
        "ruleSt": rules["overall_status"],
        "issues": [
            {"sev": i.get("severity", "low"), "sys": i.get("system", "general"),
             "d": i.get("description", ""), "tf": i.get("time_frame", ""),
             "rec": i.get("recommendation", "")}
            for i in (risk.get("issues") or []) if isinstance(i, dict)
        ] if "_error" not in risk else [],
        "coach": None if "_error" in coaching else {
            "mode": coaching.get("mode", mode),
            "cmp": coaching.get("compared_to", ""),
            "gap": coaching.get("gap_summary", ""),
            "sugg": coaching.get("suggestions") or [],
        },
        "mode": mode,
        "error": risk.get("_error") or coaching.get("_error"),
    }
    with _lock:
        lst = _state["analyses"].setdefault(drv, [])
        lst.append(entry)
        lst.sort(key=lambda a: a["n"])
        _state["pending"] -= 1


@app.post("/api/reset")
def reset(payload: ResetPayload) -> dict:
    global _state
    with _lock:
        _state = _fresh_state()
        _state["meta"].update(event=payload.event, year=payload.year,
                              featured=payload.featured,
                              total_laps=payload.total_laps,
                              analyze_every=max(1, payload.analyze_every))
    return {"ok": True}


@app.post("/api/laps")
def ingest_lap(payload: LapPayload) -> dict:
    lap = payload.model_dump()
    meta = _state["meta"]
    if meta["featured"] is None:
        raise HTTPException(409, "Session not started — POST /api/reset first.")
    drv = lap["driver"]
    every = meta["analyze_every"]
    due = (drv == meta["featured"]
           or lap["lap_number"] % every == sum(map(ord, drv)) % every)
    analyze = due and is_clean(lap)
    with _lock:
        _state["laps"].setdefault(drv, []).append(lap)
        if analyze:
            _state["pending"] += 1
    if analyze:
        _analyzer.submit(_analyze, lap)
    return {"accepted": True, "queued_analysis": analyze}


@app.post("/api/analyze")
def analyze_on_demand(payload: AnalyzePayload) -> dict:
    """Queue a Gemma analysis for any car's specific lap (replay-slider button)."""
    laps = _state["laps"].get(payload.driver)
    if not laps:
        raise HTTPException(404, f"No laps received for car {payload.driver}.")
    lap = next((l for l in laps if l["lap_number"] == payload.lap_number), None)
    if lap is None:
        raise HTTPException(404, f"Lap {payload.lap_number} not received yet.")
    if not is_clean(lap):
        raise HTTPException(400, "Lap is not analyzable (pit in/out or inaccurate timing).")
    if any(a["n"] == payload.lap_number
           for a in _state["analyses"].get(payload.driver, [])):
        return {"queued": False, "already_analyzed": True}
    with _lock:
        _state["pending"] += 1
    _analyzer.submit(_analyze, lap)
    return {"queued": True, "already_analyzed": False}


@app.get("/api/state")
def state() -> dict:
    with _lock:
        latest = {}
        for drv, laps in _state["laps"].items():
            last = max(laps, key=lambda l: l["lap_number"])
            if last.get("position"):
                latest[drv] = int(last["position"])
        leaderboard = sorted(({"d": d, "p": p} for d, p in latest.items()),
                             key=lambda r: r["p"])
        bench = _benchmark()
        current_lap = max((max(l["lap_number"] for l in laps)
                           for laps in _state["laps"].values() if laps), default=0)
        cars = {
            drv: {"laps": [{"n": l["lap_number"], "t": l.get("lap_time_s"),
                            "pos": l.get("position"), "comp": l.get("compound"),
                            "life": l.get("tyre_life_laps"), "stint": l.get("stint"),
                            "spd": l.get("speed_st_kmh"),
                            "clean": is_clean(l)} for l in laps],
                  "analyses": _state["analyses"].get(drv, [])}
            for drv, laps in _state["laps"].items()
        }
        return {"meta": _state["meta"], "current_lap": current_lap,
                "pending": _state["pending"],
                "benchmark": bench and {"driver": bench["driver"],
                                        "lap": bench["lap_number"],
                                        "time": bench["lap_time_s"]},
                "leaderboard": leaderboard, "cars": cars}


def _context_bundle(drv: str) -> dict:
    with _lock:
        d_laps = _state["laps"].get(drv, [])
        last = d_laps[-1] if d_laps else None
        latest = {}
        for d, laps in _state["laps"].items():
            l = max(laps, key=lambda x: x["lap_number"])
            if l.get("position"):
                latest[d] = int(l["position"])
        leaderboard = sorted(latest.items(), key=lambda kv: kv[1])[:5]
        bench = _benchmark()
        recent = _state["analyses"].get(drv, [])[-3:]
        clean_d = [l for l in d_laps if is_clean(l)]
        return {
            "event": f"{_state['meta']['year']} {_state['meta']['event']}",
            "car": drv,
            "current_lap": last and last["lap_number"],
            "position": last and last.get("position"),
            "tyres": last and {"compound": last.get("compound"),
                               "age_laps": last.get("tyre_life_laps")},
            "last_lap_time_s": last and last.get("lap_time_s"),
            "best_lap_time_s": min((l["lap_time_s"] for l in clean_d), default=None),
            "field_benchmark": bench and {"driver": bench["driver"],
                                          "time_s": bench["lap_time_s"]},
            "leaderboard": [{"driver": d, "position": p} for d, p in leaderboard],
            "recent_lap_times_s": [l.get("lap_time_s") for l in d_laps[-5:]],
            "recent_analyses": [
                {"lap": a["n"], "status": a["st"], "mode": a["mode"],
                 "issues": a["issues"],
                 "coaching": a["coach"] and {"gap": a["coach"]["gap"],
                                             "suggestions": a["coach"]["sugg"]}}
                for a in recent],
        }


@app.post("/api/ask")
def ask(payload: AskPayload) -> dict:
    if _state["meta"]["featured"] is None:
        raise HTTPException(409, "Session not started — no live race to ask about.")
    drv = payload.driver or _state["meta"]["featured"]
    if drv not in _state["laps"]:
        raise HTTPException(404, f"No laps received for car {drv}.")
    ctx = _context_bundle(drv)
    system = ASK_SYSTEM.format(car=drv, context=json.dumps(ctx))
    turns = "".join(f"{t.get('role', 'driver')}: {t.get('content', '')}\n"
                    for t in payload.history[-6:])
    try:
        answer = chat(system, turns + f"driver: {payload.question}",
                      temperature=0.4, max_tokens=200).strip()
    except Exception as e:
        raise HTTPException(502, f"Model call failed: {type(e).__name__}")
    return {"answer": answer, "driver": drv,
            "grounded_on": {"lap": ctx["current_lap"], "position": ctx["position"],
                            "open_issues": sum(len(a["issues"])
                                               for a in ctx["recent_analyses"])}}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(LIVE_PAGE, media_type="text/html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": settings.spur_model,
            "session": _state["meta"]["event"]}
