# PitBrain — a Gemma-Powered F1 Race Engineer

**Team Firefly** · Kaggle [Build With Gemma: Triage In Light Speed](https://www.kaggle.com/competitions/build-with-gemma-triage-in-light-speed/) · **Track 3: Pit Lane Telemetry (Formula 1 Strategy)**

> *"With hundreds of data points streaming from an F1 car every second, race engineers must predict tire degradation, engine thermals, and overtaking windows instantly."*

PitBrain is that race engineer: a Gemma-powered system that ingests real Formula 1 telemetry lap by lap, triages every car's problems in seconds, coaches each driver against the fastest car on track, and answers radio questions out loud — all live, for the entire field at once.

The architecture flow diagram lives alongside this file: **[architecture-flow.excalidraw](architecture-flow.excalidraw)** (open at [excalidraw.com](https://excalidraw.com) → Open).

---

## 1. The four-step flow

Telemetry arrives once per lap, per car. Every incoming lap runs through:

| Step | What happens | Comparison target |
|---|---|---|
| **1 · Risk scan** | Gemma marks every issue as `critical / warning / low`, each with a **time-frame to resolve before it strikes** (e.g. "box this lap", "within 2 laps") and a recommendation | the car's own recent laps |
| **2 · Chase coaching** | For any car not leading: what to change to perform and score better | the **best performer's** benchmark lap (evolving live) |
| **3 · Maintain coaching** | For the car in P1: what to keep doing to stay there | the leader's **own first clean lap** |
| **4 · Voice assistant** | Radio-style Q&A grounded *only* in the results of steps 1–3; the browser speaks the answer aloud | live session state |

Steps 1–3 run automatically for **every car in the field** (featured car every clean lap, others on a staggered cadence), and any skipped lap can be analyzed on demand from the UI.

## 2. Architecture

```
FastF1 API ──> fetch_data.py ──> data/processed/  (9 races · ~9,670 laps · 20 drivers)
                                     │
        ┌────────────────────────────┼─────────────────────────┐
        │ offline                    │ offline                 │ live
        ▼                            ▼                         ▼
  build_eval.py (36 items)    build_finetune_dataset.py   replay_race.py (1 lap / 20 s)
        │                     (753 train / 83 val JSONL)        │ POST /api/laps
        ▼                                                       ▼
  run_baseline.py ──> scores                        FastAPI server (f1coach/api.py)
                                                     │  in-memory live state
                                    steps 1–3 JSON   │▲            ▲│ GET /api/state · 2 s poll
                                    per car, per lap ▼│            │▼ POST /api/analyze · /api/ask
                                              Spur gateway    live dashboard (browser)
                                              spur-gemma4     charts · gauges · strategy · voice
```

**Components**

| Piece | Role |
|---|---|
| `f1coach/spur.py` | The only module that knows the model wire format: OpenAI-compatible calls to Spur, plus a defensive JSON-parse ladder (raw → strip code fences → first `{…}` block) so a sloppy model response can never 500 the server |
| `f1coach/features.py` | Per-lap feature extraction from FastF1 + a deterministic rule-based labeler (tyre-wear limits per compound, stint pace-trend slope, speed-trap drops, rain-on-slicks) used as scoreable ground truth |
| `f1coach/prompts.py` | The three task prompts; every task demands JSON-only output against an explicit schema |
| `f1coach/api.py` | FastAPI live server: lap ingest, per-car analysis orchestration, live state, voice endpoint |
| `scripts/` | fetch → eval → baseline → analyze race → dashboards → replay → fine-tune tooling |
| `dashboard/live.html` | Single-file live UI (no build step): stat header, replay slider, running order, vehicle systems monitor with car diagram, engineer radio with speech synthesis, recommended action + ranked strategies, issue ticker, light/dark theme |

## 3. How Gemma is used

- **Model:** `spur-gemma4` — **Gemma 4 31B Dense** served through Spur's OpenAI-compatible gateway (`https://ai.spuric.com/v1`). The API key lives only in a gitignored `.env`.
- **Structured output:** every step-1/2/3 call embeds the expected JSON schema in the system prompt and is parsed defensively; responses validate into typed shapes before reaching the UI.
- **Live orchestration:** the server analyzes the featured car on every clean lap and the rest of the field on a staggered every-Nth-lap cadence (default 3), so ~16 cars stay covered without outrunning the model. A worker pool keeps analyses in lap order; at the demo cadence (20 s/lap) verdicts land well inside each lap window.
- **Grounded voice (step 4):** `/api/ask` builds a compact context bundle — position, tyres, recent lap times, latest analyses, leaderboard — and instructs Gemma to answer *only* from it, radio-style in 1–3 sentences. The browser speaks the reply via speech synthesis.
- **Fine-tuning investigation:** we planned baseline → fine-tune. Spur turned out to expose **no fine-tuning capability** (confirmed on their site; a stage-by-stage diagnostic — `scripts/diagnose_endpoints.py` — verified transport, auth and inference are healthy while all 21 candidate fine-tune routes 404 at the gateway). We still ship a **portable chat-JSONL dataset** (753 train / 83 val examples, eval laps excluded to prevent leakage) built from rule-based ground truth, ready for Unsloth/LoRA the moment training compute is available, plus `scripts/run_finetune.py` which auto-targets any OpenAI-style fine-tune surface if one appears.

## 4. The data

- **Source:** [FastF1](https://github.com/theOehrly/Fast-F1) — official F1 timing data.
- **Scope:** 9 races — British, Hungarian and Italian Grands Prix × 2023, 2024, 2025 — **~9,670 laps across all 20 drivers**, including Lewis Hamilton at both Mercedes (2023–24) and Ferrari (2025).
- **Per-lap record:** lap/sector times, speed traps (I1/I2/FL/ST), compound + tyre age, stint, position, track status, pit flags, accuracy flag, weather (air/track temp, rainfall).
- **Honesty:** everything shown derives from real timing data. Tyre "health" is computed from compound wear limits, pace trend from lap-time slopes, top-speed from speed traps. The feed has no per-corner tyre temps or engine thermals — the UI says so rather than inventing numbers.

## 5. Evaluation

A 36-item eval set built from real laps (18 risk / 9 chase / 9 maintain), each with deterministic rule-based ground truth. Scoring: JSON validity, schema validity, and task agreement (status + system overlap for risk; mode, suggestion count, numeric-delta accuracy for coaching).

**Baseline — `spur-gemma4`, 2026-08-01** (`eval/results/baseline_spur-gemma4.json`):

| Metric | Overall | Risk (18) | Chase (9) | Maintain (9) |
|---|---|---|---|---|
| JSON valid | **100%** | 100% | 100% | 100% |
| Schema valid | **100%** | 100% | 100% | 100% |
| Task score | **0.793** | 0.585 | 1.0 | 1.0 |

The model already nails structure and coaching arithmetic; risk-severity judgment (0.585) is the headroom a future fine-tune targets — exactly what our training dataset teaches.

Full-race validation: the entire 2025 British GP field (15 classified drivers) analyzed with **zero errors** (183 live analyses); maintain-mode correctly tracked the real lead battle (Piastri 7 maintain laps, Norris taking over late before winning).

## 6. Running the demo

```bash
uv sync
# .env (never committed):  SPUR_API_KEY=...  SPUR_BASE_URL=https://ai.spuric.com/v1  SPUR_MODEL=spur-gemma4

# one-time data fetch (cached):
uv run scripts/fetch_data.py

# live demo — two terminals:
uv run main.py                     # server on http://127.0.0.1:8000
uv run scripts/replay_race.py      # replays 2025 British GP, 1 lap / 20 s
```

Open **http://127.0.0.1:8000**: the race unfolds live — lap counter, running order (click any car), tyre-diagram + gauges, Gemma verdict markers landing on the lap trace, pinned act-now issues, recommended action with ranked strategies, and the radio box (type a question; the answer is grounded and spoken). Drag the replay slider to any lap and hit **Analyze with Gemma** for laps the cadence skipped.

Offline artifacts: `uv run scripts/run_baseline.py` reproduces the eval table; `uv run scripts/analyze_race.py --all` + `uv run scripts/build_dashboard.py --all` generate the static per-driver review dashboards (`dashboard/*_index.html`).

## 7. API reference

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/reset` | Start a session `{event, year, featured, total_laps, analyze_every}` |
| POST | `/api/laps` | Ingest one car-lap (replayer **or any real telemetry feed** — same schema) |
| GET | `/api/state` | Full live state: meta, leaderboard, per-car laps + analyses, benchmark |
| POST | `/api/analyze` | On-demand Gemma analysis for `{driver, lap_number}` |
| POST | `/api/ask` | Voice assistant: `{question, driver?, history?}` → grounded radio answer |
| GET | `/health` | Liveness + model/session info |

## 8. Limitations

- Gaps between cars are lap-time pace deltas, not cumulative race gaps.
- Telemetry is timing-derived; no engine temps or brake data (stated in the UI).
- Non-featured cars are auto-analyzed on a sampled cadence (fully configurable; any lap is one click away).
- Strategy ranking (HOLD/PUSH/CONSERVE/PIT NOW) blends Gemma's per-lap verdicts with deterministic tyre/pace signals — transparent and reproducible rather than a black box.
- Hosted fine-tuning was blocked by the provider; the training dataset and submission tooling are ready for when compute is available.

## 9. Repository map

```
f1coach/            config · Spur client · features + rule labeler · prompts · FastAPI server
scripts/            fetch_data · build_eval · run_baseline · analyze_race · build_dashboard
                    replay_race · build_finetune_dataset · run_finetune · diagnose_endpoints
dashboard/          live.html (live UI) · template.html + generated per-race review pages
data/               processed laps · analysis JSON · finetune JSONL   (gitignored)
eval/               eval_set.jsonl · results/                         (results gitignored)
docs/               DOCUMENTATION.md (this file) · architecture-flow.excalidraw
```

---

*Team Firefly — Build With Gemma: Triage In Light Speed, Track 3.*
