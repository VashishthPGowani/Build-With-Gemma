# PitBrain — Build With Gemma, Pit Lane Telemetry (Track 3)

**Team Firefly** · [Build With Gemma: Triage In Light Speed](https://www.kaggle.com/competitions/build-with-gemma-triage-in-light-speed/) on Kaggle

Gemma-powered F1 race engineer: per-lap risk analysis, comparative coaching,
and a crew-chief voice assistant, built on three seasons (2023–2025) of real
race telemetry pulled via [FastF1](https://github.com/theOehrly/Fast-F1).

📄 **Full submission documentation:** [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md)
🗺️ **Architecture flow diagram:** [docs/architecture-flow.excalidraw](docs/architecture-flow.excalidraw) (open at [excalidraw.com](https://excalidraw.com))

## The lap-by-lap flow

1. **Risk analysis** — each incoming lap is scanned for issues, classified
   `critical / warning / low` with a `time_frame` to resolve before it bites.
2. **Chase coaching** — the car is compared with the session's best performer;
   the model suggests what to change to close the gap.
3. **Maintain coaching** — if the car is already P1, it is compared against its
   *own first clean lap* and told what to keep doing to hold position.
4. **Voice assistant** — a Q&A layer over the results of steps 1–3 (API layer,
   next milestone).

## Setup

```bash
uv sync
# create .env (never commit it):
#   SPUR_API_KEY=sk-spur-...
#   SPUR_BASE_URL=https://ai.spuric.com/v1
#   SPUR_MODEL=spur-gemma4      # Gemma 4 31B Dense on Spur
```

## Workflow

```bash
uv run scripts/fetch_data.py                # 1. pull + process race laps (FastF1, cached)
uv run scripts/build_eval.py                # 2. 36 task-shaped eval items w/ rule ground truth
uv run scripts/run_baseline.py              # 3. baseline: score base Gemma on the 3 tasks
uv run scripts/analyze_race.py              # 4. full per-lap risk + coaching for one driver
uv run scripts/analyze_race.py --all        # 4b. whole field (12 sampled laps per driver)
uv run scripts/build_dashboard.py --all     # 5. per-driver review pages + ranked index.html
```

Baseline result (2026-08-01, `spur-gemma4`, 36 items): 100% JSON-valid, 100%
schema-valid, mean task score 0.793 (risk 0.585, chase 1.0, maintain 1.0).

Any of the 9 fetched races works for steps 4–5, e.g.
`uv run scripts/analyze_race.py --session 2024_hungarian_grand_prix --driver HAM`
then the same flags on `build_dashboard.py`.

## Live replay (race "happening now")

```bash
uv run main.py                                    # terminal 1: server on :8000
uv run scripts/replay_race.py                     # terminal 2: one race lap every 20s
# open http://127.0.0.1:8000 — live chart, running order, verdicts, issue ticker
```

The replayer POSTs each lap of the whole field to `POST /api/laps`; the server
analyzes every car live with Gemma (featured car every clean lap, others every
`--analyze-every` laps) against the *evolving* field benchmark, and the page
polls `GET /api/state` every 2s. `--interval` changes the lap cadence;
`--session`/`--featured` pick any fetched race and car. `POST /api/analyze`
runs any skipped car/lap on demand, and the same `POST /api/laps` endpoint
accepts real (non-replay) telemetry unchanged.

**Voice assistant:** `POST /api/ask` `{"question": "...", "history": []}` →
`{"answer", "grounded_on"}` — a radio-style answer grounded ONLY in the live
session context (position, tyres, recent lap times, latest analyses,
leaderboard). The live page has an "Ask the pit wall" box wired to it, and
speaks answers aloud via the browser's speech synthesis.

Scoring: `json_valid_rate` (response parses), `schema_valid_rate` (required
keys/enums present), `mean_task_score` (agreement with deterministic rule-based
ground truth: status/system overlap for risk; mode, suggestion count, and
numeric-delta accuracy for coaching). Results land in `eval/results/`.

> **Fine-tuning note:** Spur has no fine-tuning capability (confirmed on their
> site; `/files` and `/fine_tuning/jobs` also 404). The project therefore runs
> `spur-gemma4` with task prompts directly. `scripts/build_finetune_dataset.py`
> still produces a portable chat-JSONL dataset (`data/finetune/`) if training
> elsewhere (e.g. Unsloth LoRA on Colab) becomes an option, and
> `scripts/run_finetune.py` is ready should a hosted endpoint appear.

## Layout

```
f1coach/            config, Spur client (JSON-robust), lap features + rule labeler, prompts
scripts/            fetch_data, build_eval, run_baseline, analyze_race, build_dashboard,
                    build_finetune_dataset, run_finetune
dashboard/          template.html + generated per-race review dashboards
data/processed/     per-session lap records (gitignored)
data/analysis/      per-race model analysis JSON (gitignored)
data/finetune/      train/val JSONL (gitignored)
eval/               eval_set.jsonl + results/ (results gitignored)
```
