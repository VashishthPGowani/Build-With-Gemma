"""Run the eval set against a model on Spur and score the responses.

Scores per item:
  json_valid     - response parsed to a JSON object
  schema_valid   - required keys present with allowed values
  task_score     - task-specific agreement with rule-based ground truth:
                     risk:     overall_status match (1.0) or adjacent (0.5),
                               plus credit for hitting ground-truth issue systems
                     chase /   mode + compared_to correct, suggestion count 3-5,
                     maintain  numeric deltas within 20% of ground truth

Usage:
    uv run scripts/run_baseline.py                    # model from .env (spur-gemma4)
    uv run scripts/run_baseline.py --model spur-XXX --tag finetuned
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from f1coach.config import EVAL_DIR, RESULTS_DIR, settings
from f1coach.spur import chat, parse_json_loose

SEVERITY_ORDER = {"ok": 0, "warning": 1, "critical": 2}


def score_risk(parsed: dict, gt: dict) -> tuple[float, bool]:
    status = parsed.get("overall_status")
    issues = parsed.get("issues")
    schema_ok = (status in ("critical", "warning", "ok") and isinstance(issues, list)
                 and all(isinstance(i, dict) and i.get("severity") in ("critical", "warning", "low")
                         and i.get("time_frame") for i in (issues or [])))
    score = 0.0
    if status == gt["overall_status"]:
        score += 0.5
    elif status in SEVERITY_ORDER and abs(SEVERITY_ORDER[status] - SEVERITY_ORDER[gt["overall_status"]]) == 1:
        score += 0.25
    gt_systems = {i["system"] for i in gt["issues"]}
    got_systems = {i.get("system") for i in issues} if isinstance(issues, list) else set()
    if gt_systems:
        score += 0.5 * len(gt_systems & got_systems) / len(gt_systems)
    return round(score, 3), schema_ok


def score_coach(parsed: dict, gt: dict) -> tuple[float, bool]:
    schema_ok = (parsed.get("mode") in ("chase", "maintain")
                 and isinstance(parsed.get("suggestions"), list)
                 and isinstance(parsed.get("gap_summary"), str)
                 and isinstance(parsed.get("key_metric_deltas"), dict))
    score = 0.0
    if parsed.get("mode") == gt["mode"]:
        score += 0.3
    n_sugg = len(parsed.get("suggestions") or [])
    if 3 <= n_sugg <= 5:
        score += 0.2
    gt_deltas = gt["key_metric_deltas"]
    got_deltas = parsed.get("key_metric_deltas") or {}
    checked = matched = 0
    for k, v in gt_deltas.items():
        g = got_deltas.get(k)
        if isinstance(g, (int, float)) and v is not None:
            checked += 1
            tol = max(abs(v) * 0.2, 0.05)
            if abs(g - v) <= tol:
                matched += 1
    if checked:
        score += 0.5 * matched / checked
    return round(score, 3), schema_ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=settings.spur_model)
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--limit", type=int, default=0, help="only run first N items")
    args = ap.parse_args()

    eval_path = EVAL_DIR / "eval_set.jsonl"
    items = [json.loads(l) for l in eval_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        items = items[:args.limit]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for i, item in enumerate(items, 1):
        t0 = time.time()
        try:
            raw = chat(item["system"], item["user"], model=args.model)
            error = None
        except Exception as e:
            raw, error = "", f"{type(e).__name__}: {e}"
        parsed = parse_json_loose(raw) if raw else None

        json_valid = parsed is not None
        if not json_valid:
            task_score, schema_valid = 0.0, False
        elif item["task"] == "risk":
            task_score, schema_valid = score_risk(parsed, item["ground_truth"])
        else:
            task_score, schema_valid = score_coach(parsed, item["ground_truth"])

        results.append({
            "id": item["id"], "task": item["task"], "json_valid": json_valid,
            "schema_valid": schema_valid, "task_score": task_score,
            "latency_s": round(time.time() - t0, 2), "error": error,
            "raw_response": raw, "ground_truth": item["ground_truth"],
        })
        print(f"[{i}/{len(items)}] {item['id']}: json={json_valid} "
              f"schema={schema_valid} score={task_score}"
              + (f" ERROR {error}" if error else ""))

    n = len(results)
    summary = {
        "model": args.model, "tag": args.tag, "items": n,
        "json_valid_rate": round(sum(r["json_valid"] for r in results) / n, 3),
        "schema_valid_rate": round(sum(r["schema_valid"] for r in results) / n, 3),
        "mean_task_score": round(sum(r["task_score"] for r in results) / n, 3),
        "by_task": {},
    }
    for task in ("risk", "chase", "maintain"):
        sub = [r for r in results if r["task"] == task]
        if sub:
            summary["by_task"][task] = {
                "items": len(sub),
                "json_valid_rate": round(sum(r["json_valid"] for r in sub) / len(sub), 3),
                "schema_valid_rate": round(sum(r["schema_valid"] for r in sub) / len(sub), 3),
                "mean_task_score": round(sum(r["task_score"] for r in sub) / len(sub), 3),
            }

    out = RESULTS_DIR / f"{args.tag}_{args.model}.json"
    out.write_text(json.dumps({"summary": summary, "results": results}, indent=2),
                   encoding="utf-8")
    print("\n== summary ==")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved full results to {out}")


if __name__ == "__main__":
    main()
