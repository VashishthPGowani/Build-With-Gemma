"""Inject analysis JSON into the dashboard template; build per-driver pages + index.

Usage:
    uv run scripts/build_dashboard.py                         # 2025 British GP, HAM
    uv run scripts/build_dashboard.py --session 2024_hungarian_grand_prix --driver HAM
    uv run scripts/build_dashboard.py --all                   # every analyzed driver + index.html
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from f1coach.config import DATA_DIR, PROJECT_ROOT

TEMPLATE = PROJECT_ROOT / "dashboard" / "template.html"


def compact(analysis: dict) -> dict:
    laps = [{"n": l["lap_number"], "t": l["lap_time_s"], "pos": l.get("position"),
             "comp": l.get("compound"), "life": l.get("tyre_life_laps"),
             "stint": l.get("stint"), "acc": l.get("is_accurate")}
            for l in analysis["all_laps"]]

    an = {}
    for a in analysis["analyzed_laps"]:
        risk, coach = a["risk"], a["coaching"]
        if "_error" in risk:
            continue
        c = None
        if "_error" not in coach:
            c = {"mode": coach.get("mode", a["mode"]), "cmp": coach.get("compared_to", ""),
                 "gap": coach.get("gap_summary", ""), "sugg": coach.get("suggestions") or []}
        an[str(a["lap"]["lap_number"])] = {
            "st": risk.get("overall_status", "ok"),
            "ruleSt": a["risk_rules"]["overall_status"],
            "issues": [{"sev": i.get("severity", "low"), "sys": i.get("system", "general"),
                        "d": i.get("description", ""), "tf": i.get("time_frame", ""),
                        "rec": i.get("recommendation", "")}
                       for i in (risk.get("issues") or []) if isinstance(i, dict)],
            "coach": c,
        }

    team = next((l.get("team") for l in analysis["all_laps"] if l.get("team")), "")
    # all_laps only carries chart fields; take team from the analyzed lap records
    if not team and analysis["analyzed_laps"]:
        team = analysis["analyzed_laps"][0]["lap"].get("team", "")

    return {
        "meta": {
            "event": analysis["event"], "year": analysis["year"],
            "driver": analysis["driver"], "team": team, "model": analysis["model"],
            "benchmark": {"driver": analysis["benchmark"]["driver"],
                          "lap": analysis["benchmark"]["lap_number"],
                          "time": analysis["benchmark"]["lap_time_s"]},
            "classification": [{"d": r["driver"], "p": r["position"]}
                               for r in analysis["classification"]],
        },
        "laps": laps,
        "analysis": an,
    }


def wrap_page(title: str, body: str) -> str:
    return ("<!doctype html>\n<html lang=\"en\">\n<head>\n"
            "<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            f"<title>{title}</title>\n</head>\n<body>\n{body}\n</body>\n</html>\n")


def build_driver_page(session_name: str, driver: str) -> dict | None:
    src = DATA_DIR / "analysis" / f"{session_name}_{driver}.json"
    if not src.exists():
        return None
    analysis = json.loads(src.read_text(encoding="utf-8"))
    data = compact(analysis)

    title = f"PitBrain Review — {data['meta']['year']} {data['meta']['event']} · {driver}"
    body = (TEMPLATE.read_text(encoding="utf-8")
            .replace("<title>__TITLE__</title>\n", "")
            .replace("__DATA_JSON__", json.dumps(data).replace("</", "<\\/")))
    out = PROJECT_ROOT / "dashboard" / f"{session_name}_{driver}.html"
    out.write_text(wrap_page(title, body), encoding="utf-8")
    print(f"Wrote {out.name} ({out.stat().st_size // 1024} KB)")
    return data


INDEX_CSS = """
  :root { color-scheme: light;
    --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink-2:#52514e; --ink-3:#898781;
    --grid:#e1e0d9; --axis:#c3c2b7; --ring:rgba(11,11,11,0.10);
    --crit:#d03b3b; --warn:#fab219; --good:#0ca30c; --series:#2a78d6; }
  @media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) {
    color-scheme: dark; --page:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink-2:#c3c2b7;
    --ink-3:#898781; --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,0.10); --series:#3987e5; } }
  :root[data-theme="dark"] {
    color-scheme: dark; --page:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink-2:#c3c2b7;
    --ink-3:#898781; --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,0.10); --series:#3987e5; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--page); color:var(--ink);
         font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
  .wrap { max-width:1000px; margin:0 auto; padding:28px 20px 64px; }
  .eyebrow { font-size:11px; font-weight:600; letter-spacing:0.12em; text-transform:uppercase; color:var(--ink-3); }
  h1 { font-size:26px; margin:2px 0 4px; letter-spacing:-0.01em; }
  .sub { color:var(--ink-2); font-size:14px; margin-bottom:22px; }
  .sub b { color:var(--ink); }
  .card { background:var(--surface); border:1px solid var(--ring); border-radius:10px; padding:6px 20px 10px; }
  .tbl-scroll { overflow-x:auto; }
  table { border-collapse:collapse; width:100%; font-size:13.5px; }
  th { text-align:left; font-size:11px; letter-spacing:0.08em; text-transform:uppercase;
       color:var(--ink-3); padding:10px 14px 8px 0; border-bottom:1px solid var(--axis); }
  td { padding:9px 14px 9px 0; border-bottom:1px solid var(--grid);
       font-variant-numeric:tabular-nums; color:var(--ink-2); white-space:nowrap; }
  tr:last-child td { border-bottom:none; }
  td.drv { color:var(--ink); font-weight:600; }
  th.num, td.num { text-align:right; }
  .sev { display:inline-flex; align-items:center; gap:4px; margin-right:10px; }
  a.open { color:var(--series); font-weight:600; text-decoration:none; }
  a.open:hover { text-decoration:underline; }
  a.open:focus-visible { outline:2px solid var(--series); outline-offset:2px; }
  footer { color:var(--ink-3); font-size:12.5px; margin-top:18px; }
"""


def build_index(session_name: str, pages: list[dict]) -> None:
    meta = pages[0]["meta"]
    by_driver = {p["meta"]["driver"]: p for p in pages}
    rows = []
    for r in meta["classification"]:
        p = by_driver.get(r["d"])
        if not p:
            continue
        an = p["analysis"]
        counts = {"critical": 0, "warning": 0, "ok": 0}
        n_maintain = 0
        for a in an.values():
            counts[a["st"]] = counts.get(a["st"], 0) + 1
            if (a.get("coach") or {}).get("mode") == "maintain":
                n_maintain += 1
        clean_laps = [l for l in p["laps"] if l["t"] and l["acc"]]
        best = min(l["t"] for l in clean_laps)
        gap = best - meta["benchmark"]["time"]
        rows.append(
            f"<tr><td class=\"num\">P{r['p']}</td>"
            f"<td class=\"drv\">{r['d']}</td><td>{p['meta']['team']}</td>"
            f"<td class=\"num\">{len(an)}</td>"
            f"<td><span class=\"sev\"><span style=\"color:var(--crit)\" aria-hidden=\"true\">▲</span>{counts['critical']}</span>"
            f"<span class=\"sev\"><span style=\"color:var(--warn)\" aria-hidden=\"true\">◆</span>{counts['warning']}</span>"
            f"<span class=\"sev\"><span style=\"color:var(--good)\" aria-hidden=\"true\">●</span>{counts['ok']}</span></td>"
            f"<td class=\"num\">{best:.3f}s</td>"
            f"<td class=\"num\">{'+' if gap >= 0 else ''}{gap:.3f}s</td>"
            f"<td class=\"num\">{n_maintain}</td>"
            f"<td><a class=\"open\" href=\"{session_name}_{r['d']}.html\">open review</a></td></tr>")

    body = f"""<style>{INDEX_CSS}</style>
<div class="wrap">
  <div class="eyebrow">PitBrain telemetry review · Gemma via Spur</div>
  <h1>{meta['year']} {meta['event']} — full field</h1>
  <div class="sub">Per-lap risk + coaching by <b>{meta['model']}</b> ·
    field benchmark: <b>{meta['benchmark']['driver']}</b> lap {meta['benchmark']['lap']},
    {meta['benchmark']['time']:.3f}s · ▲ critical / ◆ warning / ● clear laps</div>
  <div class="card"><div class="tbl-scroll">
  <table>
    <thead><tr><th class="num">Pos</th><th>Driver</th><th>Team</th>
      <th class="num">Laps analyzed</th><th>Model verdicts</th>
      <th class="num">Best lap</th><th class="num">Gap to benchmark</th>
      <th class="num">Maintain laps</th><th></th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  </div></div>
  <footer>Generated from FastF1 race data · each row links to that driver's full lap-by-lap review.</footer>
</div>"""
    out = PROJECT_ROOT / "dashboard" / f"{session_name}_index.html"
    out.write_text(wrap_page(f"PitBrain Review — {meta['year']} {meta['event']} · full field", body),
                   encoding="utf-8")
    print(f"Wrote {out.name} — index of {len(rows)} drivers")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="2025_british_grand_prix")
    ap.add_argument("--driver", default="HAM")
    ap.add_argument("--all", action="store_true",
                    help="build a page for every analyzed driver plus an index")
    args = ap.parse_args()

    if args.all:
        files = sorted((DATA_DIR / "analysis").glob(f"{args.session}_*.json"))
        if not files:
            sys.exit(f"No analysis files for {args.session} — run scripts/analyze_race.py --all first.")
        pages = []
        for f in files:
            driver = f.stem.replace(f"{args.session}_", "")
            data = build_driver_page(args.session, driver)
            if data:
                pages.append(data)
        build_index(args.session, pages)
    else:
        if build_driver_page(args.session, args.driver) is None:
            sys.exit(f"No analysis for {args.driver} — run scripts/analyze_race.py first.")


if __name__ == "__main__":
    main()
