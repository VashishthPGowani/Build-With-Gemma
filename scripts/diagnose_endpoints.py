"""Diagnose the Spur endpoint pipeline stage by stage, and self-update config.

Pipeline model:
  [1] client request build   [2] DNS/TLS transport   [3] auth gate
  [4] gateway route resolution   [5] capability backend

Verifies stages 2-3 and inference explicitly, then sweeps candidate fine-tune
routes/hosts (stage 4). If any host+path answers like a real fine-tuning
surface (200, or 401/403/405/422 — i.e. the route EXISTS), it writes
SPUR_FT_BASE_URL into .env so scripts/run_finetune.py uses it automatically.
If everything is an application-level 404, the verdict is a missing stage-5
capability on Spur's side — nothing a client can fix.

Safety: the API key is only ever sent to *.spuric.com hosts.

Usage: uv run scripts/diagnose_endpoints.py
"""

import json
import sys
from pathlib import Path
from urllib.parse import urlparse

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from f1coach.config import PROJECT_ROOT, settings

ENV_PATH = PROJECT_ROOT / ".env"
ALLOWED_SUFFIX = ".spuric.com"

CANDIDATE_HOSTS = [
    "https://ai.spuric.com/v1",
    "https://ai.spuric.com",
    "https://api.spuric.com/v1",
]
# OpenAI current + legacy + common provider variants
CANDIDATE_FT_PATHS = [
    "/fine_tuning/jobs", "/fine-tuning/jobs", "/fine_tunes", "/finetunes",
    "/files", "/training/jobs", "/fine_tuning/models",
]


def allowed(url: str) -> bool:
    host = urlparse(url).hostname or ""
    return host == ALLOWED_SUFFIX.lstrip(".") or host.endswith(ALLOWED_SUFFIX)


def probe(client: httpx.Client, url: str, with_auth: bool = True) -> dict:
    headers = {}
    if with_auth:
        if not allowed(url):
            return {"url": url, "verdict": "SKIPPED (host outside spuric.com — key not sent)"}
        headers["Authorization"] = f"Bearer {settings.spur_api_key}"
    try:
        r = client.get(url, headers=headers)
    except httpx.HTTPError as e:
        return {"url": url, "verdict": f"TRANSPORT FAIL (stage 2): {type(e).__name__}"}
    body = r.text[:120].replace("\n", " ")
    exists = r.status_code in (200, 401, 403, 405, 422)
    return {"url": url, "status": r.status_code, "body": body,
            "route_exists": exists,
            "verdict": ("ROUTE EXISTS" if exists else
                        "no route (stage 4): app-level 404" if r.status_code == 404
                        else f"HTTP {r.status_code}")}


def set_env_var(name: str, value: str) -> None:
    lines = ENV_PATH.read_text(encoding="utf-8").splitlines() if ENV_PATH.exists() else []
    lines = [l for l in lines if not l.startswith(f"{name}=")]
    lines.append(f"{name}={value}")
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    base = settings.spur_base_url.rstrip("/")
    print(f"Diagnosing Spur pipeline (base: {base})\n")
    ok = True
    with httpx.Client(timeout=15, follow_redirects=True) as client:
        # stage 2: transport (no auth header)
        t = probe(client, base + "/models", with_auth=False)
        transport_ok = "TRANSPORT FAIL" not in t["verdict"]
        print(f"[2] transport (DNS/TLS)      {'OK' if transport_ok else t['verdict']}")
        ok &= transport_ok

        # stage 3: auth gate — /models with key must be 200; without key should NOT be
        a = probe(client, base + "/models")
        auth_ok = a.get("status") == 200
        print(f"[3] auth gate (/models)      {'OK — key accepted' if auth_ok else a['verdict']}"
              f"{' (note: also open without key!)' if auth_ok and t.get('status') == 200 else ''}")
        ok &= auth_ok

        # stage 4+5 for inference: one tiny completion
        try:
            r = client.post(base + "/chat/completions",
                            headers={"Authorization": f"Bearer {settings.spur_api_key}"},
                            json={"model": settings.spur_model, "max_tokens": 4,
                                  "messages": [{"role": "user", "content": "say OK"}]})
            inf_ok = r.status_code == 200
            print(f"[4+5] inference route        {'OK — ' + settings.spur_model + ' responds' if inf_ok else f'HTTP {r.status_code}: {r.text[:100]}'}")
            ok &= inf_ok
        except httpx.HTTPError as e:
            print(f"[4+5] inference route        FAIL: {type(e).__name__}")
            ok = False

        # stage 4 sweep: fine-tuning surface discovery
        print("\n[4] fine-tune route sweep (key sent only to *.spuric.com):")
        found = []
        for host in CANDIDATE_HOSTS:
            for path in CANDIDATE_FT_PATHS:
                res = probe(client, host.rstrip("/") + path)
                marker = "  ->" if res.get("route_exists") else "    "
                print(f"{marker} GET {res['url']}: {res['verdict']}"
                      + (f" [{res.get('status')}]" if "status" in res and not res.get("route_exists") is None else ""))
                if res.get("route_exists"):
                    found.append((host, path, res))

    print("\n== verdict ==")
    if found:
        host = found[0][0].rstrip("/")
        ft_base = host if host.endswith("/v1") else host + "/v1"
        set_env_var("SPUR_FT_BASE_URL", ft_base)
        print(f"Fine-tune surface detected at {found[0][0]}{found[0][1]} "
              f"(status {found[0][2].get('status')}).")
        print(f"SELF-UPDATE: wrote SPUR_FT_BASE_URL={ft_base} to .env — "
              "scripts/run_finetune.py will use it. Run it next.")
    else:
        print("Stages 2, 3 and inference are healthy." if ok else
              "Some core stages failed — see above.")
        print("Every candidate fine-tune route returned an application-level 404 from")
        print("Spur's own gateway: the failure is at stage 4 because the stage-5")
        print("capability backend (file store + training queue) does not exist on")
        print("Spur's side. There is nothing to fix in this client; fine-tuning needs")
        print("a provider that hosts it (or local LoRA training with data/finetune/).")


if __name__ == "__main__":
    main()
