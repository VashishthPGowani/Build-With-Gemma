"""Submit a fine-tune job to Spur via the OpenAI-style fine-tuning API.

NOTE (probed 2026-08-01): ai.spuric.com/v1 returned 404 for /files and
/fine_tuning/jobs, so hosted fine-tuning may not be exposed at the standard
paths. This script still tries them (in case Spur enables it or uses a gateway
that routes per-key) and fails with a clear message + fallback instructions.
The dataset in data/finetune/ is portable to any trainer either way.

Usage: uv run scripts/run_finetune.py --base-model spur-gemma4
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from openai import OpenAI

from f1coach.config import FINETUNE_DIR, settings

FALLBACK = """
Spur did not accept the fine-tuning request at the OpenAI-standard endpoints.
Options:
  1. Check the Spur dashboard/docs for a dedicated fine-tuning section or a
     different base URL, and update SPUR_BASE_URL accordingly.
  2. Fallback: train a LoRA on Colab with Unsloth using data/finetune/train.jsonl
     (chat-format JSONL loads directly with
     `datasets.load_dataset("json", data_files="train.jsonl")`).
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=settings.spur_model)
    ap.add_argument("--suffix", default="f1-telemetry-coach")
    args = ap.parse_args()

    train_path = FINETUNE_DIR / "train.jsonl"
    val_path = FINETUNE_DIR / "val.jsonl"
    if not train_path.exists():
        sys.exit("data/finetune/train.jsonl missing — run scripts/build_finetune_dataset.py first.")

    # honor a fine-tune surface discovered by scripts/diagnose_endpoints.py
    ft_base = settings.spur_ft_base_url or settings.spur_base_url
    print(f"Using fine-tune base URL: {ft_base}")
    client = OpenAI(api_key=settings.spur_api_key, base_url=ft_base,
                    timeout=settings.spur_timeout_s)
    try:
        train_file = client.files.create(file=train_path.open("rb"), purpose="fine-tune")
        val_file = client.files.create(file=val_path.open("rb"), purpose="fine-tune")
        print(f"Uploaded: train={train_file.id} val={val_file.id}")

        job = client.fine_tuning.jobs.create(
            model=args.base_model,
            training_file=train_file.id,
            validation_file=val_file.id,
            suffix=args.suffix,
        )
        print(f"Fine-tune job started: {job.id} (status: {job.status})")
    except Exception as e:
        print(f"Fine-tune submission failed: {type(e).__name__}: {e}")
        print(FALLBACK)
        sys.exit(1)

    while True:
        job = client.fine_tuning.jobs.retrieve(job.id)
        print(f"  status: {job.status}")
        if job.status in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(30)

    if job.status == "succeeded":
        print(f"\nFine-tuned model id: {job.fine_tuned_model}")
        print("Evaluate it with: uv run scripts/run_baseline.py "
              f"--model {job.fine_tuned_model} --tag finetuned")


if __name__ == "__main__":
    main()
