from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FINETUNE_DIR = DATA_DIR / "finetune"
EVAL_DIR = PROJECT_ROOT / "eval"
RESULTS_DIR = EVAL_DIR / "results"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    spur_api_key: str
    spur_base_url: str = "https://ai.spuric.com/v1"
    # set by scripts/diagnose_endpoints.py if it discovers a fine-tune surface
    spur_ft_base_url: str | None = None
    spur_model: str = "spur-gemma4"
    spur_timeout_s: float = 120.0
    temperature: float = 0.2


settings = Settings()
