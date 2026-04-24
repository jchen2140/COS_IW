import os
from typing import Dict


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(BASE_DIR, ".env")


def _parse_env_file(path: str) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not os.path.exists(path):
        return values

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key:
                values[key] = value
    return values


def load_local_env() -> None:
    """
    Load .env values into process environment if variables are not already set.
    Existing exported variables always win over .env values.
    """
    file_values = _parse_env_file(ENV_FILE)
    for key, value in file_values.items():
        if key not in os.environ and value:
            os.environ[key] = value


def get_x_token() -> str:
    load_local_env()
    return os.getenv("X_BEARER_TOKEN", "").strip() or os.getenv("X_API_KEY", "").strip()
