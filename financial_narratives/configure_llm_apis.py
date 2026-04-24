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
            values[key.strip()] = value.strip().strip("'").strip('"')
    return values


def _write_env_file(path: str, values: Dict[str, str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Local project secrets. Keep this file private.\n")
        for key in sorted(values.keys()):
            f.write(f"{key}={values[key]}\n")


def _ask_for_value(label: str, current_value: str) -> str:
    masked = "(set)" if current_value else "(not set)"
    raw = input(f"{label} {masked} - paste new value or press Enter to keep current: ").strip()
    return raw if raw else current_value


def main():
    print("Configure LLM API keys for this project")
    print("This updates financial_narratives/.env")

    values = _parse_env_file(ENV_FILE)
    values["OPENAI_API_KEY"] = _ask_for_value("OPENAI_API_KEY", values.get("OPENAI_API_KEY", ""))
    values["ANTHROPIC_API_KEY"] = _ask_for_value("ANTHROPIC_API_KEY", values.get("ANTHROPIC_API_KEY", ""))
    values["GOOGLE_API_KEY"] = _ask_for_value("GOOGLE_API_KEY", values.get("GOOGLE_API_KEY", ""))

    # Remove empty keys so the file stays tidy.
    values = {k: v for k, v in values.items() if v}
    _write_env_file(ENV_FILE, values)

    print(f"Saved keys to {ENV_FILE}")
    print("Run: python3 analyze_narrative.py")


if __name__ == "__main__":
    main()
