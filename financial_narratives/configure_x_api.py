import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(BASE_DIR, ".env")


def _parse_env_file(path):
    values = {}
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


def _write_env_file(path, values):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Local project secrets. Keep this file private.\n")
        for key in sorted(values.keys()):
            f.write(f"{key}={values[key]}\n")


def main():
    print("Configure X API credential for this project")
    print("This updates financial_narratives/.env without removing other keys.")
    token = input("Paste X bearer token (or API key): ").strip()
    if not token:
        print("No token entered. Nothing changed.")
        return

    values = _parse_env_file(ENV_FILE)
    values["X_BEARER_TOKEN"] = token
    _write_env_file(ENV_FILE, values)

    print(f"Saved token to {ENV_FILE}")
    print("You can now run: python3 x.py")


if __name__ == "__main__":
    main()
