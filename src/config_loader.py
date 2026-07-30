"""
STRATEGIES:
    get_config()      -> Returns the parsed config dictionary (cached).
    resolve_path()    -> Resolves project-relative paths to absolute paths.
"""

from pathlib import Path
from typing import Any
import yaml

ROOT = Path(__file__).resolve().parent.parent

_CONFIG: dict[str, Any] | None = None

def get_config() -> dict[str, Any]:
    global _CONFIG

    if _CONFIG is None:
        config_path = ROOT / "config.yaml"

        if not config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: '{config_path}'.")

        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        if config is None:
            raise ValueError("Configuration file is empty.")

        if not isinstance(config, dict):
            raise TypeError("Configuration file must contain a YAML dictionary.")

        _CONFIG = config

    return _CONFIG


def resolve_path(relative_path: str | Path) -> Path:
    return (ROOT / relative_path).resolve()