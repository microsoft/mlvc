# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from pathlib import Path

_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


def get_env(key: str, default: str = "") -> str:
    """Read a value from the environment, falling back to the repository .env file."""
    value = os.environ.get(key)
    if value is not None:
        return value

    if _ENV_FILE.exists():
        from dotenv import dotenv_values

        value = dotenv_values(_ENV_FILE).get(key)
        if value is not None:
            return value

    return default


def get_required_env(key: str) -> str:
    value = get_env(key)
    if not value:
        raise RuntimeError(f"{key} environment variable is not set")
    return value
