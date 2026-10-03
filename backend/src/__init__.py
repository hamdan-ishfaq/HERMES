"""HERMES backend package.

Loads ``backend/.env`` here, once, so that the environment is configured
before any submodule reads it.

This matters because several modules read configuration at import time --
``src.db`` builds its engine from ``DATABASE_URL`` at module scope, and
``src.auth`` reads ``SECRET_KEY``. Previously ``load_dotenv()`` was called
from a handful of leaf modules (``rag.cache``, the ingestion loaders,
``verify``), so whether ``.env`` was honoured depended on which of those
happened to be imported first: importing ``src.routers.eval`` directly saw
an unset ``EVAL_ADMIN_EMAILS``, while importing ``src.rag.cache`` first saw
the value from ``.env``. An unset ``DATABASE_URL`` silently falls back to
the hardcoded ``hermes_db``, so that ordering bug could point the app at
the wrong database.

Importing any ``src.*`` module executes this file first, which makes the
ordering deterministic.

Real environment variables take precedence over ``.env`` so that CI and
container deployments can override without editing the file.
"""

from pathlib import Path

from dotenv import load_dotenv

# backend/.env, resolved from this file rather than the working directory.
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

if _ENV_FILE.is_file():
    load_dotenv(_ENV_FILE, override=False)