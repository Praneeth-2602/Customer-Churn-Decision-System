"""Backend package exports for ASGI import convenience.

Allows Uvicorn import strings like `Backend:app` to work by
exposing `app` at package level.
"""

from .main import app  # re-export for `uvicorn Backend:app`

__all__ = ["app"]

