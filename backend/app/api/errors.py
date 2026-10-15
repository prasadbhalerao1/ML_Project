"""Validation-error handling.

FastAPI's default 422 body echoes the offending input. When that input is
``NaN`` or ``Infinity`` -- exactly the case the schema exists to reject -- the
default response cannot be serialised to JSON and the request fails with a 500
instead of a readable 422. This handler renders non-finite inputs as strings so
the client gets the 422 and can see which field was at fault.
"""

from __future__ import annotations

import math
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

#: Spelled out rather than taken from ``fastapi.status``, whose name for this
#: code changed between Starlette versions.
UNPROCESSABLE_CONTENT = 422


def register_error_handlers(app: FastAPI) -> None:
    """Attach the project's exception handlers to ``app``."""

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(
        _: Request, error: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=UNPROCESSABLE_CONTENT,
            content={"detail": [_render_error(entry) for entry in error.errors()]},
        )


def _render_error(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "loc": [str(part) for part in entry.get("loc", ())],
        "msg": entry.get("msg", "Invalid value."),
        "type": entry.get("type", "value_error"),
        "input": _json_safe(entry.get("input")),
    }


def _json_safe(value: Any) -> Any:
    """Make a value JSON-encodable, stringifying NaN and +/-Infinity."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)
