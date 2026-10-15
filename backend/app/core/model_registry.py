"""Holds the single :class:`Predictor` instance for the application's lifetime.

Startup calls :meth:`ModelRegistry.load` once. If the artifact is missing the
app still starts, so ``/health`` can report *why* predictions are unavailable
instead of the process dying on boot.
"""

from __future__ import annotations

from app.services.predictor import Predictor
from ml.artifacts import ArtifactError


class ModelRegistry:
    """A loaded predictor, or the reason there isn't one."""

    def __init__(self) -> None:
        self._predictor: Predictor | None = None
        self._error: str | None = None

    def load(self) -> None:
        try:
            self._predictor = Predictor.from_artifacts()
            self._error = None
        except ArtifactError as error:
            self._predictor = None
            self._error = str(error)

    @property
    def predictor(self) -> Predictor | None:
        return self._predictor

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def is_loaded(self) -> bool:
        return self._predictor is not None


#: Application-wide instance, populated by the FastAPI lifespan handler.
registry = ModelRegistry()
