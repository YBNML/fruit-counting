"""Stage protocol for pipeline composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class StageResult:
    output: Any
    metadata: dict[str, Any] | None = None


@runtime_checkable
class Stage(Protocol):
    name: str

    def prepare(self, cfg: Any) -> None: ...
    def process(self, x: Any) -> StageResult: ...
    def cleanup(self) -> None: ...
