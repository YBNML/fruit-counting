"""Result I/O."""

from counting.io.results import CountingResult, CropMeta, StageTiming
from counting.io.serialize import (
    read_batch_csv,
    read_batch_json,
    write_batch_csv,
    write_batch_json,
)

__all__ = [
    "CountingResult", "CropMeta", "StageTiming",
    "read_batch_csv", "read_batch_json",
    "write_batch_csv", "write_batch_json",
]
