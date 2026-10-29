"""Conservative process/thread defaults for local interactive runs."""

import os
from typing import Optional


DEFAULT_MAX_WORKERS = 4


def resolve_worker_count(requested: Optional[int] = None) -> int:
    """Return an explicit worker count or a safe default for this machine.

    The pipeline can run two independent optimizers at the same time. Using
    every detected CPU core for each optimizer can therefore oversubscribe the
    host and duplicate large NumPy arrays in many processes.
    """
    if requested is not None:
        if requested < 1:
            raise ValueError("n_workers must be at least 1")
        return requested
    return min(DEFAULT_MAX_WORKERS, os.cpu_count() or 1)
