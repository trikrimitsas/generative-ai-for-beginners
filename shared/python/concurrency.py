"""Concurrency utilities for the Generative AI for Beginners course."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any


class Counter:
    """thread safe counter"""

    def __init__(self) -> None:
        """initialize the counter"""
        self.value = 0
        self.lock = threading.Lock()

    def increment(self) -> int:
        """increment the counter"""
        with self.lock:
            self.value += 1
            return self.value


def run_worker_threads(
    target: Callable[..., Any], thread_count: int, args: tuple[Any, ...] = ()
) -> None:
    """Start worker threads and wait for them to finish."""
    threads = []
    for _ in range(thread_count):
        thread = threading.Thread(target=target, args=args)
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
