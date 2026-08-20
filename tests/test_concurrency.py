"""Tests for shared.python.concurrency."""

from shared.python.concurrency import Counter, run_worker_threads


def test_counter_is_safe_for_concurrent_increments():
    counter = Counter()

    def increment_counter():
        for _ in range(1_000):
            counter.increment()

    run_worker_threads(increment_counter, 10)

    assert counter.value == 10_000


def test_run_worker_threads_passes_args_to_each_worker():
    values = []

    def append_value(value):
        values.append(value)

    run_worker_threads(append_value, 3, args=("value",))

    assert values == ["value", "value", "value"]
