# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Runs slow save_dir writes on a background thread, one task at a time in submission order."""

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable


class BackgroundWriter:
    def __init__(self, name: str = "background-writer"):
        self._name = name
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)

    def submit(self, description: str, fn: Callable[[], object]) -> Future:
        """Failures are logged, not raised."""
        submitted = time.perf_counter()

        def run():
            started = time.perf_counter()
            try:
                fn()
            except Exception:
                logging.exception(f"{self._name}: {description} failed")
                return
            finished = time.perf_counter()
            logging.info(
                f"{self._name}: {description} took {finished - started:.1f}s (queued for {started - submitted:.1f}s)"
            )

        return self._executor.submit(run)

    def close(self):
        start = time.perf_counter()
        self._executor.shutdown(wait=True)
        logging.info(f"{self._name}: finished pending tasks in {time.perf_counter() - start:.1f}s")
