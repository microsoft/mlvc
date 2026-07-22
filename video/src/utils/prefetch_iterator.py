# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import queue
import threading
from typing import Iterable

__all__ = ["PrefetchIterator"]


class PrefetchIterator:
    """
    Prefetching iterator

    Prefetch is done on a background thread
    """

    def __init__(self, queue_size=5):
        self._iterator = None
        self._queue = queue.Queue(maxsize=queue_size)
        self._thread = None
        self._terminated = False

    def start(self, source: Iterable):
        thread = threading.Thread(target=self._run, args=(source,), name="prefetch-iterator", daemon=True)
        thread.start()
        self._thread = thread

    def join(self):
        if self._thread is None or self._terminated:
            return

        self._terminated = True

        while not self._queue.empty():
            self._queue.get()

        self._thread.join()

    def __iter__(self):
        return self

    def __next__(self):
        if self._terminated:
            raise StopIteration

        msg = self._queue.get()
        if isinstance(msg, Exception):
            self.join()
            raise msg

        return msg

    def _run(self, source: Iterable):
        try:
            for item in source:
                if self._terminated:
                    break

                assert not isinstance(item, Exception)
                self._queue.put(item)

            self._queue.put(StopIteration())
        except Exception as e:
            self._queue.put(e)
