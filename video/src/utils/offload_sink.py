# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import queue
import threading
from typing import Callable

__all__ = ["OffloadSink"]


class OffloadSink:
    """
    Sink, offloading operations to the background thread
    """

    def __init__(self, queue_size=5):
        self._sink = None
        self._queue = queue.Queue(maxsize=queue_size)
        self._thread = None
        self._terminated = False
        self._exception = None

    def start(self, sink: Callable):
        thread = threading.Thread(target=self._run, args=(sink,), name="offload-sink", daemon=True)
        thread.start()
        self._thread = thread

    def join(self):
        if self._thread is None or self._terminated:
            return

        self._terminated = True
        if self._exception is None:
            self._queue.put(OffloadSink)
        self._thread.join()

        while not self._queue.empty():
            self._queue.get()

    def close(self):
        self.join()

        exception = self._exception
        if exception is not None:
            self._exception = None
            raise exception

    def __call__(self, *args, **kwargs):
        if self._thread is None or self._terminated:
            raise BrokenPipeError("AsyncSync is already terminated")

        if self._exception is None:
            self._queue.put((args, kwargs))
        else:
            self.close()

    def _run(self, sink: Callable):
        try:
            while True:
                msg = self._queue.get()
                if msg is OffloadSink:
                    break

                args, kwargs = msg
                sink(*args, **kwargs)
        except Exception as e:
            self._exception = e

            while not self._queue.empty():
                self._queue.get()
