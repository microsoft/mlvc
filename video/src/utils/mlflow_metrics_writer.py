# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import os
import time
from threading import Condition, RLock, Thread
from typing import Any, Mapping

from mlflow import MlflowClient
from mlflow.entities import Metric

__all__ = ["DebugMetricsWriter", "MlflowMetricsWriter", "get_mlflow_run_id"]

_MAX_QUEUE_AGE = 5.0
_MAX_QUEUE_SIZE = 1000


def get_mlflow_run_id() -> str | None:
    run_id = os.environ.get("MLFLOW_RUN_ID")
    if run_id is not None:
        return run_id

    import mlflow

    active_run = mlflow.active_run()
    return active_run.info.run_id if active_run is not None else None


class DebugMetricsWriter:
    def add_metrics(self, metrics: Mapping[str, Any], *, step: int | None = None) -> None:
        logger = logging.getLogger()
        if not logger.isEnabledFor(logging.DEBUG):
            return

        for name, value in metrics.items():
            if step is None:
                logger.debug(f"metric {name} = {value}")
            else:
                logger.debug(f"metric {name} = {value} at {step}")

    def add_scalar(self, name: str, value: Any, step: int | None = None) -> None:
        self.add_metrics({name: value}, step=step)

    def flush(self) -> None:
        pass


class _AsyncWriter:
    def __init__(self, *, client: MlflowClient, run_id: str):
        self._client = client
        self._run_id = run_id
        self._lock = RLock()
        self._metric_posted = Condition(self._lock)
        self._queue_drained = Condition(self._lock)
        self._terminated = False
        self._exception: BaseException | None = None
        self._queue: list[Metric] | None = None
        self._posted_ts = 0.0

        self._thread = Thread(target=self._async_loop, name="MLFlowMetricsWriter", daemon=True)
        self._thread.start()

    def _async_loop(self) -> None:
        try:
            while True:
                batch = self._get_batch()
                if batch is None:
                    return

                try:
                    self._client.log_batch(self._run_id, metrics=batch, synchronous=True)
                except Exception as error:
                    logging.error(f"Failed to log metrics batch of size {len(batch)}: {error}")
                    with self._lock:
                        self._exception = error
                        self._terminated = True
                        self._queue_drained.notify_all()
                    return
        except BaseException as error:
            with self._lock:
                self._exception = error
                self._queue_drained.notify_all()

    def _get_batch(self) -> list[Metric] | None:
        with self._lock:
            while True:
                batch = self._queue
                if batch:
                    self._queue = None
                    self._queue_drained.notify_all()
                    return batch

                if self._terminated:
                    return None

                self._metric_posted.wait()

    def post_metrics(self, metrics: list[Metric]) -> None:
        with self._lock:
            draining = False
            while True:
                exception = self._exception
                if exception is not None:
                    self._terminated = True
                    self._exception = None
                    raise exception

                if self._terminated:
                    raise RuntimeError("Metrics writer is already closed")

                if not metrics:
                    return

                if self._queue is None or len(self._queue) < _MAX_QUEUE_SIZE:
                    break

                if not draining:
                    if time.monotonic() - self._posted_ts < _MAX_QUEUE_AGE:
                        break

                    logging.warning(f"Metrics queue is full ({len(self._queue)} metrics), waiting for it to drain")
                    draining = True

                self._queue_drained.wait()

            if draining:
                logging.info("Metrics queue drained")

            if self._queue is None:
                self._queue = metrics
                self._posted_ts = time.monotonic()
                self._metric_posted.notify_all()
            else:
                self._queue.extend(metrics)

    def close(self, wait: bool = True) -> None:
        with self._lock:
            if not self._terminated:
                self._terminated = True
                self._metric_posted.notify_all()

        if not wait:
            return

        self._thread.join()

        with self._lock:
            exception = self._exception
            if exception is not None:
                self._exception = None
                raise exception


class MlflowMetricsWriter:
    def __init__(self, run_id: str):
        self._closed = True
        self._async_writer = _AsyncWriter(client=MlflowClient(), run_id=run_id)
        self._closed = False

    def __del__(self):
        if not getattr(self, "_closed", True):
            self._async_writer.close(False)
            logging.warning("Metrics writer was not closed explicitly; closing asynchronously in destructor")

    def add_metrics(self, metrics: Mapping[str, Any], *, step: int | None = None) -> None:
        if not metrics:
            return

        timestamp = int(time.time() * 1000)
        metric_step = 0 if step is None else step
        batch = [
            Metric(key=name, value=float(value), timestamp=timestamp, step=metric_step)
            for name, value in metrics.items()
        ]
        self._async_writer.post_metrics(batch)

    def add_scalar(self, name: str, value: Any, step: int | None = None) -> None:
        self.add_metrics({name: value}, step=step)

    def flush(self) -> None:
        if not self._closed:
            try:
                self._async_writer.close()
            finally:
                self._closed = True
