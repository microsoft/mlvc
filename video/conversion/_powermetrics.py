# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import time
import plistlib
import threading
import subprocess
from contextlib import contextmanager
from .types import PowermetricsData, PowermetricsSample, PowermetricsStats


class PowerMetricsCollector:
    def __init__(self, save_raw_data: bool = True) -> None:
        self._save_raw_data = save_raw_data
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._stopped = False
        self._context = ""
        self._samples: list[PowermetricsSample] = []

    def start(self) -> None:
        if self._thread.is_alive():
            raise ValueError("Already running")
        if self._stopped:
            raise ValueError("Already stopped")
        self._thread.start()
        time.sleep(2)  # Wait for powermetrics to start

    def stop(self) -> None:
        self._stopped = True
        if self._thread is not None:
            self._thread.join()

    @contextmanager
    def set_context(self, context: str):
        previous = self._context
        self._context = context
        try:
            yield
        finally:
            self._context = previous

    def get_data(self) -> PowermetricsData:
        stats = {}
        for context in set(row.context for row in self._samples):
            if context == "":
                continue
            stats[context] = PowermetricsStats.from_samples(context, self._samples)
        return PowermetricsData(samples=self._samples, stats=stats)

    def clear(self) -> None:
        self._samples.clear()

    def _run(self):
        if sys.platform == "darwin":
            self._run_apple()
        else:
            print("WARN: Powermetrics is only supported on macOS")

    def _run_apple(self):
        cmd = [
            "sudo",
            "powermetrics",
            "--samplers",
            "cpu_power,gpu_power,ane_power",
            "--format",
            "plist",
            "--sample-rate",
            "1000",
        ]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            assert process.stdout is not None
            buffer = []
            while process.poll() is None:
                if self._stopped:
                    break

                line = process.stdout.readline()
                if line and line[0] == 0:
                    d = b"".join(buffer)
                    data = plistlib.loads(d)
                    row = PowermetricsSample(
                        timestamp=time.time(),
                        context=self._context,
                        gpu_power=1e-3 * data["processor"]["gpu_power"],
                        npu_power=1e-3 * data["processor"]["ane_power"],
                        cpu_power=1e-3 * data["processor"]["cpu_power"],
                        combined_power=1e-3 * data["processor"]["combined_power"],
                        raw_data=data if self._save_raw_data else {},
                    )
                    self._samples.append(row)

                    buffer.clear()
                    buffer.append(line[1:])
                else:
                    buffer.append(line)
        finally:
            process.terminate()
            time.sleep(0.2)
            if process.poll() is None:
                print("WARN: Killing powermetrics process...")
                process.kill()


if __name__ == "__main__":
    powermetrics = PowerMetricsCollector(save_raw_data=False)
    powermetrics.start()
    try:
        time.sleep(3)
        with powermetrics.set_context("step1"):
            time.sleep(3)
        time.sleep(3)
        with powermetrics.set_context("step2"):
            time.sleep(3)
    finally:
        powermetrics.stop()

    print(powermetrics.get_data())
