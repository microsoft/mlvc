# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import functools
import logging
import sys

__all__ = ["get_pipe_buffer_max_size", "set_pipe_buffer_size"]


@functools.lru_cache(maxsize=None)
def get_pipe_buffer_max_size():
    max_size = 65536
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/sys/fs/pipe-max-size", "rt", encoding="utf-8") as f:
                sz = int(f.read())

            if sz <= 0:
                raise ValueError("Invalid buffer size")

            max_size = sz
        except IOError:
            logging.warning("can not read /proc/sys/fs/pipe-max-size", exc_info=True)
        except ValueError:
            logging.warning("can not decode /proc/sys/fs/pipe-max-size", exc_info=True)

    return max_size


def set_pipe_buffer_size(fd, buffer_size):
    if sys.platform.startswith("linux"):
        from fcntl import fcntl  # type: ignore[attr-defined]

        F_SETPIPE_SZ = 1024 + 7
        F_GETPIPE_SZ = 1024 + 8

        try:
            sz = fcntl(fd, F_GETPIPE_SZ)
            if sz >= buffer_size:
                return

            buffer_size = min(buffer_size, get_pipe_buffer_max_size())
            if sz >= buffer_size:
                return

            fcntl(fd, F_SETPIPE_SZ, buffer_size)
        except OSError:
            logging.warning(f"failed to set pipe buffer size {buffer_size}", exc_info=True)
