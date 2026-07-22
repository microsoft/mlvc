# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import functools
import logging
import random
import time

__all__ = ["retry"]


def retry(_func=None, *, max_tries=5):
    def decorator_retry(func):
        @functools.wraps(func)
        def wrapper_retry(*args, **kwargs):
            for attempt in range(max_tries - 1):
                # noinspection PyBroadException
                try:
                    return func(*args, **kwargs)
                except BaseException:
                    if attempt == 0:
                        if len(args) > 0:
                            logging.info(f"args: {args}")
                        if len(kwargs) > 0:
                            logging.info(f"kwargs: {kwargs}")
                    logging.warning(f"exception caught, retry #{attempt + 1}", exc_info=True)
                    time.sleep(random.random())

            return func(*args, **kwargs)

        return wrapper_retry

    if _func is None:
        return decorator_retry
    else:
        return decorator_retry(_func)
