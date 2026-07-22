# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional, Any

from .types import RansVariant

__all__ = ["RansEncoderStream", "RansDecoderStream", "EntropyEncoder", "EntropyDecoder"]

class RansEncoderStream:
    """
    rANS encoder stream
    """

    def __init__(
        self, variant: RansVariant = RansVariant.RansByte, *, initialSize: int = 4096, maxSizeStep: int = 1024 * 1024
    ):
        """
        Initialize rANS encoder stream

        :param variant: rANS variant
        :param initialSize: initial buffer size
        :param maxSizeStep: max buffer size step
        """
        pass

    def flush(self) -> memoryview:
        """
        Flushes encoding buffer and returns encoded message

        :return: encoded message
        """

        pass

    def reset(self) -> None:
        """
        Resets the stream
        """
        pass

class RansDecoderStream:
    """
    rANS decoder stream
    """

    def __init__(self, data: Optional[Any] = None, *, variant: RansVariant = RansVariant.RansByte):
        """
        Initialize rANS decoder stream

        :param data: encoded message data
        :param variant: rANS variant
        """
        pass

    def open(self, data) -> None:
        """
        Opens the stream using data

        :param data: encoded message data
        """
        pass

    def close(self) -> None:
        """
        Closes the stream
        """
        pass

    def isOpen(self) -> bool:
        """
        Checks whether stream is open

        :return: open status
        """
        pass

    def decodeEOF(self) -> None:
        """
        Check that decoding reached end of message and closes stream on success
        """
        pass

EntropyEncoder = Any
"""
Entropy encoder implmentation
"""

EntropyDecoder = Any
"""
Entropy decoder implmentation
"""
