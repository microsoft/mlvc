# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any

from .types import RansVariant

__all__ = ["EntropyDecoder", "EntropyEncoder", "RansDecoderStream", "RansEncoderStream"]

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

    def flush(self) -> memoryview:
        """
        Flushes encoding buffer and returns encoded message

        :return: encoded message
        """

    def reset(self) -> None:
        """
        Resets the stream
        """

class RansDecoderStream:
    """
    rANS decoder stream
    """

    def __init__(self, data: Any | None = None, *, variant: RansVariant = RansVariant.RansByte):
        """
        Initialize rANS decoder stream

        :param data: encoded message data
        :param variant: rANS variant
        """

    def open(self, data) -> None:
        """
        Opens the stream using data

        :param data: encoded message data
        """

    def close(self) -> None:
        """
        Closes the stream
        """

    def isOpen(self) -> bool:
        """
        Checks whether stream is open

        :return: open status
        """

    def decodeEOF(self) -> None:
        """
        Check that decoding reached end of message and closes stream on success
        """

type EntropyEncoder = Any
"""
Entropy encoder implmentation
"""

type EntropyDecoder = Any
"""
Entropy decoder implmentation
"""
