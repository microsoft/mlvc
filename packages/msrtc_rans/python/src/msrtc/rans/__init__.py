# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys

from . import _msrtc_rans
from .types import RansVariant

__all__ = ["RansVariant", "RansEncoderStream", "RansDecoderStream", "EntropyEncoder", "EntropyDecoder"]

RansEncoderStream = _msrtc_rans.RansEncoderStream
RansDecoderStream = _msrtc_rans.RansDecoderStream


class EntropyEncoder:
    """
    Entropy encoder
    """

    __slots__ = "_impl", "_variant"

    def __init__(
        self,
        *,
        pmfLengths,
        pmfOffsets,
        pmfTable,
        variant: RansVariant = RansVariant.RansByte,
        symbolBits=16,
        bypassBits=4,
    ):
        """
        Initialize entropy encoder

        :param pmfLengths: PMF table lengths (int32 1d-array, e.g. numpy.ndarray, torch.Tensor etc.)
        :param pmfOffsets: PMF table zero element offsets (int32 1d-array)
        :param pmfTable: concatenated PMF tables (int32 1d-array)
        :param variant: rANS variant to use (default RansByte)
        :param symbolBits: symbol bits (default 16)
        :param bypassBits: bypass bits (default 4)
        """
        if symbolBits <= 1 or symbolBits > sys.maxsize:
            raise ValueError("invalid symbolBits value")
        if bypassBits <= 1 or bypassBits > sys.maxsize:
            raise ValueError("invalid bypassBits value")

        self._impl = _msrtc_rans.EntropyEncoder(
            pmfLengths=pmfLengths,
            pmfOffsets=pmfOffsets,
            pmfTable=pmfTable,
            variant=variant,
            symbolBits=symbolBits,
            bypassBits=bypassBits,
        )
        self._variant = variant

    @property
    def variant(self):
        return self._variant

    def encode(self, indices, values) -> memoryview:
        """
        Encode message

        :param indices: PMF table indices (int32 1d-array)
        :param values: values (int32 1d-array)
        :return: entropy encoded message
        """
        stream = RansEncoderStream(self._variant)
        self._impl.encode(stream, indices, values)
        return stream.flush()

    def push(self, stream: RansEncoderStream, indices, values) -> None:
        """
        Push message into encoder stream
        NB: rANS requires messages to be pushed in reverse order during encoding

        :param stream: rANS encoder stream
        :param indices: PMF table indices (int32 1d-array)
        :param values: values (int32 1d-array)
        """
        return self._impl.encode(stream, indices, values)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        _ = memo
        return self.__copy__()


class EntropyDecoder:
    """
    Entropy Decoder
    """

    def __init__(
        self,
        *,
        pmfLengths,
        pmfOffsets,
        pmfTable,
        variant: RansVariant = RansVariant.RansByte,
        symbolBits=16,
        bypassBits=4,
    ):
        """
        Initialize entropy decoder

        :param pmfLengths: PMF table lengths (int32 1d-array, e.g. numpy.ndarray, torch.Tensor etc.)
        :param pmfOffsets: PMF table zero element offsets (int32 1d-array)
        :param pmfTable: concatenated PMF tables (int32 1d-array)
        :param variant: rANS variant to use (default RansByte)
        :param symbolBits: symbol bits (default 16)
        :param bypassBits: bypass bits (default 4)
        """
        if symbolBits <= 1 or symbolBits > sys.maxsize:
            raise ValueError("invalid symbolBits value")
        if bypassBits <= 1 or bypassBits > sys.maxsize:
            raise ValueError("invalid bypassBits value")

        self._impl = _msrtc_rans.EntropyDecoder(
            pmfLengths=pmfLengths,
            pmfOffsets=pmfOffsets,
            pmfTable=pmfTable,
            variant=variant,
            symbolBits=symbolBits,
            bypassBits=bypassBits,
        )

    def decode(self, values, indices, stream) -> None:
        """
        Encode message

        :param values: decoded values (output int32 1d-array)
        :param indices: PMF table indices (int32 1d-array)
        :param stream: message data (any buffer) or RansDecoderStream
        """
        return self._impl.decode(values, indices, stream)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        _ = memo
        return self.__copy__()
