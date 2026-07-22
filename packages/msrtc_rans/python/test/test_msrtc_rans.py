# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os.path
import zipfile
from typing import Optional

import numpy as np

from msrtc.rans import RansVariant, RansEncoderStream, RansDecoderStream, EntropyEncoder, EntropyDecoder


def test_encode_decode_byte_0():
    pmfLengths = np.asarray([4, 6], dtype=np.int32)
    pmfOffsets = np.array([1, 2], dtype=np.int32)
    pmfTable = np.asarray([1, 3, 1, 1, 1, 3, 5, 3, 1, 1], dtype=np.int32)

    encoder = EntropyEncoder(pmfLengths=pmfLengths, pmfOffsets=pmfOffsets, pmfTable=pmfTable)
    values = np.asarray([-2, 1, 0, 1], dtype=np.int32)
    indices = np.asarray([0, 1, 0, 1], dtype=np.int32)
    data = bytes(encoder.encode(indices, values))

    decoder = EntropyDecoder(pmfLengths=pmfLengths, pmfOffsets=pmfOffsets, pmfTable=pmfTable)
    decoded_values = np.empty_like(values)
    decoder.decode(decoded_values, indices, data)

    assert np.all(values == decoded_values)


def test_encode_decode_64_0():
    pmfLengths = np.asarray([4, 6], dtype=np.int32)
    pmfOffsets = np.array([1, 2], dtype=np.int32)
    pmfTable = np.asarray([1, 3, 1, 1, 1, 3, 5, 3, 1, 1], dtype=np.int32)

    encoder = EntropyEncoder(
        pmfLengths=pmfLengths, pmfOffsets=pmfOffsets, pmfTable=pmfTable, variant=RansVariant.Rans64
    )
    values = np.asarray([-2, 1, 0, 1], dtype=np.int32)
    indices = np.asarray([0, 1, 0, 1], dtype=np.int32)
    data = bytes(encoder.encode(indices, values))

    decoder = EntropyDecoder(
        pmfLengths=pmfLengths, pmfOffsets=pmfOffsets, pmfTable=pmfTable, variant=RansVariant.Rans64
    )
    decoded_values = np.empty_like(values)
    decoder.decode(decoded_values, indices, data)

    assert np.all(values == decoded_values)


def test_decode_unaligned():
    pmfLengths = np.asarray([4, 6], dtype=np.int32)
    pmfOffsets = np.array([1, 2], dtype=np.int32)
    pmfTable = np.asarray([1, 3, 1, 1, 1, 3, 5, 3, 1, 1], dtype=np.int32)

    encoder = EntropyEncoder(
        pmfLengths=pmfLengths, pmfOffsets=pmfOffsets, pmfTable=pmfTable, variant=RansVariant.Rans64
    )
    values = np.asarray([-2, 1, 0, 1], dtype=np.int32)
    indices = np.asarray([0, 1, 0, 1], dtype=np.int32)
    unalignedData = memoryview(b"\x00" + encoder.encode(indices, values))[1:]

    decoder = EntropyDecoder(
        pmfLengths=pmfLengths, pmfOffsets=pmfOffsets, pmfTable=pmfTable, variant=RansVariant.Rans64
    )
    decoded_values = np.empty_like(values)
    decoder.decode(decoded_values, indices, unalignedData)

    assert np.all(values == decoded_values)


def test_rans_encoder_stream_0():
    pmfLengths = np.asarray([4, 6], dtype=np.int32)
    pmfOffsets = np.array([1, 2], dtype=np.int32)
    pmfTable = np.asarray([1, 3, 1, 1, 1, 3, 5, 3, 1, 1], dtype=np.int32)

    encoder = EntropyEncoder(pmfLengths=pmfLengths, pmfOffsets=pmfOffsets, pmfTable=pmfTable)
    stream = RansEncoderStream(initialSize=0)

    values = np.arange(256, dtype=np.int32)
    indices = np.arange(256, dtype=np.int32) % 2
    encoder.push(stream, indices, values)
    data = bytes(stream.flush())

    decoder = EntropyDecoder(pmfLengths=pmfLengths, pmfOffsets=pmfOffsets, pmfTable=pmfTable)
    decoded_values = np.empty_like(values)
    decoder.decode(decoded_values, indices, data)

    assert np.all(values == decoded_values)


def test_encode_decode_multi_part_0():
    pmfLengths1 = np.asarray([4, 6], dtype=np.int32)
    pmfOffsets1 = np.array([1, 2], dtype=np.int32)
    pmfTable1 = np.asarray([1, 3, 1, 1, 1, 3, 5, 3, 1, 1], dtype=np.int32)

    encoder1 = EntropyEncoder(pmfLengths=pmfLengths1, pmfOffsets=pmfOffsets1, pmfTable=pmfTable1)

    values1 = np.asarray([-2, 1, 0, 1], dtype=np.int32)
    indices1 = np.asarray([0, 1, 0, 1], dtype=np.int32)

    pmfLengths2 = np.asarray([5], dtype=np.int32)
    pmfOffsets2 = np.array([1], dtype=np.int32)
    pmfTable2 = np.asarray([1, 3, 3, 1, 1], dtype=np.int32)

    encoder2 = EntropyEncoder(pmfLengths=pmfLengths2, pmfOffsets=pmfOffsets2, pmfTable=pmfTable2)

    values2 = np.asarray([-2, 1, 2], dtype=np.int32)
    indices2 = np.asarray([0, 0, 0], dtype=np.int32)

    encoder_stream = RansEncoderStream()
    encoder2.push(encoder_stream, indices2, values2)
    encoder1.push(encoder_stream, indices1, values1)

    data = bytes(encoder_stream.flush())

    decoder1 = EntropyDecoder(pmfLengths=pmfLengths1, pmfOffsets=pmfOffsets1, pmfTable=pmfTable1)
    decoder2 = EntropyDecoder(pmfLengths=pmfLengths2, pmfOffsets=pmfOffsets2, pmfTable=pmfTable2)

    decoder_stream = RansDecoderStream(data=data)

    decoded_values1 = np.empty_like(values1)
    decoder1.decode(decoded_values1, indices1, decoder_stream)
    assert np.all(values1 == decoded_values1)

    decoded_values2 = np.empty_like(values2)
    decoder2.decode(decoded_values2, indices2, decoder_stream)
    assert np.all(values2 == decoded_values2)

    decoder_stream.decodeEOF()


class ZipTestBundle:
    def __init__(self, filename: str):
        filename = os.path.join(os.path.dirname(__file__), filename)
        self._zip = zipfile.ZipFile(filename, "r")

    def close(self):
        self._zip.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def read_bytes(self, name):
        with self._zip.open(name, "r") as f:
            return f.read()

    def read_ndarray(self, name, dtype=np.int32):
        return np.frombuffer(self.read_bytes(name), dtype=dtype)


def run_encode_decode_on_zip_test_bundle(
    bundle_name: str,
    *,
    variant: Optional[RansVariant] = None,
    symbolBits: Optional[int] = None,
    bypassBits: Optional[int] = None,
):
    with ZipTestBundle(bundle_name) as bundle:
        pmfLengths = bundle.read_ndarray("pmf_lengths.np")
        pmfOffsets = bundle.read_ndarray("pmf_offsets.np")
        pmfTable = bundle.read_ndarray("pmf_table.np")
        indices = bundle.read_ndarray("indices.np")
        values = bundle.read_ndarray("values.np")

        bitstream = bundle.read_bytes("bitstream.bin")

    coderArgs = dict()
    if variant is not None:
        coderArgs["variant"] = variant
    if symbolBits is not None:
        coderArgs["symbolBits"] = symbolBits
    if bypassBits is not None:
        coderArgs["bypassBits"] = bypassBits

    encoder = EntropyEncoder(pmfLengths=pmfLengths, pmfOffsets=pmfOffsets, pmfTable=pmfTable, **coderArgs)
    data = bytes(encoder.encode(indices, values))
    assert data == bitstream, "encoded data mismatch"

    decoder = EntropyDecoder(pmfLengths=pmfLengths, pmfOffsets=pmfOffsets, pmfTable=pmfTable, **coderArgs)
    decoded_values = np.empty_like(values)
    decoder.decode(decoded_values, indices, data)

    assert np.all(values == decoded_values), "decoded values mismatch"


def test_gaussian_encoder0():
    run_encode_decode_on_zip_test_bundle("gaussian_encoder0.zip")


def test_bit_estimator_z0():
    run_encode_decode_on_zip_test_bundle("bit_estimator_z0.zip", bypassBits=2)
