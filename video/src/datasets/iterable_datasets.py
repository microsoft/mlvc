# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import deque
from typing import Iterable, Tuple, Callable, Any, Optional

import torch.utils.data

__all__ = ["NamedChunkDataset", "BatchBuilderDataset", "FramePairDataset"]


class NamedChunkDatasetIterator(torch.utils.data.IterableDataset):
    def __init__(self, chunks: Tuple[Tuple[str, torch.utils.data.IterableDataset], ...], *, prefetch: int = 0):
        super().__init__()
        self._chunks = chunks

        info = torch.utils.data.get_worker_info()
        if info is not None:
            self._chunk_cursor = info.id
            self._chunk_step = info.num_workers
        else:
            self._chunk_cursor = 0
            self._chunk_step = 1

        self._current_chunk = None

        if prefetch <= 0:
            self._queue = None
        else:
            self._queue = deque()
            for _ in range(prefetch):
                iterator = self._next_chunk()
                if iterator is None:
                    break

                self._queue.append(iterator)

        self._lookahead = self._next_sample()

    def _next_chunk(self):
        if self._chunk_cursor >= len(self._chunks):
            return None

        chunk_id = self._chunk_cursor
        _, ds = self._chunks[chunk_id]
        iterator = iter(ds)
        self._chunk_cursor += self._chunk_step
        return chunk_id, iterator

    def _next_sample(self):
        while True:
            if self._current_chunk is not None:
                try:
                    chunk_id, iterator = self._current_chunk
                    sample = next(iterator)
                    assert sample is not None
                    return chunk_id, sample
                except StopIteration:
                    self._current_chunk = None

            if self._queue is None:
                chunk = self._next_chunk()
            elif len(self._queue) > 0:
                next_chunk = self._next_chunk()
                chunk = self._queue.popleft()
                if next_chunk is not None:
                    self._queue.append(next_chunk)
            else:
                chunk = None

            if chunk is None:
                return None

            self._current_chunk = chunk

    def __iter__(self):
        return self

    def __next__(self):
        lookahead = self._lookahead
        if lookahead is None:
            raise StopIteration()

        chunk_id, sample = lookahead
        self._lookahead = lookahead = self._next_sample()

        is_last = lookahead is None or lookahead[0] != chunk_id
        return self._chunks[chunk_id][0], is_last, sample


class NamedChunkDataset(torch.utils.data.IterableDataset):
    def __init__(self, chunks: Iterable[Tuple[str, torch.utils.data.IterableDataset]], *, prefetch: int = 0):
        super().__init__()
        self.chunks = tuple(chunks)
        self.prefetch = prefetch

    def __iter__(self):
        return NamedChunkDatasetIterator(self.chunks, prefetch=self.prefetch)


def calc_tensor_size(x):
    if hasattr(x, "itemsize"):
        return x.size * x.itemsize
    if isinstance(x, dict):
        return sum(calc_tensor_size(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return sum(calc_tensor_size(v) for v in x)

    return 0


class BatchBuilderDatasetIterator(torch.utils.data.IterableDataset):
    def __init__(
        self, source: torch.utils.data.IterableDataset, *, batch_tensor_size: int, batch_size_step: int, collate_fn
    ):
        super().__init__()
        self.source = iter(source)
        self.batch_tensor_size = batch_tensor_size
        self.batch_size_step = batch_size_step
        self.collate_fn = collate_fn

    def __iter__(self):
        return self

    def __next__(self):
        source = self.source
        if source is None:
            raise StopIteration()

        batch = None
        try:
            sample = next(source)
            size = calc_tensor_size(sample)
            batch_size = max(self.batch_tensor_size // size, 1)
            if batch_size > self.batch_size_step:
                batch_size -= batch_size % self.batch_size_step

            batch = [sample]
            for _ in range(batch_size - 1):
                batch.append(next(source))
        except StopIteration:
            self.source = None
            if batch is None:
                raise

        return self.collate_fn(batch)


class BatchBuilderDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        source: torch.utils.data.IterableDataset,
        *,
        batch_tensor_size: int,
        batch_size_step: int,
        collate_fn: Optional[Callable[[Any], Any]] = None,
    ):
        super().__init__()
        self.source = source
        self.batch_tensor_size = batch_tensor_size
        self.batch_size_step = batch_size_step
        if collate_fn is None:
            collate_fn = torch.utils.data.default_collate
        self.collate_fn = collate_fn

    def __iter__(self):
        return BatchBuilderDatasetIterator(
            self.source,
            batch_tensor_size=self.batch_tensor_size,
            batch_size_step=self.batch_size_step,
            collate_fn=self.collate_fn,
        )


class FramePairDatasetIterator(torch.utils.data.IterableDataset):
    def __init__(self, source: torch.utils.data.IterableDataset):
        super().__init__()
        self.source = iter(source)
        self.last_frame = None

    def __iter__(self):
        return self

    def __next__(self):
        last_frame = self.last_frame
        if last_frame is None:
            last_frame = next(self.source)

        try:
            frame = next(self.source)
        except StopIteration:
            self.last_frame = None
            raise

        self.last_frame = frame
        return last_frame, frame


class FramePairDataset(torch.utils.data.IterableDataset):
    def __init__(self, source: torch.utils.data.IterableDataset):
        super().__init__()
        self.source = source

    def __iter__(self):
        return FramePairDatasetIterator(self.source)
