# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Iterable, Optional, Sequence

import numpy as np
from torch.utils.data import Dataset


class WeightedDataset(Dataset):
    def __init__(self, datasets: Iterable[Dataset], weights: Optional[Sequence[float]], length: Optional[int] = None):
        self.datasets = tuple(datasets)
        # noinspection PyTypeChecker
        self.lengths = np.fromiter(
            (len(dataset) for dataset in self.datasets),  # type: ignore[arg-type]
            count=len(self.datasets),
            dtype=np.int32,
        )
        self.total_length = length if length is not None else self.lengths.sum()

        if weights is not None:
            w = np.asarray(weights)
            if len(self.datasets) != len(w):
                raise ValueError("Number of weights does not match number of datasets")
        else:
            w = self.lengths

        self.weights = w / w.sum()

    def __getitem__(self, index):
        _ = index

        if len(self.datasets) == 1:
            dataset_idx = 0
        else:
            dataset_idx = np.random.choice(len(self.datasets), p=self.weights)

        sample_index = np.random.randint(0, self.lengths[dataset_idx])
        return self.datasets[dataset_idx][sample_index]

    def __len__(self):
        return self.total_length
