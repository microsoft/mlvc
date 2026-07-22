# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import json
import random
import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset


class ImageFolder(Dataset):
    def __init__(
        self, description_path, patch_h, patch_w, crop_method="random", random_flip=True, disable_random=False
    ):
        self.root_folder_path = os.path.dirname(description_path)
        self.dataset = list(self._read_dataset(description_path))

        self.dataset_length = len(self.dataset)
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.crop_method = crop_method
        self.random_flip = random_flip
        self.disable_random = disable_random

        if self.disable_random:
            self.crop_method = "center"
            self.random_flip = False

    def _read_dataset(self, description_path):
        with open(description_path) as json_file:
            dataset = json.load(json_file)

        for record in dataset:
            if isinstance(record, str):
                yield str
                continue

            if isinstance(record, dict):
                base_path = record.get("path")
                frame_list = record.get("frames")
                if isinstance(base_path, str) and isinstance(frame_list, list):
                    for f in frame_list:
                        yield os.path.join(base_path, f)

                    continue

            raise ValueError(f"{description_path}: Unrecognized dataset format")

    def set_patch_size(self, patch_width, patch_height):
        self.patch_w = patch_width
        self.patch_h = patch_height

    def get_patch_size(self):
        return self.patch_w, self.patch_h

    def set_dataset_length(self, dataset_length):
        self.dataset_length = min(self.dataset_length, dataset_length)

    def get_dataset_length(self):
        return self.dataset_length

    def __getitem__(self, index):
        image_path = os.path.join(self.root_folder_path, self.dataset[index])
        img = Image.open(image_path).convert("RGB")
        width, height = img.size

        pad_height = self.patch_h - height
        pad_width = self.patch_w - width
        pad_height = max(0, pad_height)
        pad_width = max(0, pad_width)
        pad_size = (
            (0, 0),
            (pad_height // 2, pad_height - pad_height // 2),
            (pad_width // 2, pad_width - pad_width // 2),
        )
        padded_height = height + pad_height
        padded_width = width + pad_width
        if self.crop_method == "center":
            y = (padded_height - self.patch_h) // 2
            x = (padded_width - self.patch_w) // 2
        elif self.crop_method == "random":
            y = random.randint(0, padded_height - self.patch_h)
            x = random.randint(0, padded_width - self.patch_w)
        else:
            assert False

        if self.random_flip and random.choice([True, False]):
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        img = np.array(img).transpose(2, 0, 1).astype(np.uint8)
        img = np.pad(img, pad_size, mode="constant")
        img = img[:, y : y + self.patch_h, x : x + self.patch_w]

        return torch.as_tensor(img.astype(np.float32) / 255.0, dtype=torch.float32)

    def __len__(self):
        return self.dataset_length
