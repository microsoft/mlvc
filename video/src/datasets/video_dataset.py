# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import functools
import json
import numbers
import os
import random
import re
from typing import Union, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class FastVideoFolder(Dataset):
    def __init__(
        self,
        description_path,
        patch_h,
        patch_w,
        *,
        frame_num=5,
        crop_method="center",
        target_resolution: Union[None, int, str] = None,
        allow_tiling: bool = False,
        max_zoom_factor=1.0,
        min_zoom_factor=1.0,
        random_flip=False,
        frame_selection="random",
        max_frame_distance: Optional[int] = None,
        frame_distance: Union[None, int, str] = None,
        disable_random=False,
        precomputed_masks_path: Optional[str] = None,
        resampling_method: Optional[str] = None,
        thread_count: Optional[int] = None,
        padding_simulation_prob: float = 0.0,
        padding_simulation_min: float = 0.0,
        padding_simulation_max: float = 0.15,
        padding_simulation_color: str = "gray",
        padding_simulation_alignment: str = "bottom_right",
        target_description_path: Optional[str] = None,
        target_prob: float = 1.0,
    ):
        """
        crop_method could be 'center', 'random'
        frame_selection could be 'random', 'fix'
        target_prob controls per-sample probabilistic
        dual-target training when target_description_path is set
        """
        self.root_folder_path = os.path.dirname(description_path)

        seqs = self._load_dataset_description(description_path)
        self._validate_seqs(seqs, description_path)
        self.seqs = seqs

        self.target_root_folder_path: Optional[str] = None
        self.target_seqs: Optional[list] = None
        if target_description_path is not None:
            self.target_root_folder_path = os.path.dirname(target_description_path)
            target_seqs = self._load_dataset_description(target_description_path)
            self._validate_seqs(target_seqs, target_description_path)
            self._validate_target_alignment(seqs, target_seqs, description_path, target_description_path)
            self.target_seqs = target_seqs

        if not (0.0 <= target_prob <= 1.0):
            raise ValueError(f"target_prob must be in [0.0, 1.0], got {target_prob}")
        if self.target_seqs is None and target_prob != 1.0:
            raise ValueError(
                f"target_prob={target_prob} is only meaningful when "
                "target_description_path is set; got target_description_path=None"
            )
        self.target_prob = float(target_prob)

        self.dataset_length = len(self.seqs)
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.frame_num = frame_num
        self.crop_method = crop_method
        self.precomputed_masks_path = precomputed_masks_path

        try:
            if target_resolution is None:
                self.target_resolution = 0
            elif isinstance(target_resolution, int):
                self.target_resolution = target_resolution
            else:
                self.target_resolution = self._parse_probability_spec(int, target_resolution)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid target_resolution specification {target_resolution}") from e

        self.allow_tiling = bool(allow_tiling)

        assert max_zoom_factor >= min_zoom_factor
        self.max_zoom_factor = max_zoom_factor
        self.min_zoom_factor = min_zoom_factor
        self.enable_random_zoom = (
            max_zoom_factor > 1.01 or max_zoom_factor < 0.99 or min_zoom_factor > 1.01 or min_zoom_factor < 0.99
        )
        self.random_flip = random_flip

        if isinstance(frame_distance, int):
            if frame_distance <= 0:
                raise ValueError(f"Invalid frame_distance: {frame_distance}")
        elif frame_distance:
            try:
                frame_distance = self._parse_probability_spec(self._parse_positive_int, frame_distance)  # type: ignore[assignment]
            except (TypeError, ValueError) as e:
                raise ValueError(f"Invalid frame_distance specification {frame_distance}") from e

            if isinstance(frame_distance, tuple):
                # convert frame_distance list to tensor
                frame_distance = torch.tensor(frame_distance[0], dtype=torch.int), *frame_distance[1:]  # type: ignore[assignment]
        else:
            frame_distance = None

        if max_frame_distance is not None:
            if frame_distance is not None:
                raise ValueError("max_frame_distance and frame_distance are mututally exclusive")
            if not isinstance(max_frame_distance, numbers.Integral) or max_frame_distance <= 0:
                raise ValueError(f"Invalid max_frame_distance: {max_frame_distance}")
        elif frame_distance is None:
            max_frame_distance = 6

        if frame_selection in ("random2",):
            if frame_distance is None:
                assert max_frame_distance is not None
                frame_distance = (  # type: ignore[assignment]
                    torch.arange(1, max_frame_distance + 1),
                    torch.distributions.Categorical(max_frame_distance),  # type: ignore[arg-type]
                )
            elif isinstance(frame_distance, tuple):
                max_frame_distance = int(frame_distance[0].max().item())  # type: ignore[union-attr]
            else:
                max_frame_distance = int(frame_distance)

            self.frame_distance = frame_distance
            self.max_frame_distance = max_frame_distance
        else:
            if frame_distance is not None:
                raise ValueError(
                    f"frame distance probability distribution is not supported by {frame_selection} method"
                )

            self.max_frame_distance = max_frame_distance

        self.frame_selection = frame_selection

        try:
            if resampling_method:
                self.resampling_method = self._parse_probability_spec(self._parse_resampling_method, resampling_method)
            else:
                if self.enable_random_zoom:
                    self.resampling_method = Image.Resampling.BILINEAR
                else:
                    self.resampling_method = Image.Resampling.BOX
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid resampling_method specification {resampling_method}") from e

        self.thread_count = max(1, min(thread_count or 1, self.frame_num))
        self._thread_pool = None

        self.padding_simulation_prob = padding_simulation_prob

        if padding_simulation_min < 0 or padding_simulation_max < 0:
            raise ValueError("padding_simulation_min and padding_simulation_max must be non-negative")
        if padding_simulation_min > padding_simulation_max:
            raise ValueError("padding_simulation_min cannot be greater than padding_simulation_max")
        self.padding_simulation_min = padding_simulation_min
        self.padding_simulation_max = padding_simulation_max

        if padding_simulation_color == "black":
            self.padding_simulation_value = 0.0
        elif padding_simulation_color == "gray":
            self.padding_simulation_value = 0.5
        else:
            raise ValueError(f"Invalid padding_simulation_color: {padding_simulation_color}, must be 'black' or 'gray'")

        if padding_simulation_alignment not in ("bottom_right", "top_left", "both", "random"):
            raise ValueError(
                f"Invalid padding_simulation_alignment: {padding_simulation_alignment}, "
                "must be 'bottom_right', 'top_left', 'both', or 'random'"
            )
        self.padding_simulation_alignment = padding_simulation_alignment

        self.disable_random = disable_random

        if self.disable_random:
            self.crop_method = "center"
            self.max_zoom_factor = 1.0
            self.min_zoom_factor = 1.0
            self.enable_random_zoom = False
            self.random_flip = False
            self.frame_selection = "fix"
            self.max_frame_distance = 1
            self.padding_simulation_prob = 0.0
            self.target_prob = 1.0

    def _load_dataset_description(self, description_path):
        """
        Supports both the old list-only format and the newer dict
        with shared frames list.
        """
        with open(description_path, "rt", encoding="utf-8") as json_file:
            datasets = json.load(json_file)

        if isinstance(datasets, list):
            seqs = datasets
        elif isinstance(datasets, dict):
            if "seqs" not in datasets or "frames" not in datasets:
                raise ValueError(f"Both 'seqs' and 'frames' keys required in {description_path}")
            seqs = datasets["seqs"]
            frames = datasets["frames"]
            for s in seqs:
                s["frames"] = frames
        else:
            raise ValueError(f"Invalid dataset description file: {description_path}")

        return seqs

    def _validate_seqs(self, seqs, description_path):
        if len(seqs) == 0:
            raise ValueError(f"Empty dataset description file: {description_path}")

        required = {"path", "seq_length", "width", "height", "frames"}
        for i, seq in enumerate(seqs):
            missing = required - seq.keys()
            if missing:
                raise ValueError(f"Sequence #{i} missing keys {missing} in {description_path}")

    @staticmethod
    def _validate_target_alignment(input_seqs, target_seqs, input_path, target_path):
        if len(input_seqs) != len(target_seqs):
            raise ValueError(
                f"Target description has {len(target_seqs)} seqs but input has "
                f"{len(input_seqs)} ({input_path} vs {target_path})"
            )
        for i, (a, b) in enumerate(zip(input_seqs, target_seqs)):
            for key in ("path", "seq_length", "width", "height"):
                if a[key] != b[key]:
                    raise ValueError(
                        f"Target seq #{i} differs from input on key {key!r}: "
                        f"input={a[key]!r}, target={b[key]!r} "
                        f"({input_path} vs {target_path})"
                    )
            if a["frames"] != b["frames"]:
                raise ValueError(f"Target seq #{i} 'frames' list differs from input ({input_path} vs {target_path})")

    def __getstate__(self):
        state = self.__dict__.copy()
        # _thread_pool is not serializable
        state["_thread_pool"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    @staticmethod
    def _parse_probability_spec(parse_value, spec):
        value_list = list()
        weight_list = list()
        spec = spec.strip()
        for value in re.split(r"\s*[,|\s]\s*", spec):
            value, sep, weight = value.partition("/")

            value_list.append(parse_value(value))
            if not sep:
                weight = 1
            else:
                weight = float(weight)
                if weight <= 0:
                    raise ValueError(f"invalid probability weight: {weight}")

            weight_list.append(weight)

        if len(value_list) == 0:
            raise ValueError("no values are specified")

        if len(value_list) == 1:
            return value_list[0]
        else:
            weight_list = torch.tensor(weight_list, dtype=torch.float)
            weight_list /= weight_list.sum()
            return value_list, torch.distributions.Categorical(weight_list)

    @staticmethod
    def _parse_resampling_method(value: str):
        lvalue = value.lower()
        for method in Image.Resampling:
            if method._name_.lower() == lvalue:
                return method

        raise ValueError(f"Unknown resampling method: {value}")

    @staticmethod
    def _parse_positive_int(value: str):
        result = int(value)
        if result <= 0:
            raise ValueError("value must be positive")
        return result

    @staticmethod
    def _sample_categorical(values_and_distribution):
        if not isinstance(values_and_distribution, tuple):
            return values_and_distribution
        values, distribution = values_and_distribution
        if len(values) == 1:
            return values
        return values[distribution.sample()]

    def set_frame_num(self, frame_num):
        self.frame_num = frame_num

    def get_frame_num(self):
        return self.frame_num

    def set_patch_size(self, patch_width, patch_height):
        self.patch_w = patch_width
        self.patch_h = patch_height

    def get_patch_size(self):
        return self.patch_w, self.patch_h

    def set_dataset_length(self, dataset_length):
        self.dataset_length = min(self.dataset_length, dataset_length)

    def get_dataset_length(self):
        return self.dataset_length

    def _ensure_thread_pool(self):
        if self._thread_pool is None:
            import concurrent.futures

            self._thread_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.thread_count, thread_name_prefix="video_dataset"
            )
        return self._thread_pool

    def _make_process_frame_partial(self, *, augparams, is_mask=False):
        """Build a functools.partial of `_process_frame` with the shared
        augmentation params. Used for input frames, target frames, and
        precomputed mask frames (the last via `is_mask=True`)."""
        return functools.partial(
            self._process_frame,
            resampling_method=augparams["resampling_method"],
            resize_size=augparams["resize_size"],
            flip=augparams["flip"],
            n_tile_w=augparams["n_tile_w"],
            n_tile_h=augparams["n_tile_h"],
            pad_size=augparams["pad_size"],
            x=augparams["x"],
            y=augparams["y"],
            width=self.patch_w,
            height=self.patch_h,
            is_mask=is_mask,
        )

    def _load_seq_frames_sync(self, *, seq, seq_root_folder_path, img_indexes, augparams):
        """Sequentially load and concatenate `len(img_indexes)` frames from
        one sequence on the calling thread. Returns the concatenated
        np.uint8 array."""
        process_frame = self._make_process_frame_partial(augparams=augparams)
        frames = []
        for img_index in img_indexes:
            img_path = os.path.join(seq_root_folder_path, seq["path"], seq["frames"][img_index])
            frames.append(process_frame(img_path))
        return np.concatenate(frames, axis=0)

    def _submit_seq_frame_futures(self, *, seq, seq_root_folder_path, img_indexes, augparams):
        """Submit `len(img_indexes)` `_process_frame` futures."""
        thread_pool = self._ensure_thread_pool()
        process_frame = self._make_process_frame_partial(augparams=augparams)
        return [
            thread_pool.submit(
                process_frame,
                os.path.join(seq_root_folder_path, seq["path"], seq["frames"][idx]),
            )
            for idx in img_indexes
        ]

    @staticmethod
    def _collect_frame_futures(futures):
        """Await frame futures and concatenate their np.uint8 results."""
        return np.concatenate([f.result() for f in futures], axis=0)

    def __getitem__(self, index: int):
        seq = self.seqs[index]
        height = seq["height"]
        width = seq["width"]
        seq_path = seq["path"]

        img_indexes = []
        if self.frame_selection == "fix":
            img_indexes = range(0, self.frame_num)
        elif self.frame_selection == "random":
            if self.frame_num < seq["seq_length"]:
                img_indexes = random.sample(range(0, seq["seq_length"]), self.frame_num)
                is_reverse_order = random.choice([True, False])
                img_indexes.sort(reverse=is_reverse_order)
                assert self.max_frame_distance is not None
                for i in range(1, len(img_indexes), 1):
                    pre_index = img_indexes[i - 1]
                    cur_index = img_indexes[i]
                    if is_reverse_order:
                        if cur_index < pre_index - self.max_frame_distance:
                            cur_index = random.randint(pre_index - self.max_frame_distance, pre_index - 1)
                            img_indexes[i] = cur_index
                    else:
                        if cur_index > pre_index + self.max_frame_distance:
                            cur_index = random.randint(pre_index + 1, pre_index + self.max_frame_distance)
                            img_indexes[i] = cur_index
            else:
                increasing = True
                frame_index = 0
                while len(img_indexes) < self.frame_num:
                    img_indexes.append(frame_index)
                    if increasing:
                        if frame_index == seq["seq_length"] - 1:
                            frame_index -= 1
                            increasing = False
                        else:
                            frame_index += 1
                    elif not increasing:
                        if frame_index == 0:
                            frame_index += 1
                            increasing = True
                        else:
                            frame_index -= 1
        elif self.frame_selection == "random2":
            img_indexes = self._sample_frame_indices_random2(seq["seq_length"])
        else:
            assert False

        source_width, source_height = width, height

        target_resolution = self._sample_categorical(self.target_resolution)
        if target_resolution:
            if not self.allow_tiling:
                target_resolution = max(target_resolution, self.patch_h, self.patch_w)
            downsample_factor = max(1, min(width, height) // target_resolution)
            if downsample_factor != 1:
                width //= downsample_factor
                height //= downsample_factor

        if self.enable_random_zoom:
            zoom_factor = random.uniform(self.min_zoom_factor, self.max_zoom_factor)
            width = int(width * zoom_factor)
            height = int(height * zoom_factor)

        flip = False
        if self.random_flip:
            flip = random.choice([True, False])

        resize_width = width
        resize_height = height

        n_tile_w = n_tile_h = 1
        if self.allow_tiling:
            if width < self.patch_w:
                n_tile_w = (self.patch_w + width - 1) // width
                width *= n_tile_w
            if height < self.patch_h:
                n_tile_h = (self.patch_h + height - 1) // height
                height *= n_tile_h

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

        resampling_method = None
        if (source_width, source_height) != (width, height):
            resampling_method = self._sample_categorical(self.resampling_method)

        augparams = dict(
            resampling_method=resampling_method,
            resize_size=(resize_width, resize_height),
            flip=flip,
            n_tile_w=n_tile_w,
            n_tile_h=n_tile_h,
            pad_size=pad_size,
            x=x,
            y=y,
        )

        mask_data = []
        target_data_arr = None
        target_futures = None

        if self.target_seqs is not None:
            use_target = self.target_prob == 1.0 or random.random() < self.target_prob
        else:
            use_target = False

        if self.thread_count <= 1:
            video_data_arr = self._load_seq_frames_sync(
                seq=seq,
                seq_root_folder_path=self.root_folder_path,
                img_indexes=img_indexes,
                augparams=augparams,
            )
            if self.precomputed_masks_path is not None:
                process_mask = self._make_process_frame_partial(augparams=augparams, is_mask=True)
                mask_path_root = os.path.join(self.root_folder_path, self.precomputed_masks_path, seq_path)
                for img_index in img_indexes:
                    mask_path = os.path.join(mask_path_root, seq["frames"][img_index] + "_mask.png")
                    mask_data.append(process_mask(mask_path))
        else:
            thread_pool = self._ensure_thread_pool()
            input_futures = self._submit_seq_frame_futures(
                seq=seq,
                seq_root_folder_path=self.root_folder_path,
                img_indexes=img_indexes,
                augparams=augparams,
            )

            mask_futures = None
            if self.precomputed_masks_path is not None:
                process_mask = self._make_process_frame_partial(augparams=augparams, is_mask=True)
                mask_path_root = os.path.join(self.root_folder_path, self.precomputed_masks_path, seq_path)
                mask_futures = [
                    thread_pool.submit(
                        process_mask,
                        os.path.join(mask_path_root, seq["frames"][idx] + "_mask.png"),
                    )
                    for idx in img_indexes
                ]

            if use_target:
                target_futures = self._submit_seq_frame_futures(
                    seq=self.target_seqs[index],
                    seq_root_folder_path=self.target_root_folder_path,
                    img_indexes=img_indexes,
                    augparams=augparams,
                )

            video_data_arr = self._collect_frame_futures(input_futures)
            if mask_futures is not None:
                for f in mask_futures:
                    mask_data.append(f.result())

        video_tensor = torch.as_tensor(video_data_arr.astype(np.float32) / 255.0)

        # Sample the padding mask once so the same synthetic letterboxing can be
        # applied to both the input and (when present) the restored target.
        pad_mask = None
        if self.padding_simulation_prob > 0 and random.random() < self.padding_simulation_prob:
            _, h, w = video_tensor.shape
            pad_mask = self._sample_pad_mask(h, w)
            if pad_mask is not None:
                video_tensor[:, pad_mask] = self.padding_simulation_value
                if mask_data:
                    pad_mask_np = pad_mask.numpy()
                    for mask in mask_data:
                        if mask is not None:
                            mask[:, pad_mask_np] = 0

        result = {"video": video_tensor}

        if self.target_seqs is not None:
            if not use_target:
                result["target_video"] = video_tensor
            else:
                if self.thread_count <= 1:
                    target_data_arr = self._load_seq_frames_sync(
                        seq=self.target_seqs[index],
                        seq_root_folder_path=self.target_root_folder_path,
                        img_indexes=img_indexes,
                        augparams=augparams,
                    )
                else:
                    target_data_arr = self._collect_frame_futures(target_futures)
                target_tensor = torch.as_tensor(target_data_arr.astype(np.float32) / 255.0)
                if pad_mask is not None:
                    target_tensor[:, pad_mask] = self.padding_simulation_value
                result["target_video"] = target_tensor
        if self.precomputed_masks_path is not None:
            if any(m is None for m in mask_data):
                raise FileNotFoundError(
                    f"Some masks are None for sequence {seq_path} at index {index}. "
                    "Check if the precomputed masks path is correct."
                )
            else:
                mask_data = np.concatenate(mask_data, axis=0)
                mask_tensor = torch.as_tensor(mask_data.astype(np.float32) / 255.0)
                result["mask"] = mask_tensor
        return result

    def _sample_pad_mask(self, h: int, w: int) -> Optional[torch.Tensor]:
        """Sample a padding mask (HxW bool, True where padding is applied), or None.

        The mask is drawn once per item so the caller can apply identical
        synthetic letterboxing to both the input and the restored target.
        """
        pad_frac_h = random.uniform(self.padding_simulation_min, self.padding_simulation_max)
        pad_frac_w = random.uniform(self.padding_simulation_min, self.padding_simulation_max)

        pad_h = int(h * pad_frac_h)
        pad_w = int(w * pad_frac_w)

        if pad_h == 0 and pad_w == 0:
            return None

        alignment = self.padding_simulation_alignment
        if alignment == "random":
            alignment = random.choice(("bottom_right", "top_left", "both"))

        if alignment == "bottom_right":
            pad_top, pad_left = 0, 0
        elif alignment == "top_left":
            pad_top, pad_left = pad_h, pad_w
        else:
            pad_top = pad_h // 2
            pad_left = pad_w // 2

        pad_bottom = pad_h - pad_top
        pad_right = pad_w - pad_left

        # true where padding should be applied
        pad_mask = torch.zeros(h, w, dtype=torch.bool)
        if pad_top > 0:
            pad_mask[:pad_top, :] = True
        if pad_bottom > 0:
            pad_mask[h - pad_bottom :, :] = True
        if pad_left > 0:
            pad_mask[:, :pad_left] = True
        if pad_right > 0:
            pad_mask[:, w - pad_right :] = True

        return pad_mask

    def _sample_frame_indices_random2(self, seq_length: int):
        scalar_shape = tuple()
        if self.frame_num >= seq_length:
            img_indexes = (seq_length - 1) * torch.randint(2, scalar_shape) + torch.arange(self.frame_num)
            # length of backward sequence (without first and last frame)
            back_length = seq_length - 2
            # make it cycle from -back_length to seq_length - 1
            img_indexes = (img_indexes + back_length) % (back_length + seq_length) - back_length
            # get real indices
            return img_indexes.abs()
        elif (self.max_frame_distance is not None and self.max_frame_distance <= 1) or self.frame_num == 1:
            img_indexes = torch.randint(seq_length - self.frame_num, scalar_shape) + torch.arange(self.frame_num)
            if torch.randint(2, scalar_shape) != 0:
                img_indexes = img_indexes.flip(0)
            return img_indexes
        else:
            if not isinstance(self.frame_distance, tuple):
                distances = torch.fill_(torch.empty(self.frame_num - 1, dtype=torch.int), self.frame_distance)  # type: ignore[arg-type]
            else:
                values, distribution = self.frame_distance
                distances = values[distribution.sample((self.frame_num - 1,))]

            img_indexes = torch.empty(self.frame_num, dtype=torch.int)
            img_indexes[0] = 0
            img_indexes[1:] = distances.cumsum_(0)

            remainder = seq_length - img_indexes[-1].item() - 1
            if remainder > 0:
                img_indexes += torch.randint(int(remainder), scalar_shape)
            elif remainder < 0:
                # clip indices
                img_indexes = torch.minimum(img_indexes, torch.arange(seq_length - self.frame_num, seq_length))
                # and shuffle distances
                distances = img_indexes[1:] - img_indexes[:-1]
                distances = distances[torch.randperm(len(distances))]
                img_indexes[1:] = distances.cumsum_(0)

            if torch.randint(2, scalar_shape) != 0:
                img_indexes = img_indexes.flip(0)

            return img_indexes

    def _process_frame(
        self,
        img_path,
        *,
        resize_size,
        resampling_method,
        flip,
        n_tile_w,
        n_tile_h,
        pad_size,
        x,
        y,
        width,
        height,
        is_mask=False,
    ):
        if is_mask:
            try:
                img = Image.open(img_path).convert("L")
            except FileNotFoundError:
                return None
        else:
            img = Image.open(img_path).convert("RGB")

        if resampling_method is not None:
            img = img.resize(resize_size, resampling_method)
        if flip:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if n_tile_w > 1 or n_tile_h > 1:
            img = self._tile(img, n_tile_w, n_tile_h)

        if is_mask:
            img = np.asarray(img)[np.newaxis, ...].astype(np.uint8)
        else:
            img = np.asarray(img).transpose(2, 0, 1).astype(np.uint8)
        if any(any(x > 0 for x in p) for p in pad_size):
            img = np.pad(img, pad_size, mode="constant")
        if (height, width) != img.shape[1:]:
            img = img[:, y : y + height, x : x + width]
        return img

    @staticmethod
    def _tile(img, n_tile_w, n_tile_h):
        new_img = Image.new("RGB", (img.width * n_tile_w, img.height * n_tile_h))
        srcs = ([img, Image.Transpose.FLIP_TOP_BOTTOM], [Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.ROTATE_180])
        for i in range(n_tile_w):
            x = i * img.width
            i = i % 2
            for j in range(n_tile_h):
                y = j * img.height
                j = j % 2

                s = srcs[i][j]
                if type(s) is Image.Transpose:
                    srcs[i][j] = s = img.transpose(s)

                new_img.paste(s, (x, y))

        return new_img

    def __len__(self):
        return self.dataset_length
