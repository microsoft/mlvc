# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict

import torch.nn

__all__ = ["RenameModule_LoadHook"]


class RenameModule_LoadHook:
    def __init__(self, rename_map: Dict[str, str]):
        self.rename_map = rename_map

    def register(self, module: torch.nn.Module):
        # noinspection PyProtectedMember
        module._register_load_state_dict_pre_hook(self)

    def __call__(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):

        source_keys = dict()
        start = len(prefix)
        for key in state_dict.keys():
            if not key.startswith(prefix):
                continue

            dot = key.find(".", start)
            if dot == -1:
                dot = len(key)

            old_name = key[start:dot]
            if old_name not in self.rename_map:
                continue

            key_list = source_keys.get(old_name)
            if key_list is None:
                source_keys[old_name] = key_list = list()

            key_list.append(key)

        for old_name, key_list in source_keys.items():
            remainder = len(prefix) + len(old_name)
            new_prefix = prefix + self.rename_map[old_name]
            for key in key_list:
                state_dict[new_prefix + key[remainder:]] = state_dict.pop(key)
