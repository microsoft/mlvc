# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import json
import logging
import os
import pathlib

from .retry import retry

__all__ = ["MountPointResolver"]


class MountPointResolver:
    def __init__(self, args: argparse.Namespace):
        self._args = args

        store_regions = args.store_regions
        if store_regions is None or store_regions == "":
            store_regions = tuple()
        else:
            store_regions = store_regions.split(",")
        self._store_regions = store_regions

        self._region = None
        self._mount_point_map = dict()

    def resolve_mount(self, mount_name: str):
        mount_point = self._mount_point_map.get(mount_name)
        if mount_point is None:
            self._mount_point_map[mount_name] = mount_point = self._resolve_mount(mount_name)

        return mount_point

    def _resolve_mount(self, mount_name: str):
        mount_list = getattr(self._args, mount_name)
        if mount_list is None:
            return None
        if len(mount_list) == 1:
            return mount_list[0]

        store_regions = self._store_regions
        if len(mount_list) != len(store_regions):
            raise ValueError(
                f"Number of {mount_name} mount points ({len(mount_list)}) does not match"
                f" number of store regions ({len(store_regions)})"
            )

        default_mount = None
        default_region = None
        current_region = self.resolve_region()

        for region, mount_point in zip(self._store_regions, mount_list):
            if region == current_region:
                logging.info(f"Using {region} mount for {mount_name}")
                return pathlib.Path(mount_point)

            if default_mount is None:
                default_region = region
                default_mount = mount_point

        if default_mount is not None:
            logging.warning(
                f"No region mount point for {mount_name} in {current_region},"
                f" using default mount for {default_region} region"
            )
        else:
            default_mount = os.getcwd()

        return pathlib.Path(default_mount)

    def resolve_region(self):
        region = self._region
        if region is None:
            region = self._resolve_region()
            if region is None:
                region = "<local>"

            self._region = region

        return region

    @staticmethod
    @retry
    def _resolve_region():
        import urllib.request

        # see https://docs.microsoft.com/en-us/azure/virtual-machines/windows/instance-metadata-service
        request = urllib.request.Request(
            "http://169.254.169.254/metadata/instance?api-version=2021-02-01", headers=dict(Metadata="true")
        )
        with urllib.request.urlopen(request) as f:
            response = json.load(f)

        return response["compute"]["location"].lower()
