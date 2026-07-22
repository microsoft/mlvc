# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import atexit
import threading
import importlib.metadata
from pathlib import Path
import faulthandler

faulthandler.enable()


class WindowsMLRuntime:
    def __init__(self) -> None:
        self._initialized = False
        self._lock = threading.Lock()
        self._win_app_sdk_handle: object | None = None

    @staticmethod
    def _is_winml_installed() -> bool:
        try:
            importlib.metadata.distribution("onnxruntime-windowsml")
            return True
        except importlib.metadata.PackageNotFoundError:
            return False

    @staticmethod
    def _fix_winrt_runtime() -> None:
        site_packages = Path(str(importlib.metadata.distribution("winrt-runtime").locate_file("")))
        dll_path = site_packages / "winrt" / "msvcp140.dll"
        if dll_path.exists():
            try:
                dll_path.unlink()
            except PermissionError:
                print(f"Warning: Could not remove conflicting DLL: {dll_path}")

    def ensure_initialized(self) -> None:
        with self._lock:
            if self._initialized:
                return

            if not self._is_winml_installed():
                self._initialized = True
                return

            print("Bootstrapping Windows App SDK (onnxruntime-windowsml detected)")
            self._fix_winrt_runtime()
            self._bootstrap()
            self._initialized = True

    def _shutdown(self) -> None:
        if self._win_app_sdk_handle is not None:
            self._win_app_sdk_handle.__exit__(None, None, None)  # type: ignore
            self._win_app_sdk_handle = None

    def _bootstrap(self) -> None:
        import winui3.microsoft.windows.applicationmodel.dynamicdependency.bootstrap as bootstrap  # type: ignore[import-not-found]
        from winui3 import (  # type: ignore[import-not-found]
            _winui3_microsoft_windows_applicationmodel_dynamicdependency_bootstrap as bootstrap_impl,
        )
        import winui3.microsoft.windows.ai.machinelearning as winml  # type: ignore[import-not-found]

        # Print Windows app SDK versions
        wasdk_version = importlib.metadata.version(
            "wasdk-Microsoft.Windows.ApplicationModel.DynamicDependency.Bootstrap"
        )
        winml_version = importlib.metadata.version("wasdk-Microsoft.Windows.AI.MachineLearning")
        print(
            f"Windows App SDK version: {wasdk_version}, Microsoft.Windows.AI.MachineLearning version: {winml_version}"
        )

        # Initialize Windows App SDK runtime via bootstrap
        print(
            f"WindowsML bootstrap initialize(product version={bootstrap_impl.RELEASE_VERSION},"
            f" minimum runtime version={self._unpack_runtime_version(bootstrap_impl.RUNTIME_VERSION)}..."
        )
        self._win_app_sdk_handle = bootstrap.initialize(
            options=bootstrap.InitializeOptions.ON_NO_MATCH_SHOW_UI,
        )
        self._win_app_sdk_handle.__enter__()  # type: ignore[union-attr]
        atexit.register(self._shutdown)

        # Ensure WindowsML execution providers are ready
        catalog = winml.ExecutionProviderCatalog.get_default()
        print("Ensure ready WindowsML execution providers:")
        for provider in catalog.find_all_providers():
            provider.ensure_ready_async().get()
            pkg = provider.package_id
            ver = pkg.version
            print(
                f"  - name={provider.name} ready_state={provider.ready_state.name} certification={provider.certification.name}"
                f" version={ver.major}.{ver.minor}.{ver.build}.{ver.revision} package={pkg.full_name}"
                f" library_path={provider.library_path}"
            )

        import onnxruntime as ort  # type: ignore[import-not-found]

        print(
            f"Loaded Windows App Runtime version: {self._get_runtime_version()}, ONNX Runtime version: {ort.__version__}"
        )

        # Register WindowsML execution providers to ONNX Runtime
        print("Registering WindowsML execution providers to ONNX Runtime:")
        for provider in catalog.find_all_providers():
            if provider.library_path == "":
                print(f"  - Skipping registration: {provider.name} (ready_state={provider.ready_state.name})")
                continue
            print(f"  - name={provider.name}")
            ort.register_execution_provider_library(provider.name, provider.library_path)

        # Print available ONNX Runtime providers and EP devices after registration
        print(f"Available providers: {ort.get_available_providers()}")
        print("Available EP devices:")
        for ep_device in ort.get_ep_devices():
            dev = ep_device.device
            print(
                f"  - ep_name={ep_device.ep_name} type={dev.type.name} device_id={dev.device_id}"
                f" vendor={dev.vendor} vendor_id={dev.vendor_id} metadata={dev.metadata}"
            )

    @staticmethod
    def _unpack_runtime_version(version: str) -> str:
        """Unpack MSIX runtime version like '8000.770.947.0' to a human-readable string."""
        from datetime import date, timedelta

        try:
            nppp, e, b, _r = (int(x) for x in version.split("."))
            minor = nppp // 1000
            patch = nppp % 1000
            build_date = date(2024, 1, 1) + timedelta(days=e)
            return f"{version} (x.{minor}.{patch}, built {build_date}, build #{b})"
        except Exception:
            return version

    @staticmethod
    def _get_runtime_version() -> str:
        """Query the loaded Windows App Runtime version via ctypes."""
        import ctypes

        try:
            dll = ctypes.WinDLL("Microsoft.WindowsAppRuntime.Insights.Resource.dll")  # type: ignore[reportAttributeAccessIssue]

            class _Version(ctypes.Structure):
                _fields_ = [
                    ("Major", ctypes.c_uint16),
                    ("Minor", ctypes.c_uint16),
                    ("Build", ctypes.c_uint16),
                    ("Revision", ctypes.c_uint16),
                    ("UInt64", ctypes.c_uint64),
                    ("DotQuadString", ctypes.c_wchar_p),
                ]

            class _Identity(ctypes.Structure):
                _fields_ = [
                    ("Publisher", ctypes.c_wchar_p),
                    ("PublisherId", ctypes.c_wchar_p),
                ]

            class _Runtime(ctypes.Structure):
                _fields_ = [("Identity", _Identity), ("Version", _Version)]

            class _Release(ctypes.Structure):
                _fields_ = [
                    ("Major", ctypes.c_uint16),
                    ("Minor", ctypes.c_uint16),
                    ("Patch", ctypes.c_uint16),
                    ("_padding", ctypes.c_uint16),
                    ("MajorMinor", ctypes.c_uint32),
                    ("Channel", ctypes.c_wchar_p),
                    ("VersionTag", ctypes.c_wchar_p),
                    ("VersionShortTag", ctypes.c_wchar_p),
                ]

            class _VersionInfo(ctypes.Structure):
                _fields_ = [("Release", _Release), ("Runtime", _Runtime)]

            func = dll.WindowsAppRuntime_GetVersionInfo
            func.restype = ctypes.POINTER(_VersionInfo)
            func.argtypes = []

            rt = func().contents.Runtime
            ver = rt.Version.DotQuadString or (
                f"{rt.Version.Major}.{rt.Version.Minor}.{rt.Version.Build}.{rt.Version.Revision}"
            )
            return WindowsMLRuntime._unpack_runtime_version(ver)
        except Exception as e:
            print(f"Warning: Could not query Windows App Runtime version info: {e}")
            return "unknown"


_instance = WindowsMLRuntime()


def ensure_initialized() -> None:
    _instance.ensure_initialized()
