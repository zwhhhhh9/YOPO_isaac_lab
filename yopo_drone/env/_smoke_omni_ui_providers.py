#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import ctypes

with contextlib.suppress(ModuleNotFoundError):
    import isaacsim  # noqa: F401

from isaacsim import SimulationApp
import numpy as np


def main() -> None:
    simulation_app = SimulationApp({"headless": True})
    try:
        import omni.ui as ui
        import omni.gpu_foundation_factory._gpu_foundation_factory as gpu_factory

        provider_names = [name for name in dir(ui) if "Provider" in name or "Image" in name]
        for name in sorted(provider_names):
            print(name)
        for cls_name in ("ByteImageProvider", "RasterImageProvider", "DynamicTextureProvider"):
            cls = getattr(ui, cls_name, None)
            if cls is None:
                continue
            print(f"\n[{cls_name}]")
            for attr in sorted(name for name in dir(cls) if not name.startswith("_")):
                print(attr)
            with contextlib.suppress(Exception):
                print(f"{cls_name}.set_data_array.__doc__ = {cls.set_data_array.__doc__}")
            with contextlib.suppress(Exception):
                print(f"{cls_name}.set_data.__doc__ = {cls.set_data.__doc__}")
            with contextlib.suppress(Exception):
                print(f"{cls_name}.set_image_data.__doc__ = {cls.set_image_data.__doc__}")

        uint8_rgba = np.zeros((2, 2, 4), dtype=np.uint8)
        float_gray = np.linspace(0.0, 1.0, 4, dtype=np.float32).reshape(2, 2, 1)
        float_rgba = np.repeat(float_gray, 4, axis=2).astype(np.float32)

        texture_format_names = [name for name in dir(gpu_factory.TextureFormat) if name.isupper()]
        print(f"\n[TextureFormat names] {texture_format_names[:20]} ... total={len(texture_format_names)}")

        py_capsule_new = ctypes.pythonapi.PyCapsule_New
        py_capsule_new.restype = ctypes.py_object
        py_capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

        def _make_capsule(np_array: np.ndarray):
            return py_capsule_new(ctypes.c_void_p(np_array.ctypes.data), None, None)

        for provider_cls_name in ("ByteImageProvider", "DynamicTextureProvider"):
            provider_cls = getattr(ui, provider_cls_name, None)
            if provider_cls is None:
                continue
            print(f"\n[{provider_cls_name} call test]")
            try:
                provider = provider_cls()
                print("provider_ctor: ok")
            except Exception as exc:
                print(f"provider_ctor: {type(exc).__name__}: {exc}")
                continue
            for label, array in (
                ("uint8_rgba_set_data_array", uint8_rgba),
                ("float_rgba_set_data_array", float_rgba),
            ):
                try:
                    provider.set_data_array(array, [array.shape[1], array.shape[0]])
                    print(f"{label}: ok")
                except Exception as exc:
                    print(f"{label}: {type(exc).__name__}: {exc}")
            for label, array, texture_format in (
                ("uint8_rgba_set_image_data", uint8_rgba, gpu_factory.TextureFormat.R8G8B8A8_UNORM),
                ("float_rgba_set_image_data", float_rgba, gpu_factory.TextureFormat.R32G32B32A32_SFLOAT),
            ):
                try:
                    print(f"{label}: begin")
                    provider.set_image_data(_make_capsule(array), array.shape[1], array.shape[0], texture_format)
                    print(f"{label}: ok")
                except Exception as exc:
                    print(f"{label}: {type(exc).__name__}: {exc}")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
