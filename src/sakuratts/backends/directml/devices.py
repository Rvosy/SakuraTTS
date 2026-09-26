"""Read adapter identifiers in the DXGI order used by DirectML device_id."""

import ctypes
import sys
import uuid


def list_adapters():
    """Describe adapters without loading models or testing DirectML support."""
    if sys.platform != "win32":
        raise RuntimeError("DXGI adapter enumeration requires Windows")
    from ctypes import wintypes

    class LUID(ctypes.Structure):
        _fields_ = [("low", wintypes.DWORD), ("high", wintypes.LONG)]

    class Description(ctypes.Structure):
        _fields_ = [("name", wintypes.WCHAR * 128), ("vendor", wintypes.UINT),
                    ("device", wintypes.UINT), ("subsystem", wintypes.UINT), ("revision", wintypes.UINT),
                    ("video_memory", ctypes.c_size_t), ("system_memory", ctypes.c_size_t),
                    ("shared_memory", ctypes.c_size_t), ("luid", LUID)]

    def method(pointer, index, result, *arguments):
        table = ctypes.cast(pointer, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        return ctypes.WINFUNCTYPE(result, ctypes.c_void_p, *arguments)(table[index])

    guid = (ctypes.c_ubyte * 16).from_buffer_copy(uuid.UUID("7b7166ec-21c7-44ae-b21a-c9ae321ae369").bytes_le)
    factory = ctypes.c_void_p()
    create = ctypes.WinDLL("dxgi").CreateDXGIFactory
    create.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    create.restype = wintypes.LONG
    result = create(ctypes.byref(guid), ctypes.byref(factory))
    if result < 0:
        raise RuntimeError(f"Cannot enumerate DXGI adapters: HRESULT 0x{result & 0xffffffff:08x}")
    adapters = []
    try:
        enum = method(factory, 7, wintypes.LONG, wintypes.UINT, ctypes.POINTER(ctypes.c_void_p))
        while True:
            adapter = ctypes.c_void_p()
            device_id = len(adapters)
            result = enum(factory, device_id, ctypes.byref(adapter))
            if result & 0xffffffff == 0x887a0002:  # DXGI_ERROR_NOT_FOUND ends enumeration.
                break
            if result < 0:
                raise RuntimeError(f"Cannot inspect DXGI adapter {device_id}: HRESULT 0x{result & 0xffffffff:08x}")
            try:
                description = Description()
                result = method(adapter, 8, wintypes.LONG, ctypes.POINTER(Description))(adapter, ctypes.byref(description))
                if result < 0:
                    raise RuntimeError(f"Cannot read DXGI adapter {device_id}: HRESULT 0x{result & 0xffffffff:08x}")
                adapters.append({"device_id": device_id, "description": description.name,
                    "vendor_id": description.vendor, "pci_device_id": description.device,
                    "subsystem_id": description.subsystem, "revision": description.revision,
                    "luid": f"0x{description.luid.high & 0xffffffff:08x}_0x{description.luid.low:08x}",
                    "dedicated_video_memory_bytes": description.video_memory,
                    "dedicated_system_memory_bytes": description.system_memory,
                    "shared_system_memory_bytes": description.shared_memory})
            finally:
                method(adapter, 2, wintypes.ULONG)(adapter)
    finally:
        method(factory, 2, wintypes.ULONG)(factory)
    return adapters
