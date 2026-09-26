"""Read DXGI adapter order and Windows display-driver inventory for evidence."""

import ctypes
import json
import subprocess
import sys
import uuid


def capture_hardware(device_id=0):
    if sys.platform != "win32":
        raise RuntimeError("DirectML hardware evidence requires Windows")
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
    factory, adapter = ctypes.c_void_p(), ctypes.c_void_p()
    create = ctypes.WinDLL("dxgi").CreateDXGIFactory
    create.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    create.restype = wintypes.LONG
    if create(ctypes.byref(guid), ctypes.byref(factory)) < 0:
        raise RuntimeError("Cannot enumerate DXGI adapters")
    try:
        enum = method(factory, 7, wintypes.LONG, wintypes.UINT, ctypes.POINTER(ctypes.c_void_p))
        if enum(factory, device_id, ctypes.byref(adapter)) < 0:
            raise RuntimeError("Cannot inspect the selected DirectML adapter")
        try:
            description = Description()
            if method(adapter, 8, wintypes.LONG, ctypes.POINTER(Description))(adapter, ctypes.byref(description)) < 0:
                raise RuntimeError("Cannot read the selected DXGI adapter description")
        finally:
            method(adapter, 2, wintypes.ULONG)(adapter)
    finally:
        method(factory, 2, wintypes.ULONG)(factory)
    command = ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
               "Get-CimInstance Win32_VideoController | Select-Object Name,PNPDeviceID,DriverVersion | ConvertTo-Json -Compress"]
    process = subprocess.run(command, capture_output=True, text=True, check=True, timeout=20,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    inventory = json.loads(process.stdout)
    if isinstance(inventory, dict):
        inventory = [inventory]
    identifier = f"VEN_{description.vendor:04X}&DEV_{description.device:04X}"
    matches = [item for item in inventory if identifier in item.get("PNPDeviceID", "").upper()]
    return {"api": "IDXGIFactory.EnumAdapters / IDXGIAdapter.GetDesc", "device_id": device_id,
            "description": description.name, "vendor_id": description.vendor, "pci_device_id": description.device,
            "subsystem_id": description.subsystem, "revision": description.revision,
            "display_drivers": matches, "driver_inventory": inventory,
            "scope": "Hardware observed for this experiment; other adapters and driver versions are not validated"}
