"""Read DXGI adapter order and Windows display-driver inventory for evidence."""

import json
import subprocess


def capture_hardware(device_id=0):
    from sakuratts.backends.directml.devices import list_adapters
    description = next((item for item in list_adapters() if item["device_id"] == device_id), None)
    if description is None:
        raise RuntimeError("Cannot inspect the selected DirectML adapter")
    command = ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
               "Get-CimInstance Win32_VideoController | Select-Object Name,PNPDeviceID,DriverVersion | ConvertTo-Json -Compress"]
    process = subprocess.run(command, capture_output=True, text=True, check=True, timeout=20,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    inventory = json.loads(process.stdout)
    if isinstance(inventory, dict):
        inventory = [inventory]
    identifier = f"VEN_{description['vendor_id']:04X}&DEV_{description['pci_device_id']:04X}"
    matches = [item for item in inventory if identifier in item.get("PNPDeviceID", "").upper()]
    return {"api": "IDXGIFactory.EnumAdapters / IDXGIAdapter.GetDesc", **description,
            "display_drivers": matches, "driver_inventory": inventory,
            "scope": "Hardware observed for this experiment; other adapters and driver versions are not validated"}
