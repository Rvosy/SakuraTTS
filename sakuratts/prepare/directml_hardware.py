"""Read DXGI adapter order and Windows display-driver inventory for evidence."""

import json
import subprocess


def capture_hardware(device_id=0):
    from sakuratts.backends.directml.devices import list_adapters
    result = {"api": "IDXGIFactory.EnumAdapters / IDXGIAdapter.GetDesc", "device_id": device_id,
              "scope": "Hardware observed for this experiment; other adapters and driver versions are not validated"}
    try:
        description = next((item for item in list_adapters() if item["device_id"] == device_id), None)
    except (OSError, RuntimeError) as error:
        result["adapter_inventory_error"] = f"{type(error).__name__}: {error}"
        return result
    if description is None:
        result["adapter_inventory_error"] = "Selected DirectML adapter was not found in the DXGI inventory"
        return result
    result.update(description)
    command = ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
               "Get-CimInstance Win32_VideoController | Select-Object Name,PNPDeviceID,DriverVersion | ConvertTo-Json -Compress"]
    try:
        process = subprocess.run(command, capture_output=True, text=True, check=True, timeout=20,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        inventory = json.loads(process.stdout) or []
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        result["driver_inventory_error"] = f"{type(error).__name__}: {error}"
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            result["driver_inventory_error"] += "\n" + error.stderr.strip()
        return result
    if isinstance(inventory, dict):
        inventory = [inventory]
    identifier = f"VEN_{description['vendor_id']:04X}&DEV_{description['pci_device_id']:04X}"
    matches = [item for item in inventory if identifier in item.get("PNPDeviceID", "").upper()]
    result.update(display_drivers=matches, driver_inventory=inventory)
    return result
