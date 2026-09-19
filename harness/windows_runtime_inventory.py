"""Read-only file inventory for explicitly selected runtime, resource and cache roots.

Uses only the Python standard library. Logical sizes are not allocated disk space,
download sizes, loaded-module sizes, or evidence that duplicate DLLs can be removed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat


SCOPES = {"runtime", "shared_python", "model", "cache", "source"}


def normalized(path):
    return os.path.normcase(os.path.abspath(path))


def is_link(info):
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def parse_root(value):
    try:
        identity, path = value.split("=", 1)
        scope, name = identity.split(":", 1)
    except ValueError as error:
        raise ValueError("Use --root SCOPE:NAME=PATH") from error
    if scope not in SCOPES or not name or not path:
        raise ValueError("Root needs a supported scope, a name and a path: " + value)
    return {"name": name, "scope": scope, "path": normalized(path)}


def file_identity(info, path):
    # A zero inode is not a usable file identity on some filesystems.
    return ("inode", info.st_dev, info.st_ino) if info.st_ino else ("path", path)


def signature(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def sha256(path, expected):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        if signature(os.fstat(stream.fileno())) != signature(expected):
            raise RuntimeError("File changed before hashing: " + path)
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        if signature(os.fstat(stream.fileno())) != signature(expected):
            raise RuntimeError("File changed while hashing: " + path)
    return digest.hexdigest()


def component(root, relative):
    parts = Path(relative).parts
    lowered = [part.lower() for part in parts]
    if "site-packages" in lowered:
        index = lowered.index("site-packages") + 1
        package = parts[index] if index < len(parts) else "site-packages"
        if package == "nvidia" and index + 1 < len(parts):
            package += "/" + parts[index + 1]
        return root["name"] + "/" + package
    if parts and parts[0].lower() == "cuda":
        name = Path(relative).name.lower()
        family = next((prefix for prefix in ("cublas", "cudnn", "cudart", "cufft", "nvrtc", "nvjitlink")
                       if name.startswith(prefix)), "other")
        return root["name"] + "/cuda/" + family
    return root["name"] + "/" + (parts[0] if len(parts) > 1 else "root-files")


def totals(files):
    identities = {}
    hashes = {}
    for row in files:
        identities[row["storage_id"]] = row["bytes"]
        hashes[(row["sha256"], row["bytes"])] = row["bytes"]
    logical = sum(row["bytes"] for row in files)
    unique = sum(identities.values())
    content = sum(hashes.values())
    return {"file_count": len(files), "logical_bytes": logical,
            "unique_file_objects": len(identities), "unique_file_bytes": unique,
            "hardlink_alias_bytes": logical - unique, "unique_content_bytes": content,
            "duplicate_content_bytes_after_hardlinks": unique - content}


def inventory(roots):
    roots = [dict(root, path=normalized(root["path"])) for root in roots]
    if len({root["name"] for root in roots}) != len(roots):
        raise ValueError("Root names must be unique")
    for root in roots:
        info = os.lstat(root["path"])
        if is_link(info) or not stat.S_ISDIR(info.st_mode):
            raise ValueError("Root must be a real directory, not a link: " + root["path"])
    for index, left in enumerate(roots):
        for right in roots[index + 1:]:
            if left["scope"] != right["scope"] and (
                    Path(left["path"]).is_relative_to(right["path"]) or
                    Path(right["path"]).is_relative_to(left["path"])):
                raise ValueError("Overlapping roots must use the same scope")

    entries = {}
    skipped = {}
    def fail(error):
        raise error

    for root in roots:
        for folder, directories, filenames in os.walk(root["path"], followlinks=False, onerror=fail):
            for name in list(directories):
                path = normalized(Path(folder) / name)
                if is_link(os.lstat(path)):
                    directories.remove(name)
                    skipped[path] = {"path": path, "reason": "directory link or reparse point"}
            for name in filenames:
                path = normalized(Path(folder) / name)
                if path in entries:
                    entries[path]["memberships"].append(root["name"])
                    continue
                info = os.lstat(path)
                if is_link(info) or not stat.S_ISREG(info.st_mode):
                    skipped[path] = {"path": path, "reason": "link, reparse point or non-regular file"}
                    continue
                entries[path] = {"info": info, "memberships": [root["name"]]}

    roots_by_name = {root["name"]: root for root in roots}
    identities = {}
    files = []
    for path, entry in sorted(entries.items()):
        info = entry["info"]
        identity = file_identity(info, path)
        previous = identities.get(identity)
        if previous is None:
            previous = {"id": len(identities), "sha256": sha256(path, info), "signature": signature(info)}
            identities[identity] = previous
        elif previous["signature"] != signature(info):
            raise RuntimeError("Hardlinked file changed during inventory: " + path)
        root = max((roots_by_name[name] for name in entry["memberships"]), key=lambda item: len(item["path"]))
        relative = Path(path).relative_to(root["path"]).as_posix()
        row = {"root": root["name"], "scope": root["scope"], "path": relative,
               "bytes": info.st_size, "sha256": previous["sha256"], "storage_id": previous["id"],
               "component": component(root, relative)}
        if len(entry["memberships"]) > 1:
            row["memberships"] = sorted(entry["memberships"])
        files.append(row)

    # Do not silently publish hashes for files that changed after their read.
    for path, entry in entries.items():
        if signature(os.lstat(path)) != signature(entry["info"]):
            raise RuntimeError("File changed during inventory: " + path)

    duplicate_groups = defaultdict(list)
    for row in files:
        if row["bytes"]:
            duplicate_groups[(row["sha256"], row["bytes"])].append(row)
    duplicates = []
    for (digest, size), rows in duplicate_groups.items():
        if len(rows) > 1:
            unique_objects = len({row["storage_id"] for row in rows})
            duplicates.append({"sha256": digest, "bytes_each": size, "path_count": len(rows),
                               "unique_file_objects": unique_objects,
                               "duplicate_content_bytes_after_hardlinks": size * (unique_objects - 1),
                               "files": [{key: row[key] for key in ("root", "scope", "path", "storage_id")}
                                         for row in rows]})
    duplicates.sort(key=lambda group: (-group["duplicate_content_bytes_after_hardlinks"], group["sha256"]))
    return {"format": "sakuratts-windows-runtime-inventory-v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "measurement": {"read_only_inputs": True, "hash": "sha256", "gpu_executed": False,
                "logical_bytes": "Each distinct absolute file path counted once; overlapping roots do not add bytes.",
                "unique_file_bytes": "Each available (device, inode) identity counted once; not allocated disk space.",
                "duplicate_content_bytes_after_hardlinks": "Content redundancy among distinct file objects; not proven removable bytes.",
                "links": "Symbolic links and Windows reparse points are reported and not followed or counted.",
                "root_totals": "Inclusive; overlapping roots share files. Use scope_totals or totals for union size.",
                "consistency": "File size, identity and timestamps checked before/after hashing and at end; no filesystem snapshot lock."},
            "roots": [dict(root, totals=totals([row for row in files if root["name"] in
                                               row.get("memberships", [row["root"]])])) for root in roots],
            "totals": totals(files),
            "scope_totals": {scope: totals([row for row in files if row["scope"] == scope])
                             for scope in sorted({root["scope"] for root in roots})},
            "component_totals": {name: totals([row for row in files if row["component"] == name])
                                 for name in sorted({row["component"] for row in files})},
            "skipped_entries": list(skipped.values()), "duplicate_groups": duplicates, "files": files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, action="append", help="SCOPE:NAME=PATH; scopes: " + ", ".join(sorted(SCOPES)))
    parser.add_argument("--output", type=Path, required=True, help="New full inventory JSON, outside all measured roots")
    parser.add_argument("--summary-output", type=Path, help="Optional new compact JSON with the largest files and duplicate groups")
    args = parser.parse_args()
    roots = [parse_root(value) for value in args.root]
    outputs = [args.output] + ([args.summary_output] if args.summary_output else [])
    for output in outputs:
        if os.path.lexists(output):
            raise FileExistsError("Refusing to overwrite existing output: " + str(output))
        if any(output.resolve().is_relative_to(Path(root["path"]).resolve()) for root in roots):
            raise ValueError("Output must be outside measured roots")
    if len({normalized(path) for path in outputs}) != len(outputs):
        raise ValueError("Full and summary outputs must differ")
    report = inventory(roots)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    if args.summary_output:
        summary = {key: value for key, value in report.items() if key not in ("files", "duplicate_groups")}
        summary["full_inventory"] = {"path": str(args.output.resolve()), "bytes": args.output.stat().st_size,
                                     "sha256": sha256(str(args.output), args.output.stat())}
        summary["largest_files"] = sorted(report["files"], key=lambda row: -row["bytes"])[:30]
        summary["duplicate_group_count"] = len(report["duplicate_groups"])
        summary["largest_duplicate_groups"] = report["duplicate_groups"][:30]
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_output.open("x", encoding="utf-8") as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    print(json.dumps({"output": str(args.output), "scope_totals": report["scope_totals"],
                      "skipped_entries": len(report["skipped_entries"])}, indent=2))


if __name__ == "__main__":
    main()
