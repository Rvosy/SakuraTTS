#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
unset PYTHONHOME PYTHONPATH DYLD_LIBRARY_PATH DYLD_FALLBACK_LIBRARY_PATH DYLD_INSERT_LIBRARIES
exec "$ROOT/runtime/main/bin/python3" -I -B -u "$ROOT/launcher.py" "$@"
