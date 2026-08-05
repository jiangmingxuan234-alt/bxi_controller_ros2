#!/usr/bin/env python3
"""Prepare the packaged XR runtime, then execute the PICO-GMR worker."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

from service_runtime import (
    configured_environment,
    mod_root,
    runtime_python_tag,
)


CONFIG_ERROR = getattr(os, "EX_CONFIG", 78)


def main() -> int:
    root = mod_root()
    worker = root / "pico_gmr_process.py"
    if not worker.is_file():
        print(f"PICO GMR worker is missing: {worker}", file=sys.stderr)
        return CONFIG_ERROR
    try:
        _, env = configured_environment()
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"PICO GMR service configuration error: {exc}", file=sys.stderr)
        return CONFIG_ERROR

    if importlib.util.find_spec("xrobotoolkit_sdk") is None:
        vendor = root / "vendor" / "python" / runtime_python_tag()
        if not vendor.is_dir():
            print(
                "xrobotoolkit_sdk is unavailable and no matching bundled binding exists: "
                f"{vendor}",
                file=sys.stderr,
            )
            return CONFIG_ERROR
        inherited = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(
            item for item in (str(vendor), inherited) if item
        )

    env["BXI_PICO_GMR_MOD_ROOT"] = str(root)
    command = (sys.executable, "-B", str(worker), *sys.argv[1:])
    os.execve(sys.executable, command, env)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
