"""Resolve the XRoboToolkit PC Service owned by this Mod."""

from __future__ import annotations

import os
from pathlib import Path
import re
import sys
import sysconfig


_SYSTEM_SERVICE_ROOT = Path("/opt/apps/roboticsservice")
_SERVICE_SDK_DIRECTORIES = {
    "linux-x86_64": "x64",
    "linux-aarch64": "arm64",
}


def runtime_platform_tag() -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", sysconfig.get_platform().lower())


def runtime_python_tag() -> str:
    cache_tag = sys.implementation.cache_tag
    if not cache_tag:
        cache_tag = f"python-{sys.version_info.major}{sys.version_info.minor}"
    safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", cache_tag.lower())
    return f"{runtime_platform_tag()}-{safe_tag}"


def mod_root() -> Path:
    explicit = os.environ.get("BXI_PICO_GMR_MOD_ROOT", "").strip()
    if explicit:
        return Path(explicit).expanduser().absolute()
    return Path(__file__).absolute().parent


def bundled_service_root() -> Path:
    return mod_root() / "runtime" / runtime_platform_tag() / "roboticsservice"


def resolve_service_root() -> Path:
    explicit = os.environ.get("PICO_GMR_XRT_SERVICE_DIR", "").strip()
    if explicit:
        return Path(explicit).expanduser().absolute()
    if _SYSTEM_SERVICE_ROOT.exists():
        return _SYSTEM_SERVICE_ROOT
    bundled = bundled_service_root()
    if bundled.exists():
        return bundled
    return _SYSTEM_SERVICE_ROOT


def service_library_paths(root: Path) -> tuple[Path, ...]:
    candidates: list[Path] = []
    sdk_name = _SERVICE_SDK_DIRECTORIES.get(runtime_platform_tag())
    if sdk_name is not None:
        candidates.append(root / "SDK" / sdk_name)
    sdk_root = root / "SDK"
    if sdk_root.is_dir():
        candidates.extend(
            child
            for child in sorted(sdk_root.iterdir())
            if child.is_dir() and (child / "libPXREARobotSDK.so").is_file()
        )
    candidates.extend((root, root / "lib"))
    return tuple(dict.fromkeys(path for path in candidates if path.is_dir()))


def configured_environment() -> tuple[Path, dict[str, str]]:
    root = resolve_service_root()
    executable = root / "RoboticsServiceProcess"
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise FileNotFoundError(
            f"RoboticsServiceProcess is not executable: {executable}"
        )
    env = os.environ.copy()
    libraries = [str(path) for path in service_library_paths(root)]
    inherited = env.get("LD_LIBRARY_PATH", "")
    if inherited:
        libraries.extend(item for item in inherited.split(os.pathsep) if item)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(dict.fromkeys(libraries))
    plugin = root / "plugins"
    qml = root / "qml"
    if plugin.is_dir():
        env["QT_PLUGIN_PATH"] = os.pathsep.join(
            item for item in (str(plugin) + "/", env.get("QT_PLUGIN_PATH", "")) if item
        )
    if qml.is_dir():
        env["QT_QML_PATH"] = os.pathsep.join(
            item for item in (str(qml) + "/", env.get("QT_QML_PATH", "")) if item
        )
    env["PICO_GMR_XRT_SERVICE_DIR"] = str(root)
    return root, env


__all__ = [
    "bundled_service_root",
    "configured_environment",
    "mod_root",
    "resolve_service_root",
    "runtime_platform_tag",
    "runtime_python_tag",
    "service_library_paths",
]
