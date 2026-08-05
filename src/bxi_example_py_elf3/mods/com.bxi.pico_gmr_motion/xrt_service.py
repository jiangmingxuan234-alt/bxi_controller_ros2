"""Deterministic owner for one XRoboToolkit RoboticsServiceProcess."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import time

from service_runtime import configured_environment


class ManagedRoboticsService:
    def __init__(self) -> None:
        self.root: Path | None = None
        self.process: subprocess.Popen | None = None

    def start(self) -> None:
        if self.process is not None:
            raise RuntimeError("RoboticsServiceProcess is already started")
        root, env = configured_environment()
        executable = root / "RoboticsServiceProcess"
        print(f"Starting XRoboToolkit PC Service: {executable}", flush=True)
        process = subprocess.Popen(
            [str(executable)],
            cwd=str(root),
            env=env,
        )
        self.root = root
        self.process = process
        time.sleep(0.2)
        return_code = process.poll()
        if return_code is not None:
            self.process = None
            raise RuntimeError(
                "RoboticsServiceProcess exited during startup with code "
                f"{return_code}; another service may already own port 60061"
            )
        print(f"XRoboToolkit PC Service ready: pid={process.pid}", flush=True)

    def close(self) -> None:
        process = self.process
        self.process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3.0)


__all__ = ["ManagedRoboticsService"]
