from __future__ import annotations

import os
import shlex
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    classifier_command: tuple[str, ...]
    work_dir: str
    keep_artifacts: bool

    @classmethod
    def from_env(cls) -> "Settings":
        command = tuple(shlex.split(os.getenv("BAT_CLASSIFIER_COMMAND", "")))
        work_dir = os.getenv("BAT_WORK_DIR", "/tmp/bat-classifier")
        keep_artifacts = os.getenv("BAT_KEEP_ARTIFACTS", "").lower() in {"1", "true", "yes", "on"}
        return cls(
            classifier_command=command,
            work_dir=work_dir,
            keep_artifacts=keep_artifacts,
        )
