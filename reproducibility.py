"""Reproducibility helpers for SAE multilingual steering experiments.

This module centralizes:
- RNG seeding (python/random, numpy, torch)
- Optional deterministic settings (best-effort; warn-only)
- Run manifests for publication-grade provenance
"""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def seed_everything(seed: int, deterministic: Optional[bool] = None) -> None:
    """Seed python, numpy, and torch (if available).

    Set DETERMINISTIC=1 to enable best-effort deterministic behavior.
    """
    if deterministic is None:
        deterministic = str(os.environ.get("DETERMINISTIC", "0")).lower() in ("1", "true", "yes")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            # Best-effort: keep warn_only=True to avoid hard crashes on ops
            # that are nondeterministic on certain CUDA versions.
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
            try:
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
            except Exception:
                pass
    except Exception:
        # torch not installed or unavailable in this environment
        return


def _git_commit() -> Optional[str]:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        if r.returncode == 0:
            return r.stdout.strip() or None
    except Exception:
        pass
    return None


def collect_run_metadata(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Collect minimal run metadata for reproducibility."""
    meta: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "env": {
            k: os.environ.get(k)
            for k in [
                "USE_9B",
                "SAE_CACHE_SIZE",
                "STEERING_SCHEDULE",
                "STEERING_DECAY",
                "STRICT_DATA",
                "STRICT_PREREG",
                "STRICT_JUDGE",
                "DETERMINISTIC",
                "REPETITION_UNIT",
            ]
        },
    }

    try:
        import config

        meta["config"] = {
            "MODEL_ID": getattr(config, "MODEL_ID", None),
            "SAE_RELEASE": getattr(config, "SAE_RELEASE", None),
            "SAE_WIDTH": getattr(config, "SAE_WIDTH", None),
            "TARGET_LAYERS": getattr(config, "TARGET_LAYERS", None),
            "SEED": getattr(config, "SEED", None),
            "SEMANTIC_SIM_THRESHOLD": getattr(config, "SEMANTIC_SIM_THRESHOLD", None),
            "DEVANAGARI_THRESHOLD": getattr(config, "DEVANAGARI_THRESHOLD", None),
            "MIN_LANG_SAMPLES": getattr(config, "MIN_LANG_SAMPLES", None),
            "MIN_STEERING_PROMPTS": getattr(config, "MIN_STEERING_PROMPTS", None),
            "MIN_JUDGE_CALIB_PER_CLASS": getattr(config, "MIN_JUDGE_CALIB_PER_CLASS", None),
        }
    except Exception:
        pass

    if extra:
        meta["extra"] = extra

    return meta


def write_run_manifest(out_dir: str | Path = "results", extra: Optional[Dict[str, Any]] = None) -> Path:
    """Write a single manifest.json in the results directory."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    manifest_path = out_path / "manifest.json"
    meta = collect_run_metadata(extra=extra)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=_json_default)
    return manifest_path


def _json_default(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

