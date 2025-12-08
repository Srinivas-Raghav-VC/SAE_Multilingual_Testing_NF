#!/usr/bin/env python3
"""
Smart experiment runner for SAE Multilingual Steering.

Goals:
- Prioritise "lighter" experiments first.
- For each experiment, try to use GPU if there is enough free memory.
- If the GPU looks busy and the experiment is light, fall back to CPU
  instead of doing nothing.
- For heavier experiments, optionally wait until the GPU is mostly free
  before running (so you can leave this script overnight).

Usage (inside the venv):

    python smart_run.py

All logs are written under ./logs as exp<N>_*.log.
"""

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Experiments in the order we want to run them.
# Mapping: flag -> description
LIGHT_EXPERIMENTS = [
    ("--exp1", "exp1_feature_discovery"),
    ("--exp2", "exp2_steering_comparison"),
    ("--exp3", "exp3_hindi_urdu_overlap"),
    ("--exp4", "exp4_spillover"),
    ("--exp5", "exp5_hierarchical"),
    ("--exp6", "exp6_script_semantics_controls"),
    ("--exp7", "exp7_causal_feature_probing"),
    ("--exp11", "exp11_judge_calibration"),
]

# "Heavier" experiments – we prefer to wait for a mostly free GPU.
HEAVY_EXPERIMENTS = [
    ("--exp8", "exp8_scaling_9b_low_resource"),
    ("--exp9", "exp9_layer_sweep_steering"),
    ("--exp10", "exp10_attribution_occlusion"),
    ("--exp14", "exp14_language_agnostic_space"),
    ("--exp15", "exp15_directional_symmetry"),
    ("--exp16", "exp16_code_mix_robustness"),
    ("--exp12", "exp12_qa_degradation"),
    ("--exp13", "exp13_script_semantic_ablation"),
]

# How much free GPU memory (in MiB) we want before we consider the GPU "free
# enough" for an experiment.
LIGHT_GPU_THRESHOLD_MB = 8000   # if less than this, run light exps on CPU
HEAVY_GPU_THRESHOLD_MB = 20000  # heavy exps wait until at least this much free

# How long to sleep (seconds) between GPU checks for heavy experiments.
POLL_INTERVAL_SEC = 120


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gpu_free_memory_mb() -> Tuple[bool, int]:
    """Return (ok, free_mem_mb) using nvidia-smi.

    If nvidia-smi is not available or something goes wrong, returns (False, 0).
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return False, 0
        # In a MIG setup this will usually be a single line; we use the first.
        line = result.stdout.strip().splitlines()[0]
        free_mb = int(line.strip())
        return True, free_mb
    except Exception:
        return False, 0


def ensure_logs_dir() -> Path:
    """Create ./logs directory if needed and return its Path."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    return logs_dir


def run_experiment(flag: str, name: str, use_gpu: bool) -> bool:
    """Run a single experiment.

    Args:
        flag: e.g. '--exp1'
        name: short name for logging
        use_gpu: if False, we set CUDA_VISIBLE_DEVICES="" to force CPU

    Returns:
        True if subprocess exits with code 0, False otherwise.
    """
    logs_dir = ensure_logs_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"{name}_{ts}.log"

    env = os.environ.copy()
    if not use_gpu:
        # Force CPU for this subprocess
        env["CUDA_VISIBLE_DEVICES"] = ""

    cmd = [sys.executable, "-u", "run.py", flag]
    print(f"\n=== Running {name} ({'GPU' if use_gpu else 'CPU'}) ===")
    print(f"Command: {' '.join(cmd)}")
    print(f"Logging to: {log_path}")

    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)

    if proc.returncode == 0:
        print(f"✓ {name} finished successfully")
        return True
    else:
        print(f"✗ {name} failed (see {log_path})")
        return False


def run_light_experiments():
    """Run light experiments, using GPU if reasonably free, else CPU."""
    for flag, name in LIGHT_EXPERIMENTS:
        ok, free_mb = gpu_free_memory_mb()
        if ok:
            print(f"\n[GPU] Free memory: {free_mb} MiB")
        else:
            print("\n[GPU] nvidia-smi not available or failed; treating as no GPU.")

        use_gpu = ok and free_mb >= LIGHT_GPU_THRESHOLD_MB

        success = run_experiment(flag, name, use_gpu=use_gpu)
        if not success:
            # We log the failure but continue so other experiments can run.
            print(f"[warning] Continuing despite failure in {name}")


def wait_for_gpu(threshold_mb: int):
    """Block until GPU free memory is at least threshold_mb."""
    while True:
        ok, free_mb = gpu_free_memory_mb()
        if not ok:
            print("[GPU] nvidia-smi not available; cannot wait for GPU. Running on CPU.")
            return False
        print(f"[GPU] Free memory: {free_mb} MiB (need ≥ {threshold_mb} MiB)")
        if free_mb >= threshold_mb:
            return True
        print(f"[GPU] Not enough free memory yet; sleeping {POLL_INTERVAL_SEC}s...")
        time.sleep(POLL_INTERVAL_SEC)


def run_heavy_experiments():
    """Run heavy experiments, preferring GPU and waiting if necessary."""
    for flag, name in HEAVY_EXPERIMENTS:
        print(f"\n=== Preparing to run heavy experiment {name} ===")
        ok, free_mb = gpu_free_memory_mb()
        if ok:
            print(f"[GPU] Current free memory: {free_mb} MiB")
        else:
            print("[GPU] nvidia-smi not available; will run on CPU.")

        if ok:
            # Wait until the GPU has enough free memory
            gpu_available = wait_for_gpu(HEAVY_GPU_THRESHOLD_MB)
            if gpu_available:
                success = run_experiment(flag, name, use_gpu=True)
            else:
                # nvidia-smi disappeared while waiting; run on CPU.
                success = run_experiment(flag, name, use_gpu=False)
        else:
            # No visible GPU; run on CPU.
            success = run_experiment(flag, name, use_gpu=False)

        if not success:
            print(f"[warning] Heavy experiment {name} failed (see logs). Continuing.")


def main():
    print("=" * 60)
    print("SMART EXPERIMENT RUNNER")
    print("=" * 60)
    print("This script will:")
    print("  1. Run light experiments first (GPU if free, else CPU).")
    print("  2. Then run heavy experiments, waiting for a mostly free GPU.")
    print("Logs are written to ./logs/*.log\n")

    run_light_experiments()
    run_heavy_experiments()

    print("\nAll scheduled experiments launched. Check ./logs for details.")


if __name__ == "__main__":
    main()
