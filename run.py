#!/usr/bin/env python
"""Main runner for SAE multilingual steering experiments.

Usage:
    python run.py exp1  # Feature discovery (H1, H3)
    python run.py exp2  # Steering comparison (H2)
    python run.py exp3  # Hindi-Urdu overlap (H4)
    python run.py all   # Run all experiments
"""

import sys
from pathlib import Path


def run_exp1():
    print("=" * 60)
    print("EXPERIMENT 1: Feature Discovery (H1, H3)")
    print("=" * 60)
    from experiments.exp1_feature_discovery import main
    main()


def run_exp2():
    print("=" * 60)
    print("EXPERIMENT 2: Steering Comparison (H2)")
    print("=" * 60)
    from experiments.exp2_steering import main
    main()


def run_exp3():
    print("=" * 60)
    print("EXPERIMENT 3: Hindi-Urdu Overlap (H4)")
    print("=" * 60)
    from experiments.exp3_hindi_urdu import main
    main()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    # Create results directory
    Path("results").mkdir(exist_ok=True)
    
    if cmd == "exp1":
        run_exp1()
    elif cmd == "exp2":
        run_exp2()
    elif cmd == "exp3":
        run_exp3()
    elif cmd == "all":
        run_exp1()
        print("\n" * 2)
        run_exp2()
        print("\n" * 2)
        run_exp3()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
