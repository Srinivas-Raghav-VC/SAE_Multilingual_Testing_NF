#!/usr/bin/env python3
"""Download fastText language ID models (lid.176.{ftz,bin}).

Why:
  - Spillover controls need reliable language ID for Latin-script outputs.
  - fastText LID models are a strong, lightweight baseline and work well on
    short text compared to regex heuristics.

Models (official):
  - https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz
  - https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

We default to the .ftz model because it is much smaller; use --bin for the full
.bin model.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.request import urlretrieve


URL_FTZ = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
URL_BIN = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    if dest.exists():
        print(f"[download] Already exists: {dest}")
        return
    print(f"[download] Fetching {url}")
    print(f"[download] Saving to {dest}")
    try:
        urlretrieve(url, tmp)
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dest-dir",
        default="models",
        help="Directory to place downloaded models (default: models/).",
    )
    ap.add_argument(
        "--bin",
        action="store_true",
        help="Download lid.176.bin (larger, full model).",
    )
    ap.add_argument(
        "--ftz",
        action="store_true",
        help="Download lid.176.ftz (small, quantized model).",
    )
    args = ap.parse_args()

    dest_dir = Path(args.dest_dir)
    want_bin = bool(args.bin)
    want_ftz = bool(args.ftz)

    if not want_bin and not want_ftz:
        want_ftz = True

    if want_ftz:
        _download(URL_FTZ, dest_dir / "lid.176.ftz")
    if want_bin:
        _download(URL_BIN, dest_dir / "lid.176.bin")

    print("\nNext steps (recommended):")
    print("  - Install LID backend: pip install fast-langdetect")
    print("  - Export:")
    print("      export LID_BACKEND=fasttext")
    print(f"      export FASTTEXT_LID_MODEL_PATH={dest_dir / 'lid.176.ftz'}")


if __name__ == "__main__":
    main()

