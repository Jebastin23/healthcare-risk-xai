"""Dedicated explain CLI (thin wrapper over `main.py explain`)."""
from __future__ import annotations

import sys

from main import main

if __name__ == "__main__":
    raise SystemExit(main(["explain", *sys.argv[1:]]))
