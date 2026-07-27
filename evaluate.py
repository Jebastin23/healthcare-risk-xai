"""Dedicated evaluate CLI (thin wrapper over `main.py evaluate`)."""
from __future__ import annotations

import sys

from main import main

if __name__ == "__main__":
    raise SystemExit(main(["evaluate", *sys.argv[1:]]))
