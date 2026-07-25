"""Dedicated train CLI (thin wrapper over `main.py train`)."""
from __future__ import annotations

import sys

from main import main

if __name__ == "__main__":
    raise SystemExit(main(["train", *sys.argv[1:]]))
