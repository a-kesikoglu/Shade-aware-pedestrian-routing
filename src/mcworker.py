"""Importable wrapper around the Monte Carlo worker of 12_height_checks.py (for multiprocessing on Windows)."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("hc12", Path(__file__).with_name("12_height_checks.py"))
_hc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hc)


def run(args):
    return _hc._run(args)
