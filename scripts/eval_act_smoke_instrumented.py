#!/usr/bin/env python3
"""Compatibility entry point for older local experiment scripts.

New commands should invoke :mod:`eval_policy_instrumented` directly.
"""

from pathlib import Path
import runpy


runpy.run_path(Path(__file__).with_name("eval_policy_instrumented.py"), run_name="__main__")
