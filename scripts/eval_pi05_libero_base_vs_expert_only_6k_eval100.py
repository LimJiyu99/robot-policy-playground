#!/usr/bin/env python3
"""Public entry point for the resumable raw-base versus 6K paired evaluation."""

from pathlib import Path
import runpy


runpy.run_path(
    Path(__file__).with_name("eval_pi05_libero_base_vs_expert6k_eval100_resume.py"),
    run_name="__main__",
)
