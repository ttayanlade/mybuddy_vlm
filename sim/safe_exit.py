"""Shut Isaac Sim down without hanging the Slurm job.

Headless Isaac Sim 5.1 in Apptainer can hang forever inside SimulationApp.close()
(seen on Nova: work finished in 1 s, job then sat until its time limit and the
buffered stdout was lost). Flush everything, give close() a short grace period on
a daemon thread, then hard-exit with the right status code.
"""

from __future__ import annotations

import os
import sys
import threading


def close_and_exit(app, code: int = 0, grace_s: float = 20.0):
    sys.stdout.flush()
    sys.stderr.flush()
    t = threading.Thread(target=app.close, daemon=True)
    t.start()
    t.join(grace_s)
    if t.is_alive():
        print(f"[safe_exit] SimulationApp.close() still running after {grace_s:.0f}s; forcing exit", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
