"""Master experiment pipeline entry point.

Forwards to scripts/run_experiments.py for sequential execution of all 5 benchmark models.

Usage:
    python experiments/run_all.py
"""
import os
import sys
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    cmd = [sys.executable, os.path.join(PROJECT_ROOT, "scripts", "run_experiments.py")]
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
