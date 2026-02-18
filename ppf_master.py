#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

def run_step(cmd):
    print(f"\n>>> RUNNING: {cmd}\n", flush=True)
    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    for line in process.stdout:
        print(line, end="", flush=True)
    process.wait()
    if process.returncode != 0:
        sys.exit(process.returncode)

def main():
    root = Path(".").resolve()
    # Wipe the mapping and processed files to force a fresh join
    run_step(f"rm -rf {root}/data/processed/*")
    run_step(f"rm -f {root}/data/reference/asset_to_sector_mapping.csv")
    run_step(f"rm -f {root}/outputs/ppf_asset_universe.csv")
    run_step(f"rm -f {root}/outputs/ppf_transactions_unified.csv")

    run_step(f"python3 {root}/unify_ppf.py --project-root {root}")
    run_step(f"python3 {root}/ppf_pipeline.py --project-root {root}")

if __name__ == "__main__":
    main()
