import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "runKANAD.py",
    "runCATCH.py",
    "runPGRF.py",
    "runCAD.py",
    "runSARAD.py",
]

def main():
    here = Path(__file__).parent
    failed = []

    for script in SCRIPTS:
        print(f"\n{'='*60}")
        print(f"Running {script}")
        print('='*60)
        result = subprocess.run(
            [sys.executable, str(here / script)],
            check=False,
        )
        if result.returncode != 0:
            print(f"[FAILED] {script} exited with code {result.returncode}")
            failed.append(script)

    print(f"\n{'='*60}")
    if failed:
        print(f"Completed with failures: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("All experiments completed successfully.")

if __name__ == "__main__":
    main()
