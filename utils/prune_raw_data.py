"""Prune raw GazebaseVR data to a lean subset.

Retains only specified subjects and required task rounds (2_PUR, 4_TEX, 5_RAN) for sessions S1 and S2.
Moves all other CSV files from data/raw/ to data/raw_excluded/ (preserving filenames).

Run:
    python -m utils.prune_raw_data --subjects 1002 1003 1004 1005 1007 1008 1010 1011 1013 1015

Adjust subject list as needed.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil

REQUIRED_TASKS = {"2_PUR", "4_TEX", "5_RAN"}
SESSIONS = {"S1", "S2"}


def should_keep(filename: str, subjects: set[str]) -> bool:
    parts = filename.split("_")
    if len(parts) < 5:  # S, <id>, S<session>, <round>, <task>.csv
        return False
    subj = parts[1]
    sess = parts[2]
    rnd = parts[3]
    task = parts[4].split(".")[0]
    key = f"{rnd}_{task}"
    return subj in subjects and sess in SESSIONS and key in REQUIRED_TASKS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", required=True, help="Subject IDs to keep (e.g. 1002 1003 ...)")
    parser.add_argument("--raw_dir", default="data/raw", help="Directory containing raw CSV files")
    parser.add_argument("--excluded_dir", default="data/raw_excluded", help="Directory to move excluded files")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    excluded_dir = Path(args.excluded_dir)
    excluded_dir.mkdir(parents=True, exist_ok=True)

    subjects = set(args.subjects)
    kept, moved = 0, 0

    for f in raw_dir.glob("S_*.csv"):
        if should_keep(f.name, subjects):
            kept += 1
        else:
            target = excluded_dir / f.name
            shutil.move(str(f), target)
            moved += 1

    print(f"Kept files: {kept}")
    print(f"Moved files: {moved} -> {excluded_dir}")


if __name__ == "__main__":
    main()
