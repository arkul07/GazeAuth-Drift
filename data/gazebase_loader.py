"""
GazebaseVR Data Loader

This module contains functions for loading and preprocessing GazebaseVR dataset
for gaze-based continuous authentication research.

Filename Format: S_[SubjectID]_S[Session]_[Round]_[Task].csv
Example: S_1002_S1_1_VRG.csv
- Subject ID: 1002
- Session: 1
- Round: 1
- Task: VRG (Video Random Gaze) or TEX (Text Reading)
"""

import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union


def load_single_file(file_path: str) -> pd.DataFrame:
    """
    Load a single GazebaseVR CSV file with metadata extraction.

    Args:
        file_path (str): Path to a single CSV file

    Returns:
        pd.DataFrame: Gaze data with metadata columns added
    """
    # Extract metadata from filename
    filename = os.path.basename(file_path)
    metadata = parse_filename(filename)

    # Load CSV
    df = pd.read_csv(file_path)

    # Add metadata columns
    df["user_id"] = metadata["subject_id"]
    df["session"] = metadata["session"]
    df["round"] = metadata["round"]
    df["task"] = metadata["task"]
    df["file_path"] = file_path

    # Rename columns for consistency
    df = df.rename(
        columns={
            "n": "timestamp_ms",
            "x": "gaze_x",
            "y": "gaze_y",
            "lx": "left_eye_x",
            "ly": "left_eye_y",
            "rx": "right_eye_x",
            "ry": "right_eye_y",
        }
    )

    return df


def parse_filename(filename: str) -> Dict[str, Union[int, str]]:
    """
    Parse GazebaseVR filename to extract metadata.

    Filename format: S_[SubjectID]_S[Session]_[Round]_[Task].csv

    Args:
        filename (str): Filename to parse

    Returns:
        Dict with subject_id, session, round, task
    """
    # Pattern: S_1002_S1_1_VRG.csv
    pattern = r"S_(\d+)_S(\d+)_(\d+)_([A-Z]+)\.csv"
    match = re.match(pattern, filename)

    if not match:
        raise ValueError(f"Filename '{filename}' does not match expected pattern")

    return {
        "subject_id": int(match.group(1)),
        "session": int(match.group(2)),
        "round": int(match.group(3)),
        "task": match.group(4),
    }


def load_gazebase_data(
    data_path: Union[str, List[str]],
    subjects: Optional[List[int]] = None,
    sessions: Optional[List[int]] = None,
    tasks: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load GazebaseVR data from file(s) or directory with filtering options.

    Args:
        data_path: Single file, list of files, or directory path
        subjects: List of subject IDs to load (None = all)
        sessions: List of session numbers to load (None = all)
        tasks: List of task types to load (None = all), e.g., ['VRG', 'TEX']

    Returns:
        pd.DataFrame: Combined gaze data from all loaded files
    """
    # Collect all CSV files to load
    csv_files = []

    if isinstance(data_path, list):
        csv_files = data_path
    elif os.path.isfile(data_path):
        csv_files = [data_path]
    elif os.path.isdir(data_path):
        csv_files = list(Path(data_path).glob("S_*.csv"))
    else:
        raise ValueError(f"Invalid data_path: {data_path}")

    # Filter files based on criteria
    if subjects or sessions or tasks:
        csv_files = filter_files(csv_files, subjects, sessions, tasks)

    if len(csv_files) == 0:
        raise ValueError("No files found matching the specified criteria")

    # Load and combine all files
    print(f"Loading {len(csv_files)} file(s)...")
    dfs = []
    for file_path in csv_files:
        try:
            df = load_single_file(str(file_path))
            dfs.append(df)
            print(f"  ✓ Loaded: {os.path.basename(file_path)} ({len(df)} rows)")
        except Exception as e:
            print(f"  ✗ Failed to load {file_path}: {e}")

    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal loaded: {len(combined_df)} rows from {len(dfs)} files")

    return combined_df


def filter_files(
    file_paths: List[Union[str, Path]],
    subjects: Optional[List[int]] = None,
    sessions: Optional[List[int]] = None,
    tasks: Optional[List[str]] = None,
) -> List[str]:
    """
    Filter file paths based on subject, session, and task criteria.

    Args:
        file_paths: List of file paths to filter
        subjects: Subject IDs to include
        sessions: Session numbers to include
        tasks: Task types to include

    Returns:
        List of filtered file paths
    """
    filtered = []

    for file_path in file_paths:
        filename = os.path.basename(str(file_path))
        try:
            metadata = parse_filename(filename)

            # Check filters
            if subjects and metadata["subject_id"] not in subjects:
                continue
            if sessions and metadata["session"] not in sessions:
                continue
            if tasks and metadata["task"] not in tasks:
                continue

            filtered.append(str(file_path))
        except ValueError:
            # Skip files that don't match pattern
            continue

    return filtered


def preprocess_gaze_data(
    df: pd.DataFrame,
    remove_missing: bool = True,
    normalize: bool = True,
    infer_fixations: bool = False,
    clip_outlier_speed_quantile: float = 0.999,
) -> pd.DataFrame:
    """
    Preprocess raw gaze data for feature extraction.

    Args:
        df: Raw gaze data
        remove_missing: Remove rows with missing gaze coordinates
        normalize: Normalize gaze coordinates to [0, 1] range

    Returns:
        Preprocessed gaze data
    """
    df_clean = df.copy()

    # Remove rows with missing gaze data
    if remove_missing:
        before_count = len(df_clean)
        df_clean = df_clean.dropna(subset=["gaze_x", "gaze_y"])
        removed = before_count - len(df_clean)
        if removed > 0:
            print(f"Removed {removed} rows with missing gaze data")

    # Normalize timestamps to seconds
    df_clean["timestamp_sec"] = df_clean["timestamp_ms"].astype(float) / 1000.0

    # Optional normalization of gaze_x/y into [0,1]
    if normalize:
        def _norm(group: pd.DataFrame) -> pd.DataFrame:
            gx = group["gaze_x"].astype(float)
            gy = group["gaze_y"].astype(float)
            # If already in [0,1], leave as-is
            if (gx.between(0, 1).mean() > 0.98) and (gy.between(0, 1).mean() > 0.98):
                group["gaze_x_norm"] = gx
                group["gaze_y_norm"] = gy
                return group
            # Min-max per user-session
            gx_min, gx_max = gx.min(), gx.max()
            gy_min, gy_max = gy.min(), gy.max()
            gx_rng = gx_max - gx_min if gx_max > gx_min else 1.0
            gy_rng = gy_max - gy_min if gy_max > gy_min else 1.0
            group["gaze_x_norm"] = ((gx - gx_min) / gx_rng).clip(0.0, 1.0)
            group["gaze_y_norm"] = ((gy - gy_min) / gy_rng).clip(0.0, 1.0)
            return group

        if "user_id" in df_clean.columns and "session" in df_clean.columns:
            df_clean = df_clean.groupby(["user_id", "session"], group_keys=False).apply(_norm)
        elif "user_id" in df_clean.columns:
            df_clean = df_clean.groupby(["user_id"], group_keys=False).apply(_norm)
        else:
            df_clean = _norm(df_clean)
        # Replace original columns for downstream compatibility
        df_clean["gaze_x"] = df_clean["gaze_x_norm"]
        df_clean["gaze_y"] = df_clean["gaze_y_norm"]

    # Compute per-user/session temporal deltas and kinematics
    def _kin(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("timestamp_sec", kind="mergesort").copy()
        dt = group["timestamp_sec"].diff().fillna(0.0)
        # Replace zeros with median dt to avoid inf speeds
        med_dt = dt[dt > 0].median() if (dt > 0).any() else 1.0 / 90.0
        dt = dt.where(dt > 0, other=med_dt)
        dx = group["gaze_x"].astype(float).diff().fillna(0.0)
        dy = group["gaze_y"].astype(float).diff().fillna(0.0)
        speed = np.sqrt(dx * dx + dy * dy) / dt
        group["dt"] = dt
        group["dx"] = dx
        group["dy"] = dy
        group["gaze_speed"] = speed
        # Clip extreme speeds for robustness
        if clip_outlier_speed_quantile:
            q = np.nanquantile(group["gaze_speed"], clip_outlier_speed_quantile)
            if np.isfinite(q) and q > 0:
                group["gaze_speed"] = np.clip(group["gaze_speed"], 0, q)
        return group

    if "user_id" in df_clean.columns and "session" in df_clean.columns:
        df_clean = df_clean.groupby(["user_id", "session"], group_keys=False).apply(_kin)
    elif "user_id" in df_clean.columns:
        df_clean = df_clean.groupby(["user_id"], group_keys=False).apply(_kin)
    else:
        df_clean = _kin(df_clean)

    # Optional fixation inference from speed
    if infer_fixations and "fixation_status" not in df_clean.columns:
        thr = df_clean.groupby("user_id")["gaze_speed"].transform(lambda s: s.quantile(0.6)) if "user_id" in df_clean.columns else df_clean["gaze_speed"].quantile(0.6)
        df_clean["fixation_status"] = (df_clean["gaze_speed"] < thr).astype(bool)

    return df_clean


def validate_data_quality(df: pd.DataFrame) -> Dict[str, any]:
    """
    Validate the quality of loaded gaze data.

    Args:
        df: Gaze data to validate

    Returns:
        Dictionary with validation metrics
    """
    validation = {}

    # Check for required columns
    required_cols = ["user_id", "timestamp_ms", "gaze_x", "gaze_y"]
    validation["has_required_columns"] = all(col in df.columns for col in required_cols)

    # Missing data percentage
    validation["missing_gaze_pct"] = (
        df[["gaze_x", "gaze_y"]].isnull().mean().mean() * 100
    )

    # Data range checks
    validation["gaze_x_range"] = (df["gaze_x"].min(), df["gaze_x"].max())
    validation["gaze_y_range"] = (df["gaze_y"].min(), df["gaze_y"].max())

    # Temporal continuity (check for gaps)
    validation["num_users"] = df["user_id"].nunique()
    validation["num_sessions"] = df.groupby("user_id")["session"].nunique().sum()
    validation["total_duration_sec"] = (
        df["timestamp_sec"].max() - df["timestamp_sec"].min()
    )

    return validation


def get_dataset_summary(df: pd.DataFrame) -> None:
    """
    Print summary statistics of the loaded dataset.

    Args:
        df: Loaded gaze data
    """
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total samples: {len(df):,}")
    print(f"Number of users: {df['user_id'].nunique()}")
    print(f"Number of sessions: {df.groupby('user_id')['session'].nunique().sum()}")
    print(f"Tasks: {df['task'].unique().tolist()}")
    print(f"\nSamples per user:")
    print(df["user_id"].value_counts().sort_index())
    print(f"\nMissing data: {df[['gaze_x', 'gaze_y']].isnull().sum().sum()} samples")
    print("=" * 60 + "\n")
