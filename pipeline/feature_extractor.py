"""
Gaze Feature Extractor

This module contains core functions for calculating behavioral gaze features
from raw gaze data for continuous authentication.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from math import atan2


def extract_gaze_features(
    df: pd.DataFrame,
    window_size_sec: float = 5.0,
    overlap_sec: float = 1.0,
    grid_bins: int = 8,
    min_samples: int = 10,
) -> pd.DataFrame:
    """
    Extract behavioral gaze features from (preprocessed) gaze data using sliding windows.

    Expected columns (preprocessed preferred):
      - user_id, timestamp, x_gaze, y_gaze, fixation_status (bool)
      - Optional (computed if missing): dt, dx, dy, gaze_speed

    Args:
        df: Gaze dataframe.
        window_size_sec: Window duration in seconds.
        overlap_sec: Overlap between consecutive windows in seconds.
        grid_bins: Number of bins per axis for scanpath entropy (grid_bins x grid_bins).
        min_samples: Minimum samples required in a window to compute features.

    Returns:
        DataFrame with one row per user/window and aggregated features.
    """
    # Normalize common column name variants from loader/preprocess
    df = df.copy()
    if "timestamp" not in df.columns:
        if "timestamp_sec" in df.columns:
            df["timestamp"] = df["timestamp_sec"].astype(float)
        elif "timestamp_ms" in df.columns:
            df["timestamp"] = df["timestamp_ms"].astype(float) / 1000.0
    if "x_gaze" not in df.columns and "gaze_x" in df.columns:
        df = df.rename(columns={"gaze_x": "x_gaze"})
    if "y_gaze" not in df.columns and "gaze_y" in df.columns:
        df = df.rename(columns={"gaze_y": "y_gaze"})

    required = {"user_id", "timestamp", "x_gaze", "y_gaze"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for feature extraction: {sorted(missing)}")

    # Ensure sorted per user/time for stable windowing
    df.sort_values(["user_id", "timestamp"], inplace=True, kind="mergesort")

    # Ensure auxiliaries exist
    if "dt" not in df.columns or (df["dt"].isna().all()):
        # approximate dt per user
        df["dt"] = df.groupby("user_id")["timestamp"].diff().fillna(0.0)
        # replace 0 with median
        med_dt = df.groupby("user_id")["dt"].transform(lambda s: s[s > 0].median() if (s > 0).any() else 1.0 / 90.0)
        df["dt"] = df["dt"].where(df["dt"] > 0, med_dt)
    if "dx" not in df.columns or "dy" not in df.columns:
        df["dx"] = df.groupby("user_id")["x_gaze"].diff().fillna(0.0)
        df["dy"] = df.groupby("user_id")["y_gaze"].diff().fillna(0.0)
    if "gaze_speed" not in df.columns or (df["gaze_speed"].isna().all()):
        # normalized distance per second
        step_len = np.sqrt(df["dx"] * df["dx"] + df["dy"] * df["dy"])
        gaze_speed = step_len / df["dt"].replace(0, np.nan)
        gaze_speed = gaze_speed.replace([np.inf, -np.inf], np.nan)
        df["gaze_speed"] = gaze_speed.fillna(gaze_speed.median())
    if "fixation_status" not in df.columns:
        # Infer fixation via per-user adaptive threshold on speed
        thr = df.groupby("user_id")["gaze_speed"].transform(lambda s: s.quantile(0.6))
        df["fixation_status"] = (df["gaze_speed"] < thr).astype(bool)

    step = max(window_size_sec - overlap_sec, 1e-6)
    features: List[Dict[str, float]] = []

    for user_id, g in df.groupby("user_id", sort=False):
        t0 = float(g["timestamp"].min())
        t1 = float(g["timestamp"].max())
        w_start = t0
        while w_start < t1:
            w_end = w_start + window_size_sec
            win = g[(g["timestamp"] >= w_start) & (g["timestamp"] < w_end)]
            if len(win) >= min_samples:
                fx = calculate_fixation_features(win)
                sc = calculate_saccade_features(win)
                pu = calculate_pursuit_features(win)
                vl = calculate_velocity_features(win)
                sp = calculate_scanpath_features(win, grid_bins=grid_bins)
                rec = {
                    "user_id": int(user_id),
                    "t_start": w_start,
                    "t_end": w_end,
                    "sample_count": int(len(win)),
                    **fx,
                    **sc,
                    **pu,
                    **vl,
                    **sp,
                }
                features.append(rec)
            w_start += step

    return pd.DataFrame(features)


def calculate_fixation_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    Fixation-based features within a window.

    Returns fixation ratio, count, mean/std durations.
    """
    out: Dict[str, float] = {}
    if df.empty:
        return {
            "fix_ratio": np.nan,
            "fix_count": 0.0,
            "fix_dur_mean": np.nan,
            "fix_dur_std": np.nan,
        }

    dt = df["dt"].to_numpy(dtype=float)
    fix = df["fixation_status"].astype(bool).to_numpy()
    total_time = float(np.nansum(dt)) if np.isfinite(dt).any() else float(len(df))
    fix_time = float(np.nansum(dt[fix]))

    # Identify contiguous fixation runs
    # Label segments when value changes
    seg_ids = np.cumsum(np.r_[True, fix[1:] != fix[:-1]])
    # For each segment where fix==True, sum duration
    fix_durations = []
    for seg in np.unique(seg_ids):
        mask = seg_ids == seg
        if fix[mask][0]:
            fix_durations.append(float(np.nansum(dt[mask])))

    out["fix_ratio"] = (fix_time / total_time) if total_time > 0 else np.nan
    out["fix_count"] = float(len(fix_durations))
    out["fix_dur_mean"] = float(np.nanmean(fix_durations)) if fix_durations else np.nan
    out["fix_dur_std"] = float(np.nanstd(fix_durations)) if fix_durations else np.nan
    return out


def calculate_saccade_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    Saccade-based features within a window.

    Approximates saccade episodes as contiguous non-fixation segments and
    computes per-episode path length (amplitude) and mean speed.
    """
    out: Dict[str, float] = {}
    if df.empty:
        return {
            "sac_count": 0.0,
            "sac_amp_mean": np.nan,
            "sac_amp_std": np.nan,
            "sac_speed_mean": np.nan,
        }

    fix = df["fixation_status"].astype(bool).to_numpy()
    dt = df["dt"].to_numpy(dtype=float)
    dx = df["dx"].to_numpy(dtype=float)
    dy = df["dy"].to_numpy(dtype=float)
    speed = df["gaze_speed"].to_numpy(dtype=float)
    step_len = np.sqrt(dx * dx + dy * dy)

    seg_ids = np.cumsum(np.r_[True, fix[1:] != fix[:-1]])
    sac_amplitudes: List[float] = []
    sac_mean_speeds: List[float] = []
    sac_peak_speeds: List[float] = []
    for seg in np.unique(seg_ids):
        mask = seg_ids == seg
        if not fix[mask][0]:  # saccade episode
            amp = float(np.nansum(step_len[mask]))
            # mean speed weighted by time within the segment
            dur = float(np.nansum(dt[mask]))
            mspd = float(np.nansum(speed[mask] * dt[mask]) / dur) if dur > 0 else np.nan
            sac_amplitudes.append(amp)
            sac_mean_speeds.append(mspd)
        sac_peak_speeds.append(float(np.nanmax(speed[mask])) if np.any(~np.isnan(speed[mask])) else np.nan)

    out["sac_count"] = float(len(sac_amplitudes))
    out["sac_amp_mean"] = float(np.nanmean(sac_amplitudes)) if sac_amplitudes else np.nan
    out["sac_amp_std"] = float(np.nanstd(sac_amplitudes)) if sac_amplitudes else np.nan
    out["sac_speed_mean"] = float(np.nanmean(sac_mean_speeds)) if sac_mean_speeds else np.nan
    out["sac_peak_speed_mean"] = float(np.nanmean(sac_peak_speeds)) if sac_peak_speeds else np.nan
    return out


def calculate_scanpath_features(df: pd.DataFrame, grid_bins: int = 8) -> Dict[str, float]:
    """
    Scanpath features: path length, spatial dispersion, and grid entropy.
    """
    out: Dict[str, float] = {}
    if df.empty:
        return {
            "path_len": 0.0,
            "disp_x_std": np.nan,
            "disp_y_std": np.nan,
            "grid_entropy": np.nan,
        }

    dx = df["dx"].to_numpy(dtype=float)
    dy = df["dy"].to_numpy(dtype=float)
    x = df["x_gaze"].to_numpy(dtype=float)
    y = df["y_gaze"].to_numpy(dtype=float)
    step_len = np.sqrt(dx * dx + dy * dy)
    path_len = float(np.nansum(step_len))

    # Dispersion via std dev on each axis
    disp_x = float(np.nanstd(x))
    disp_y = float(np.nanstd(y))

    # Grid entropy on discretized 2D space
    bins = max(int(grid_bins), 2)
    # Clip to [0,1] just in case
    x_clipped = np.clip(x, 0.0, 1.0)
    y_clipped = np.clip(y, 0.0, 1.0)
    H, _, _ = np.histogram2d(x_clipped, y_clipped, bins=bins, range=[[0, 1], [0, 1]])
    p = H.ravel().astype(float)
    p_sum = p.sum()
    if p_sum > 0:
        p /= p_sum
        p = p[p > 0]
        grid_entropy = float(-np.sum(p * np.log2(p)))
    else:
        grid_entropy = np.nan

    out["path_len"] = path_len
    out["disp_x_std"] = disp_x
    out["disp_y_std"] = disp_y
    out["grid_entropy"] = grid_entropy
    return out


def calculate_velocity_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    Velocity-based features: summary stats of instantaneous gaze speed.
    """
    out: Dict[str, float] = {}
    if df.empty or "gaze_speed" not in df.columns:
        return {
            "vel_mean": np.nan,
            "vel_std": np.nan,
            "vel_p95": np.nan,
            "vel_max": np.nan,
        }

    v = df["gaze_speed"].to_numpy(dtype=float)
    out["vel_mean"] = float(np.nanmean(v)) if v.size else np.nan
    out["vel_std"] = float(np.nanstd(v)) if v.size else np.nan
    out["vel_p95"] = float(np.nanpercentile(v, 95)) if v.size else np.nan
    out["vel_max"] = float(np.nanmax(v)) if v.size else np.nan
    return out


def calculate_pursuit_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    Smooth pursuit features based on speed and directional consistency.

    Heuristic classification:
      - fixation: low speed
      - saccade: high speed
      - pursuit: mid speed AND high directional consistency (low circular variance)

    Returns pursuit ratio, count, duration stats, mean speed, and direction consistency.
    """
    out: Dict[str, float] = {}
    if df.empty:
        return {
            "pursuit_ratio": np.nan,
            "pursuit_count": 0.0,
            "pursuit_dur_mean": np.nan,
            "pursuit_dur_std": np.nan,
            "pursuit_speed_mean": np.nan,
            "pursuit_dir_consistency": np.nan,
        }

    v = df["gaze_speed"].to_numpy(dtype=float)
    dt = df["dt"].to_numpy(dtype=float)
    dx = df["dx"].to_numpy(dtype=float)
    dy = df["dy"].to_numpy(dtype=float)
    # Direction angles
    theta = np.arctan2(dy, dx)

    # Per-window adaptive thresholds
    v_fix_thr = np.nanpercentile(v, 60)
    v_sac_thr = np.nanpercentile(v, 90)

    # Candidate pursuit where speed in [v_fix_thr, v_sac_thr)
    is_pursuit_speed = (v >= v_fix_thr) & (v < v_sac_thr)

    # Directional consistency via circular variance over a small rolling window
    # Compute circular variance per sample using a short window (e.g., 9 samples)
    k = 9
    if len(theta) >= k:
        # rolling mean resultant length R
        # Use simple convolution on cos/sin
        kernel = np.ones(k)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        cs = np.convolve(cos_t, kernel, mode="same")
        sn = np.convolve(sin_t, kernel, mode="same")
        R = np.sqrt(cs * cs + sn * sn) / k
        circ_var = 1.0 - R  # 0=perfectly consistent, 1=random
        # Threshold for consistency
        var_thr = np.nanpercentile(circ_var, 50)  # prefer more consistent than median
        is_dir_consistent = circ_var <= var_thr
    else:
        is_dir_consistent = np.ones_like(theta, dtype=bool)

    is_pursuit = is_pursuit_speed & is_dir_consistent

    total_time = float(np.nansum(dt)) if np.isfinite(dt).any() else float(len(df))
    pursuit_time = float(np.nansum(dt[is_pursuit]))

    # Segment pursuits
    seg_ids = np.cumsum(np.r_[True, is_pursuit[1:] != is_pursuit[:-1]])
    purs_durations: List[float] = []
    purs_mean_speeds: List[float] = []
    for seg in np.unique(seg_ids):
        mask = seg_ids == seg
        if is_pursuit[mask][0]:
            dur = float(np.nansum(dt[mask]))
            mspd = float(np.nanmean(v[mask])) if np.any(mask) else np.nan
            purs_durations.append(dur)
            purs_mean_speeds.append(mspd)

    # Overall direction consistency during pursuit only
    if np.any(is_pursuit):
        cos_t = np.cos(theta[is_pursuit])
        sin_t = np.sin(theta[is_pursuit])
        R = float(np.sqrt(np.nanmean(cos_t) ** 2 + np.nanmean(sin_t) ** 2))
        dir_consistency = R  # 0..1
    else:
        dir_consistency = np.nan

    out["pursuit_ratio"] = (pursuit_time / total_time) if total_time > 0 else np.nan
    out["pursuit_count"] = float(len(purs_durations))
    out["pursuit_dur_mean"] = float(np.nanmean(purs_durations)) if purs_durations else np.nan
    out["pursuit_dur_std"] = float(np.nanstd(purs_durations)) if purs_durations else np.nan
    out["pursuit_speed_mean"] = float(np.nanmean(purs_mean_speeds)) if purs_mean_speeds else np.nan
    out["pursuit_dir_consistency"] = float(dir_consistency) if dir_consistency == dir_consistency else np.nan
    return out
