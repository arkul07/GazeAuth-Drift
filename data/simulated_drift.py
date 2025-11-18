"""
Simulated Drift Data Generator

This module contains functions for generating simulated longitudinal drift data
to test the robustness of gaze-based authentication systems.

Drift Types:
- Linear: Gradual shift over time (e.g., user gets more comfortable with VR)
- Exponential: Rapid initial change that stabilizes (e.g., VR sickness adaptation)
- Periodic: Cyclical patterns (e.g., fatigue throughout day)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from copy import deepcopy


def generate_drift_patterns(
    base_data: pd.DataFrame,
    drift_type: str,
    drift_magnitude: float = 0.1,
    num_periods: int = 5,
) -> List[pd.DataFrame]:
    """
    Generate simulated drift patterns for testing temporal robustness.

    This creates multiple "time periods" where gaze patterns progressively change
    to simulate how a user's behavior evolves over weeks/months.

    Args:
        base_data: Base gaze data without drift (from GazebaseVR)
        drift_type: Type of drift ('linear', 'exponential', 'periodic', 'none')
        drift_magnitude: Magnitude of drift effect (0.0-1.0, typically 0.05-0.3)
        num_periods: Number of time periods to simulate (e.g., 5 = 5 weeks)

    Returns:
        List of DataFrames, one per time period with drift applied
    """
    drift_periods = []

    for period in range(num_periods):
        # Copy base data for this period
        period_data = base_data.copy()

        # Calculate drift factor based on type
        if drift_type == "linear":
            drift_factor = drift_magnitude * (period / (num_periods - 1))
        elif drift_type == "exponential":
            # Rapid change early, stabilizes later
            drift_factor = drift_magnitude * (1 - np.exp(-2 * period / num_periods))
        elif drift_type == "periodic":
            # Sine wave pattern
            drift_factor = drift_magnitude * np.sin(2 * np.pi * period / num_periods)
        elif drift_type == "none":
            drift_factor = 0.0
        else:
            raise ValueError(f"Unknown drift_type: {drift_type}")

        # Apply drift to gaze coordinates
        period_data = apply_drift_to_features(period_data, drift_factor)

        # Add time period metadata
        period_data["time_period"] = period
        period_data["drift_factor"] = drift_factor

        drift_periods.append(period_data)

        print(f"  Period {period}: drift_factor = {drift_factor:.4f}")

    return drift_periods


def apply_drift_to_features(data: pd.DataFrame, drift_factor: float) -> pd.DataFrame:
    """
    Apply drift transformation to gaze features.

    Simulates how gaze patterns change over time:
    - Spatial shift: Gaze positions shift slightly
    - Scale change: Gaze range expands/contracts
    - Noise increase: More variability in patterns

    Args:
        data: Gaze data
        drift_factor: How much drift to apply (0.0 = none, 1.0 = maximum)

    Returns:
        Data with drift applied
    """
    df_drift = data.copy()

    # 1. Spatial Shift - gaze positions drift in a consistent direction
    if "gaze_x" in df_drift.columns and "gaze_y" in df_drift.columns:
        shift_x = drift_factor * 0.5  # Slight rightward drift
        shift_y = drift_factor * 0.3  # Slight upward drift

        df_drift["gaze_x"] = df_drift["gaze_x"] + shift_x
        df_drift["gaze_y"] = df_drift["gaze_y"] + shift_y

    # 2. Scale Change - gaze range expands (user gets more comfortable)
    if "gaze_x" in df_drift.columns and "gaze_y" in df_drift.columns:
        scale_factor = 1 + (drift_factor * 0.2)  # Up to 20% scale change

        # Scale around mean
        mean_x = df_drift["gaze_x"].mean()
        mean_y = df_drift["gaze_y"].mean()

        df_drift["gaze_x"] = mean_x + (df_drift["gaze_x"] - mean_x) * scale_factor
        df_drift["gaze_y"] = mean_y + (df_drift["gaze_y"] - mean_y) * scale_factor

    # 3. Noise Increase - patterns become more variable
    if "gaze_x" in df_drift.columns and "gaze_y" in df_drift.columns:
        noise_level = drift_factor * 0.1
        noise_x = np.random.normal(0, noise_level, len(df_drift))
        noise_y = np.random.normal(0, noise_level, len(df_drift))

        df_drift["gaze_x"] = df_drift["gaze_x"] + noise_x
        df_drift["gaze_y"] = df_drift["gaze_y"] + noise_y

    # Also apply to binocular data if present
    if "left_eye_x" in df_drift.columns and "right_eye_x" in df_drift.columns:
        # Apply same transformations to individual eyes
        shift_x = drift_factor * 0.5
        shift_y = drift_factor * 0.3
        scale_factor = 1 + (drift_factor * 0.2)
        noise_level = drift_factor * 0.1

        for eye in ["left_eye", "right_eye"]:
            if f"{eye}_x" in df_drift.columns:
                # Shift
                df_drift[f"{eye}_x"] = df_drift[f"{eye}_x"] + shift_x
                df_drift[f"{eye}_y"] = df_drift[f"{eye}_y"] + shift_y

                # Scale
                mean_x = df_drift[f"{eye}_x"].mean()
                mean_y = df_drift[f"{eye}_y"].mean()
                df_drift[f"{eye}_x"] = (
                    mean_x + (df_drift[f"{eye}_x"] - mean_x) * scale_factor
                )
                df_drift[f"{eye}_y"] = (
                    mean_y + (df_drift[f"{eye}_y"] - mean_y) * scale_factor
                )

                # Noise
                noise_x = np.random.normal(0, noise_level, len(df_drift))
                noise_y = np.random.normal(0, noise_level, len(df_drift))
                df_drift[f"{eye}_x"] = df_drift[f"{eye}_x"] + noise_x
                df_drift[f"{eye}_y"] = df_drift[f"{eye}_y"] + noise_y

    return df_drift


def create_longitudinal_dataset(
    base_data: pd.DataFrame,
    num_periods: int = 5,
    drift_type: str = "linear",
    drift_magnitude: float = 0.1,
) -> pd.DataFrame:
    """
    Create a complete longitudinal dataset spanning multiple time periods.

    This is the main function to use - it generates a full synthetic dataset
    simulating data collected over weeks/months with natural drift.

    Args:
        base_data: Base gaze data from GazebaseVR
        num_periods: Number of time periods (e.g., 5 = 5 weeks)
        drift_type: Type of drift ('linear', 'exponential', 'periodic', 'none')
        drift_magnitude: How much drift to apply (0.0-1.0)

    Returns:
        Combined DataFrame with all periods
    """
    print(f"\nGenerating longitudinal dataset:")
    print(f"  Drift type: {drift_type}")
    print(f"  Drift magnitude: {drift_magnitude}")
    print(f"  Time periods: {num_periods}")

    # Generate drift for each period
    period_dfs = generate_drift_patterns(
        base_data=base_data,
        drift_type=drift_type,
        drift_magnitude=drift_magnitude,
        num_periods=num_periods,
    )

    # Add session variability to each period
    varied_periods = []
    for i, period_df in enumerate(period_dfs):
        # Add realistic session-to-session variation
        varied_df = inject_session_variability(period_df, variability_factor=0.05)
        varied_periods.append(varied_df)

    # Combine all periods
    longitudinal_df = pd.concat(varied_periods, ignore_index=True)

    print(
        f"\n✅ Generated {len(longitudinal_df):,} samples across {num_periods} periods"
    )

    return longitudinal_df


def inject_session_variability(
    data: pd.DataFrame, variability_factor: float = 0.05
) -> pd.DataFrame:
    """
    Inject realistic session-to-session variability into gaze data.

    Simulates natural variations like:
    - Different calibration each session
    - Slight head position changes
    - Environmental differences (lighting, time of day)

    Args:
        data: Gaze data
        variability_factor: Amount of random variation (0.0-1.0, typically 0.03-0.1)

    Returns:
        Data with session variability added
    """
    df_varied = data.copy()

    # Add small random offset (different calibration)
    if "gaze_x" in df_varied.columns and "gaze_y" in df_varied.columns:
        offset_x = np.random.normal(0, variability_factor)
        offset_y = np.random.normal(0, variability_factor)

        df_varied["gaze_x"] = df_varied["gaze_x"] + offset_x
        df_varied["gaze_y"] = df_varied["gaze_y"] + offset_y

    # Add random noise to individual samples
    if "gaze_x" in df_varied.columns and "gaze_y" in df_varied.columns:
        noise_x = np.random.normal(0, variability_factor * 0.5, len(df_varied))
        noise_y = np.random.normal(0, variability_factor * 0.5, len(df_varied))

        df_varied["gaze_x"] = df_varied["gaze_x"] + noise_x
        df_varied["gaze_y"] = df_varied["gaze_y"] + noise_y

    return df_varied


def visualize_drift(
    period_dfs: List[pd.DataFrame],
    feature: str = "gaze_x",
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize how gaze features drift over time periods.

    Args:
        period_dfs: List of DataFrames for each time period
        feature: Feature to visualize (e.g., 'gaze_x', 'gaze_y')
        save_path: Optional path to save the plot
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Distribution changes over time
    for i, df in enumerate(period_dfs):
        if feature in df.columns:
            axes[0].hist(df[feature].dropna(), bins=50, alpha=0.5, label=f"Period {i}")

    axes[0].set_xlabel(feature)
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(f"Distribution of {feature} Across Time Periods")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Mean and std over time
    periods = []
    means = []
    stds = []

    for i, df in enumerate(period_dfs):
        if feature in df.columns:
            periods.append(i)
            means.append(df[feature].mean())
            stds.append(df[feature].std())

    axes[1].errorbar(periods, means, yerr=stds, marker="o", capsize=5)
    axes[1].set_xlabel("Time Period")
    axes[1].set_ylabel(f"{feature} Mean ± Std")
    axes[1].set_title(f"Drift in {feature} Over Time")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()


def compare_drift_types(
    base_data: pd.DataFrame, num_periods: int = 5, drift_magnitude: float = 0.1
) -> Dict[str, List[pd.DataFrame]]:
    """
    Compare different drift types on the same base data.

    Useful for validating that your drift simulation looks realistic.

    Args:
        base_data: Base gaze data
        num_periods: Number of time periods
        drift_magnitude: Drift magnitude

    Returns:
        Dictionary mapping drift_type to list of period DataFrames
    """
    drift_types = ["none", "linear", "exponential", "periodic"]
    results = {}

    print("\nComparing drift types...")
    for drift_type in drift_types:
        print(f"\nGenerating {drift_type} drift:")
        periods = generate_drift_patterns(
            base_data=base_data,
            drift_type=drift_type,
            drift_magnitude=drift_magnitude,
            num_periods=num_periods,
        )
        results[drift_type] = periods

    return results


# ========================= Calibrated Drift (New) ========================= #

def _safe_std(s: pd.Series, eps: float = 1e-8) -> float:
    try:
        v = float(np.nanstd(s.values))
    except Exception:
        v = 0.0
    return max(v, eps)


def estimate_drift_from_sessions(
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    groupby: str = "task",
) -> Dict[str, Dict[str, float]]:
    """
    Estimate per-group drift parameters (shift/scale/noise) between S1 and S2
    in raw gaze space. Returns a dict mapping group key -> params.

    groupby: 'task' or 'user_task' (user_id+task)
    Params per group:
      - shift_x, shift_y
      - scale_x, scale_y
      - noise_x, noise_y (residual std not explained by scaling)
      - optional per-eye variants if columns exist: left_eye/right_eye
    """
    if groupby not in ("task", "user_task"):
        raise ValueError("groupby must be 'task' or 'user_task'")

    def key_fn(row):
        if groupby == "task":
            return str(row.get("task", "UNK"))
        return f"{int(row.get('user_id', -1))}:{str(row.get('task', 'UNK'))}"

    df1 = df_s1.copy(); df1["__key__"] = df1.apply(key_fn, axis=1)
    df2 = df_s2.copy(); df2["__key__"] = df2.apply(key_fn, axis=1)

    keys = sorted(set(df1["__key__"].unique()).intersection(set(df2["__key__"].unique())))
    calib: Dict[str, Dict[str, float]] = {}

    for k in keys:
        g1 = df1[df1["__key__"] == k]
        g2 = df2[df2["__key__"] == k]
        params: Dict[str, float] = {}

        # Core gaze
        if {"gaze_x", "gaze_y"}.issubset(g1.columns) and {"gaze_x", "gaze_y"}.issubset(g2.columns):
            mean1x, mean1y = float(np.nanmean(g1["gaze_x"])), float(np.nanmean(g1["gaze_y"]))
            mean2x, mean2y = float(np.nanmean(g2["gaze_x"])), float(np.nanmean(g2["gaze_y"]))
            std1x, std1y = _safe_std(g1["gaze_x"]), _safe_std(g1["gaze_y"])
            std2x, std2y = _safe_std(g2["gaze_x"]), _safe_std(g2["gaze_y"])

            params["shift_x"] = mean2x - mean1x
            params["shift_y"] = mean2y - mean1y
            params["scale_x"] = max(0.1, min(5.0, std2x / std1x))
            params["scale_y"] = max(0.1, min(5.0, std2y / std1y))

            # Residual noise beyond pure scaling of S1
            noise_x = max(0.0, std2x - params["scale_x"] * std1x)
            noise_y = max(0.0, std2y - params["scale_y"] * std1y)
            params["noise_x"] = noise_x
            params["noise_y"] = noise_y

        # Per-eye (optional)
        for eye in ("left_eye", "right_eye"):
            xcol, ycol = f"{eye}_x", f"{eye}_y"
            if xcol in g1.columns and xcol in g2.columns and ycol in g1.columns and ycol in g2.columns:
                mean1x, mean1y = float(np.nanmean(g1[xcol])), float(np.nanmean(g1[ycol]))
                mean2x, mean2y = float(np.nanmean(g2[xcol])), float(np.nanmean(g2[ycol]))
                std1x, std1y = _safe_std(g1[xcol]), _safe_std(g1[ycol])
                std2x, std2y = _safe_std(g2[xcol]), _safe_std(g2[ycol])
                params[f"{eye}_shift_x"] = mean2x - mean1x
                params[f"{eye}_shift_y"] = mean2y - mean1y
                scx = max(0.1, min(5.0, std2x / std1x))
                scy = max(0.1, min(5.0, std2y / std1y))
                params[f"{eye}_scale_x"] = scx
                params[f"{eye}_scale_y"] = scy
                params[f"{eye}_noise_x"] = max(0.0, std2x - scx * std1x)
                params[f"{eye}_noise_y"] = max(0.0, std2y - scy * std1y)

        calib[k] = params

    return calib


def _apply_calibrated_params(df: pd.DataFrame, params: Dict[str, float], alpha: float = 1.0, jitter: float = 0.05) -> pd.DataFrame:
    """
    Apply a fraction (alpha) of the calibrated drift to a copy of df.
    jitter adds small random variation to avoid deterministic patterns.
    """
    out = df.copy()
    rng = np.random.default_rng()

    # Gaze
    if "gaze_x" in out.columns and "gaze_y" in out.columns:
        shift_x = params.get("shift_x", 0.0) * alpha
        shift_y = params.get("shift_y", 0.0) * alpha
        scale_x = 1.0 + (params.get("scale_x", 1.0) - 1.0) * alpha
        scale_y = 1.0 + (params.get("scale_y", 1.0) - 1.0) * alpha
        noise_x = params.get("noise_x", 0.0) * alpha
        noise_y = params.get("noise_y", 0.0) * alpha

        mx = float(np.nanmean(out["gaze_x"]))
        my = float(np.nanmean(out["gaze_y"]))
        out["gaze_x"] = mx + (out["gaze_x"] - mx) * scale_x + shift_x + rng.normal(0, max(1e-8, noise_x), len(out))
        out["gaze_y"] = my + (out["gaze_y"] - my) * scale_y + shift_y + rng.normal(0, max(1e-8, noise_y), len(out))

        # mild random jitter
        out["gaze_x"] += rng.normal(0, jitter * max(1e-3, float(np.nanstd(out["gaze_x"]))), len(out))
        out["gaze_y"] += rng.normal(0, jitter * max(1e-3, float(np.nanstd(out["gaze_y"]))), len(out))

    for eye in ("left_eye", "right_eye"):
        xcol, ycol = f"{eye}_x", f"{eye}_y"
        if xcol in out.columns and ycol in out.columns:
            shift_x = params.get(f"{eye}_shift_x", 0.0) * alpha
            shift_y = params.get(f"{eye}_shift_y", 0.0) * alpha
            scale_x = 1.0 + (params.get(f"{eye}_scale_x", 1.0) - 1.0) * alpha
            scale_y = 1.0 + (params.get(f"{eye}_scale_y", 1.0) - 1.0) * alpha
            noise_x = params.get(f"{eye}_noise_x", 0.0) * alpha
            noise_y = params.get(f"{eye}_noise_y", 0.0) * alpha
            mx = float(np.nanmean(out[xcol])); my = float(np.nanmean(out[ycol]))
            out[xcol] = mx + (out[xcol] - mx) * scale_x + shift_x + rng.normal(0, max(1e-8, noise_x), len(out))
            out[ycol] = my + (out[ycol] - my) * scale_y + shift_y + rng.normal(0, max(1e-8, noise_y), len(out))
    return out


def create_longitudinal_dataset_calibrated(
    base_data: pd.DataFrame,
    calibration: Dict[str, Dict[str, float]],
    groupby: str = "task",
    num_periods: int = 2,
    alpha_schedule: str = "linear",
) -> pd.DataFrame:
    """
    Generate longitudinal data by applying estimated per-group drift gradually.

    - groupby: 'task' or 'user_task' must match keys in calibration
    - alpha_schedule: how to scale drift across periods; currently 'linear'
    """
    print("\nGenerating calibrated longitudinal dataset:")
    print(f"  Groupby: {groupby}")
    print(f"  Periods: {num_periods}")

    def key_from_row(r: pd.Series) -> str:
        if groupby == "task":
            return str(r.get("task", "UNK"))
        return f"{int(r.get('user_id', -1))}:{str(r.get('task', 'UNK'))}"

    # Period 0 is base
    periods = []
    for p in range(num_periods):
        if num_periods <= 1:
            alpha = 0.0
        else:
            alpha = p / (num_periods - 1) if alpha_schedule == "linear" else p / (num_periods - 1)
        dfp = base_data.copy()
        # apply per-group params
        keys = dfp.apply(key_from_row, axis=1)
        dfp["__key__"] = keys
        if p > 0:
            out_chunks = []
            for k, chunk in dfp.groupby("__key__"):
                params = calibration.get(k) or calibration.get("DEFAULT", {})
                out_chunks.append(_apply_calibrated_params(chunk, params, alpha=alpha))
            dfp = pd.concat(out_chunks, ignore_index=True)
        dfp["time_period"] = p
        periods.append(dfp)

    out = pd.concat(periods, ignore_index=True)
    print(f"\n✓ Generated {len(out):,} samples across {num_periods} periods (calibrated)")
    return out
