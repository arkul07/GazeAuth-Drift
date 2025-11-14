"""
Gaze Feature Extractor

This module contains core functions for calculating behavioral gaze features
from raw gaze data for continuous authentication.

Features extracted:
- Fixation features (duration, count, dispersion)
- Saccade features (amplitude, velocity, direction)
- Scanpath features (length, entropy, coverage)
- Velocity features (mean, std, max)
- Statistical features (position, range, binocular)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy


def create_windows(df: pd.DataFrame, window_size_sec: float = 5.0, 
                  overlap_sec: float = 1.0) -> List[pd.DataFrame]:
    """
    Split gaze data into overlapping time windows.
    
    Args:
        df: Raw gaze data with timestamp_sec column
        window_size_sec: Window size in seconds (default 5.0)
        overlap_sec: Overlap between windows in seconds (default 1.0)
        
    Returns:
        List of DataFrames, one per window
    """
    windows = []
    stride_sec = window_size_sec - overlap_sec
    
    # Group by user and session to maintain boundaries
    for (user_id, session), group in df.groupby(['user_id', 'session']):
        group = group.sort_values('timestamp_sec').reset_index(drop=True)
        
        start_time = group['timestamp_sec'].min()
        end_time = group['timestamp_sec'].max()
        
        current_start = start_time
        window_id = 0
        
        while current_start + window_size_sec <= end_time:
            current_end = current_start + window_size_sec
            
            # Extract window
            window_mask = (group['timestamp_sec'] >= current_start) & \
                         (group['timestamp_sec'] < current_end)
            window_df = group[window_mask].copy()
            
            if len(window_df) >= 10:  # Minimum samples per window
                window_df['window_id'] = window_id
                windows.append(window_df)
                window_id += 1
            
            current_start += stride_sec
    
    return windows


def detect_fixations(df: pd.DataFrame, velocity_threshold: float = 100.0) -> List[Dict]:
    """
    Detect fixations using velocity-threshold algorithm (I-VT).
    
    Fixations are periods where gaze velocity is below threshold.
    
    Args:
        df: Window of gaze data
        velocity_threshold: Velocity threshold in degrees/second (default 100)
        
    Returns:
        List of fixation dictionaries with start_idx, end_idx, duration, center_x, center_y
    """
    # Calculate velocity
    dx = df['gaze_x'].diff()
    dy = df['gaze_y'].diff()
    dt = df['timestamp_sec'].diff()
    
    velocity = np.sqrt(dx**2 + dy**2) / dt
    velocity = velocity.fillna(0)
    
    # Identify fixation points (velocity < threshold)
    is_fixation = velocity < velocity_threshold
    
    # Find continuous fixation periods
    fixations = []
    in_fixation = False
    start_idx = None
    
    for idx, (is_fix, row) in enumerate(zip(is_fixation, df.itertuples())):
        if is_fix and not in_fixation:
            # Start new fixation
            in_fixation = True
            start_idx = idx
        elif not is_fix and in_fixation:
            # End fixation
            in_fixation = False
            end_idx = idx
            
            # Calculate fixation properties
            fix_data = df.iloc[start_idx:end_idx]
            if len(fix_data) >= 3:  # Minimum 3 samples for valid fixation
                fixations.append({
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'duration': fix_data['timestamp_sec'].iloc[-1] - fix_data['timestamp_sec'].iloc[0],
                    'center_x': fix_data['gaze_x'].mean(),
                    'center_y': fix_data['gaze_y'].mean(),
                    'dispersion': np.sqrt(fix_data['gaze_x'].std()**2 + fix_data['gaze_y'].std()**2)
                })
    
    # Handle case where fixation extends to end of window
    if in_fixation and start_idx is not None:
        fix_data = df.iloc[start_idx:]
        if len(fix_data) >= 3:
            fixations.append({
                'start_idx': start_idx,
                'end_idx': len(df),
                'duration': fix_data['timestamp_sec'].iloc[-1] - fix_data['timestamp_sec'].iloc[0],
                'center_x': fix_data['gaze_x'].mean(),
                'center_y': fix_data['gaze_y'].mean(),
                'dispersion': np.sqrt(fix_data['gaze_x'].std()**2 + fix_data['gaze_y'].std()**2)
            })
    
    return fixations


def detect_saccades(df: pd.DataFrame, fixations: List[Dict]) -> List[Dict]:
    """
    Detect saccades as movements between fixations.
    
    Args:
        df: Window of gaze data
        fixations: List of detected fixations
        
    Returns:
        List of saccade dictionaries with amplitude, velocity, duration, direction
    """
    saccades = []
    
    for i in range(len(fixations) - 1):
        fix1 = fixations[i]
        fix2 = fixations[i + 1]
        
        # Saccade is the movement between fixations
        saccade_data = df.iloc[fix1['end_idx']:fix2['start_idx']]
        
        if len(saccade_data) > 0:
            # Calculate saccade properties
            amplitude = np.sqrt(
                (fix2['center_x'] - fix1['center_x'])**2 + 
                (fix2['center_y'] - fix1['center_y'])**2
            )
            
            duration = (
                saccade_data['timestamp_sec'].iloc[-1] - 
                saccade_data['timestamp_sec'].iloc[0]
            ) if len(saccade_data) > 1 else 0.001
            
            velocity = amplitude / duration if duration > 0 else 0
            
            # Direction (angle in radians)
            dx = fix2['center_x'] - fix1['center_x']
            dy = fix2['center_y'] - fix1['center_y']
            direction = np.arctan2(dy, dx)
            
            saccades.append({
                'amplitude': amplitude,
                'velocity': velocity,
                'duration': duration,
                'direction': direction
            })
    
    return saccades


def calculate_fixation_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate fixation-based features.
    
    Args:
        df: Gaze data window
        
    Returns:
        Dictionary of fixation features
    """
    fixations = detect_fixations(df)
    
    if len(fixations) == 0:
        return {
            'fixation_count': 0,
            'fixation_duration_mean': 0,
            'fixation_duration_std': 0,
            'fixation_duration_max': 0,
            'fixation_duration_min': 0,
            'fixation_dispersion_mean': 0,
            'fixation_dispersion_std': 0,
        }
    
    durations = [f['duration'] for f in fixations]
    dispersions = [f['dispersion'] for f in fixations]
    
    return {
        'fixation_count': len(fixations),
        'fixation_duration_mean': np.mean(durations),
        'fixation_duration_std': np.std(durations),
        'fixation_duration_max': np.max(durations),
        'fixation_duration_min': np.min(durations),
        'fixation_dispersion_mean': np.mean(dispersions),
        'fixation_dispersion_std': np.std(dispersions),
    }


def calculate_saccade_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate saccade-based features.
    
    Args:
        df: Gaze data window
        
    Returns:
        Dictionary of saccade features
    """
    fixations = detect_fixations(df)
    saccades = detect_saccades(df, fixations)
    
    if len(saccades) == 0:
        return {
            'saccade_count': 0,
            'saccade_amplitude_mean': 0,
            'saccade_amplitude_std': 0,
            'saccade_amplitude_max': 0,
            'saccade_velocity_mean': 0,
            'saccade_velocity_std': 0,
            'saccade_velocity_max': 0,
            'saccade_duration_mean': 0,
            'saccade_direction_entropy': 0,
        }
    
    amplitudes = [s['amplitude'] for s in saccades]
    velocities = [s['velocity'] for s in saccades]
    durations = [s['duration'] for s in saccades]
    directions = [s['direction'] for s in saccades]
    
    # Direction entropy (binned into 8 directions)
    direction_hist, _ = np.histogram(directions, bins=8, range=(-np.pi, np.pi))
    direction_entropy = entropy(direction_hist + 1)  # +1 to avoid log(0)
    
    return {
        'saccade_count': len(saccades),
        'saccade_amplitude_mean': np.mean(amplitudes),
        'saccade_amplitude_std': np.std(amplitudes),
        'saccade_amplitude_max': np.max(amplitudes),
        'saccade_velocity_mean': np.mean(velocities),
        'saccade_velocity_std': np.std(velocities),
        'saccade_velocity_max': np.max(velocities),
        'saccade_duration_mean': np.mean(durations),
        'saccade_direction_entropy': direction_entropy,
    }


def calculate_scanpath_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate scanpath-based features.
    
    Args:
        df: Gaze data window
        
    Returns:
        Dictionary of scanpath features
    """
    # Scanpath length (total distance traveled)
    dx = df['gaze_x'].diff()
    dy = df['gaze_y'].diff()
    distances = np.sqrt(dx**2 + dy**2)
    scanpath_length = distances.sum()
    
    # Scanpath coverage (area explored)
    x_range = df['gaze_x'].max() - df['gaze_x'].min()
    y_range = df['gaze_y'].max() - df['gaze_y'].min()
    coverage = x_range * y_range
    
    # Scanpath entropy (spatial distribution)
    # Divide screen into grid and calculate entropy
    x_bins = np.histogram(df['gaze_x'], bins=10)[0]
    y_bins = np.histogram(df['gaze_y'], bins=10)[0]
    spatial_entropy = entropy(x_bins + 1) + entropy(y_bins + 1)
    
    # Fixation-based scanpath features
    fixations = detect_fixations(df)
    if len(fixations) > 1:
        # Distance between consecutive fixations
        fix_distances = []
        for i in range(len(fixations) - 1):
            dx = fixations[i+1]['center_x'] - fixations[i]['center_x']
            dy = fixations[i+1]['center_y'] - fixations[i]['center_y']
            fix_distances.append(np.sqrt(dx**2 + dy**2))
        
        fixation_distance_mean = np.mean(fix_distances)
        fixation_distance_std = np.std(fix_distances)
    else:
        fixation_distance_mean = 0
        fixation_distance_std = 0
    
    return {
        'scanpath_length': scanpath_length,
        'scanpath_coverage': coverage,
        'scanpath_entropy': spatial_entropy,
        'fixation_distance_mean': fixation_distance_mean,
        'fixation_distance_std': fixation_distance_std,
    }


def calculate_velocity_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate velocity-based features.
    
    Args:
        df: Gaze data window
        
    Returns:
        Dictionary of velocity features
    """
    # Calculate velocity
    dx = df['gaze_x'].diff()
    dy = df['gaze_y'].diff()
    dt = df['timestamp_sec'].diff()
    
    velocity = np.sqrt(dx**2 + dy**2) / dt
    velocity = velocity.dropna()
    
    if len(velocity) == 0:
        return {
            'velocity_mean': 0,
            'velocity_std': 0,
            'velocity_max': 0,
            'velocity_median': 0,
            'velocity_q25': 0,
            'velocity_q75': 0,
        }
    
    # Acceleration
    acceleration = velocity.diff() / dt.iloc[1:]
    acceleration = acceleration.dropna()
    
    return {
        'velocity_mean': velocity.mean(),
        'velocity_std': velocity.std(),
        'velocity_max': velocity.max(),
        'velocity_median': velocity.median(),
        'velocity_q25': velocity.quantile(0.25),
        'velocity_q75': velocity.quantile(0.75),
        'acceleration_mean': acceleration.mean() if len(acceleration) > 0 else 0,
        'acceleration_std': acceleration.std() if len(acceleration) > 0 else 0,
    }


def calculate_statistical_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate statistical features from gaze data.
    
    Args:
        df: Gaze data window
        
    Returns:
        Dictionary of statistical features
    """
    features = {
        # Gaze position statistics
        'gaze_x_mean': df['gaze_x'].mean(),
        'gaze_x_std': df['gaze_x'].std(),
        'gaze_x_min': df['gaze_x'].min(),
        'gaze_x_max': df['gaze_x'].max(),
        'gaze_x_range': df['gaze_x'].max() - df['gaze_x'].min(),
        'gaze_y_mean': df['gaze_y'].mean(),
        'gaze_y_std': df['gaze_y'].std(),
        'gaze_y_min': df['gaze_y'].min(),
        'gaze_y_max': df['gaze_y'].max(),
        'gaze_y_range': df['gaze_y'].max() - df['gaze_y'].min(),
    }
    
    # Binocular disparity (if available)
    if 'left_eye_x' in df.columns and 'right_eye_x' in df.columns:
        disparity_x = (df['left_eye_x'] - df['right_eye_x']).abs()
        disparity_y = (df['left_eye_y'] - df['right_eye_y']).abs()
        
        features['binocular_disparity_x_mean'] = disparity_x.mean()
        features['binocular_disparity_x_std'] = disparity_x.std()
        features['binocular_disparity_y_mean'] = disparity_y.mean()
        features['binocular_disparity_y_std'] = disparity_y.std()
    
    return features


def extract_window_features(window_df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract all features from a single window.
    
    Args:
        window_df: Single window of gaze data
        
    Returns:
        Dictionary of all extracted features
    """
    features = {}
    
    # Add metadata
    features['user_id'] = window_df['user_id'].iloc[0]
    features['session'] = window_df['session'].iloc[0]
    if 'window_id' in window_df.columns:
        features['window_id'] = window_df['window_id'].iloc[0]
    if 'time_period' in window_df.columns:
        features['time_period'] = window_df['time_period'].iloc[0]
    
    # Extract all feature types
    features.update(calculate_fixation_features(window_df))
    features.update(calculate_saccade_features(window_df))
    features.update(calculate_scanpath_features(window_df))
    features.update(calculate_velocity_features(window_df))
    features.update(calculate_statistical_features(window_df))
    
    return features


def extract_gaze_features(df: pd.DataFrame, window_size_sec: float = 5.0,
                         overlap_sec: float = 1.0) -> pd.DataFrame:
    """
    Extract behavioral gaze features from raw gaze data.
    
    Main function to extract all features from gaze data.
    
    Args:
        df: Raw gaze data with columns [user_id, timestamp_sec, gaze_x, gaze_y, ...]
        window_size_sec: Window size in seconds (default 5.0)
        overlap_sec: Overlap between windows in seconds (default 1.0)
        
    Returns:
        DataFrame with one row per window and columns for each feature
    """
    print(f"Extracting features with {window_size_sec}s windows, {overlap_sec}s overlap...")
    
    # Create windows
    windows = create_windows(df, window_size_sec, overlap_sec)
    print(f"Created {len(windows)} windows")
    
    # Extract features from each window
    feature_list = []
    for i, window in enumerate(windows):
        if (i + 1) % 10 == 0:
            print(f"  Processing window {i+1}/{len(windows)}...")
        
        try:
            features = extract_window_features(window)
            feature_list.append(features)
        except Exception as e:
            print(f"  Warning: Failed to extract features from window {i}: {e}")
    
    features_df = pd.DataFrame(feature_list)
    print(f"✅ Extracted {len(features_df)} feature vectors with {len(features_df.columns)} features each")
    
    return features_df
