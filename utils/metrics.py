"""
Metrics and Evaluation Functions

This module contains functions for calculating EER, FMR, FRR, time-to-detection,
and other evaluation metrics for gaze-based continuous authentication.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from scipy import stats
import matplotlib.pyplot as plt


def calculate_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """
    Calculate Equal Error Rate (EER) and corresponding threshold.
    
    EER is the rate at which False Match Rate (FMR) equals False Reject Rate (FRR).
    
    Args:
        y_true (np.ndarray): True binary labels (0 for impostor, 1 for genuine)
        y_scores (np.ndarray): Confidence scores
        
    Returns:
        Tuple[float, float]: EER value and corresponding threshold
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    
    # FRR = 1 - TPR (False Reject Rate)
    frr = 1 - tpr
    
    # FMR = FPR (False Match Rate)
    fmr = fpr
    
    # Find where FMR and FRR are closest
    abs_diff = np.abs(frr - fmr)
    min_index = np.argmin(abs_diff)
    
    eer = (frr[min_index] + fmr[min_index]) / 2
    eer_threshold = thresholds[min_index]
    
    return eer, eer_threshold


def calculate_fmr_frr(y_true: np.ndarray, y_scores: np.ndarray, 
                     threshold: float) -> Tuple[float, float]:
    """
    Calculate False Match Rate (FMR) and False Reject Rate (FRR).
    
    Args:
        y_true (np.ndarray): True binary labels (0 for impostor, 1 for genuine)
        y_scores (np.ndarray): Confidence scores
        threshold (float): Decision threshold
        
    Returns:
        Tuple[float, float]: FMR and FRR values
    """
    # Predictions based on threshold
    y_pred = (y_scores >= threshold).astype(int)
    
    # True Positives, False Positives, True Negatives, False Negatives
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # FMR = FP / (FP + TN) - rate of accepting impostors
    fmr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    # FRR = FN / (FN + TP) - rate of rejecting genuine users
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    return fmr, frr


def calculate_time_to_detection(authentication_results: List[bool], 
                              ground_truth: List[bool],
                              window_size_sec: float = 5.0) -> Dict[str, float]:
    """
    Calculate time-to-detection metrics for continuous authentication.
    
    Measures how quickly an impostor is detected.
    
    Args:
        authentication_results (List[bool]): Authentication decisions over time (True = authenticated)
        ground_truth (List[bool]): True authentication states (True = genuine user)
        window_size_sec (float): Size of each window in seconds
        
    Returns:
        Dict[str, float]: Time-to-detection metrics
    """
    metrics = {}
    
    # Find impostor sessions (ground_truth = False)
    detection_times = []
    
    i = 0
    while i < len(ground_truth):
        # Find start of impostor session
        if not ground_truth[i]:
            # Find when impostor is detected (authentication_results = False)
            detection_idx = None
            for j in range(i, len(authentication_results)):
                if not authentication_results[j]:
                    detection_idx = j
                    break
                if j < len(ground_truth) and ground_truth[j]:
                    # Genuine user returned
                    break
            
            if detection_idx is not None:
                time_to_detect = (detection_idx - i) * window_size_sec
                detection_times.append(time_to_detect)
            
            # Skip to end of impostor session
            while i < len(ground_truth) and not ground_truth[i]:
                i += 1
        else:
            i += 1
    
    if len(detection_times) > 0:
        metrics['mean_detection_time'] = np.mean(detection_times)
        metrics['median_detection_time'] = np.median(detection_times)
        metrics['min_detection_time'] = np.min(detection_times)
        metrics['max_detection_time'] = np.max(detection_times)
        metrics['num_detected'] = len(detection_times)
    else:
        metrics['mean_detection_time'] = 0.0
        metrics['median_detection_time'] = 0.0
        metrics['min_detection_time'] = 0.0
        metrics['max_detection_time'] = 0.0
        metrics['num_detected'] = 0
    
    return metrics


def calculate_continuous_metrics(ground_truth: List[bool], 
                               predictions: List[bool],
                               confidence_scores: Optional[List[float]] = None) -> Dict[str, float]:
    """
    Calculate comprehensive continuous authentication metrics.
    
    Args:
        ground_truth (List[bool]): True authentication states (True = genuine)
        predictions (List[bool]): Predicted authentication states (True = authenticated)
        confidence_scores (List[float]): Confidence scores (optional)
        
    Returns:
        Dict[str, float]: Comprehensive metrics dictionary
    """
    ground_truth = np.array(ground_truth)
    predictions = np.array(predictions)
    
    metrics = {}
    
    # Basic accuracy metrics
    metrics['accuracy'] = np.mean(ground_truth == predictions)
    
    # True/False Positives/Negatives
    tp = np.sum((ground_truth == True) & (predictions == True))
    fp = np.sum((ground_truth == False) & (predictions == True))
    tn = np.sum((ground_truth == False) & (predictions == False))
    fn = np.sum((ground_truth == True) & (predictions == False))
    
    metrics['true_positives'] = int(tp)
    metrics['false_positives'] = int(fp)
    metrics['true_negatives'] = int(tn)
    metrics['false_negatives'] = int(fn)
    
    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1_score'] = f1
    
    # FMR and FRR
    fmr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    metrics['fmr'] = fmr
    metrics['frr'] = frr
    
    # If confidence scores provided, calculate EER
    if confidence_scores is not None:
        y_true_binary = ground_truth.astype(int)
        eer, eer_threshold = calculate_eer(y_true_binary, confidence_scores)
        metrics['eer'] = eer
        metrics['eer_threshold'] = eer_threshold
    
    return metrics


def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray, 
                  title: str = "ROC Curve", save_path: Optional[str] = None) -> None:
    """
    Plot ROC curve for authentication performance.
    
    Args:
        y_true (np.ndarray): True binary labels (0 for impostor, 1 for genuine)
        y_scores (np.ndarray): Confidence scores
        title (str): Plot title
        save_path (str): Optional path to save the plot
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    roc_auc = auc(fpr, tpr)
    
    # Calculate EER
    eer, eer_threshold = calculate_eer(y_true, y_scores)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random classifier')
    plt.plot([eer], [1-eer], 'ro', markersize=8, 
             label=f'EER = {eer:.3f}')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FMR)')
    plt.ylabel('True Positive Rate (1 - FRR)')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    else:
        plt.show()


def plot_detection_timeline(authentication_results: List[bool], 
                          ground_truth: List[bool],
                          window_size_sec: float = 5.0,
                          title: str = "Authentication Timeline",
                          save_path: Optional[str] = None) -> None:
    """
    Plot authentication detection timeline.
    
    Args:
        authentication_results (List[bool]): Authentication decisions (True = authenticated)
        ground_truth (List[bool]): True authentication states (True = genuine user)
        window_size_sec (float): Window size in seconds
        title (str): Plot title
        save_path (str): Optional path to save the plot
    """
    timestamps = np.arange(len(authentication_results)) * window_size_sec
    
    plt.figure(figsize=(14, 6))
    
    # Plot ground truth
    plt.subplot(2, 1, 1)
    plt.plot(timestamps, ground_truth, 'g-', linewidth=2, label='Ground Truth')
    plt.fill_between(timestamps, 0, ground_truth, alpha=0.3, color='green')
    plt.ylabel('Genuine User')
    plt.title(f'{title} - Ground Truth')
    plt.ylim([-0.1, 1.1])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot authentication results
    plt.subplot(2, 1, 2)
    plt.plot(timestamps, authentication_results, 'b-', linewidth=2, label='Authentication')
    plt.fill_between(timestamps, 0, authentication_results, alpha=0.3, color='blue')
    
    # Highlight errors
    errors = np.array(authentication_results) != np.array(ground_truth)
    if np.any(errors):
        error_times = timestamps[errors]
        plt.scatter(error_times, np.ones(len(error_times)) * 0.5, 
                   color='red', s=50, marker='x', label='Errors', zorder=10)
    
    plt.ylabel('Authenticated')
    plt.xlabel('Time (seconds)')
    plt.title(f'{title} - Authentication Decisions')
    plt.ylim([-0.1, 1.1])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Timeline plot saved to {save_path}")
    else:
        plt.show()


def plot_performance_over_time(metrics_by_period: Dict[int, Dict[str, float]],
                              metric_name: str = 'accuracy',
                              title: str = "Performance Over Time",
                              save_path: Optional[str] = None) -> None:
    """
    Plot how performance changes over time periods (for drift analysis).
    
    Args:
        metrics_by_period: Dictionary mapping time_period to metrics dict
        metric_name: Which metric to plot (e.g., 'accuracy', 'eer', 'f1_score')
        title: Plot title
        save_path: Optional path to save the plot
    """
    periods = sorted(metrics_by_period.keys())
    values = [metrics_by_period[p][metric_name] for p in periods]
    
    plt.figure(figsize=(10, 6))
    plt.plot(periods, values, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Time Period')
    plt.ylabel(metric_name.replace('_', ' ').title())
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(periods, values, 1)
    p = np.poly1d(z)
    plt.plot(periods, p(periods), "r--", alpha=0.8, label='Trend')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance plot saved to {save_path}")
    else:
        plt.show()
