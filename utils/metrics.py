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


# ===================== Biometric Score Utilities =====================
def build_user_score_vectors(y_true: np.ndarray, proba: np.ndarray, classes: List) -> Dict[str, Dict[str, np.ndarray]]:
    """Construct genuine and impostor score arrays per user.

    Args:
        y_true: Array of true user labels (same length as proba rows)
        proba: (n_windows, n_users) probability estimates (higher = more likely user)
        classes: List/array of user identifiers aligned to proba columns

    Returns:
        dict mapping user -> { 'genuine': scores_when_true, 'impostor': scores_when_not_true }
    """
    result: Dict[str, Dict[str, np.ndarray]] = {}
    for idx, user in enumerate(classes):
        user_scores = proba[:, idx]
        genuine_mask = (y_true == user)
        result[str(user)] = {
            'genuine': user_scores[genuine_mask],
            'impostor': user_scores[~genuine_mask]
        }
    return result


def compute_eer_from_scores(genuine_scores: np.ndarray, impostor_scores: np.ndarray) -> Tuple[float, float]:
    """Compute EER given genuine + impostor score distributions.

    Returns:
        (eer, threshold)
    """
    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        return 0.0, 0.0
    scores = np.concatenate([genuine_scores, impostor_scores])
    labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
    # filter out NaN/Inf scores to avoid sklearn errors
    finite_mask = np.isfinite(scores)
    scores = scores[finite_mask]
    labels = labels[finite_mask]
    if scores.size == 0:
        return 0.0, 0.0
    # guard if all scores identical -> roc_curve may return nan thresholds
    try:
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    except ValueError:
        return 0.5, 0.0
    frr = 1 - tpr
    diff = np.abs(fpr - frr)
    i = np.argmin(diff)
    eer = (fpr[i] + frr[i]) / 2
    thr = thresholds[i] if np.isfinite(thresholds[i]) else 0.0
    return float(eer), float(thr)


def aggregate_global_eer(per_user_scores: Dict[str, Dict[str, np.ndarray]]) -> Tuple[float, float]:
    """Aggregate global EER treating all users' genuine/impostor scores pooled.

    Args:
        per_user_scores: Output of build_user_score_vectors
    Returns:
        (global_eer, threshold)
    """
    all_genuine = []
    all_impostor = []
    for user, d in per_user_scores.items():
        all_genuine.append(d['genuine'])
        all_impostor.append(d['impostor'])
    if not all_genuine or not all_impostor:
        return 0.0, 0.0
    # filter out any empty arrays before concatenate
    genuines = [arr for arr in all_genuine if arr is not None and arr.size > 0]
    impostors = [arr for arr in all_impostor if arr is not None and arr.size > 0]
    if not genuines or not impostors:
        return 0.0, 0.0
    g = np.concatenate(genuines)
    imp = np.concatenate(impostors)
    return compute_eer_from_scores(g, imp)


def far_frr_at_threshold(genuine_scores: np.ndarray, impostor_scores: np.ndarray, threshold: float) -> Tuple[float, float]:
    """Compute FAR (FMR) and FRR for a given threshold.
    FAR = fraction of impostor scores >= threshold
    FRR = fraction of genuine scores < threshold
    """
    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        return 0.0, 0.0
    # filter non-finite
    g = genuine_scores[np.isfinite(genuine_scores)]
    imp = impostor_scores[np.isfinite(impostor_scores)]
    if len(g) == 0 or len(imp) == 0:
        return 0.0, 0.0
    far = np.mean(imp >= threshold)
    frr = np.mean(g < threshold)
    return far, frr


# ===================== Sequence-level Fusion Utilities =====================
def aggregate_window_proba_to_sequences(
    proba: np.ndarray,
    window_to_seq: List,
    reducer: str = "mean"
) -> Tuple[np.ndarray, List]:
    """Aggregate window-level class probabilities to sequence-level scores.

    Args:
        proba: (n_windows, n_classes) window-level probabilities or scores
        window_to_seq: list/array of sequence identifiers per window (length n_windows)
        reducer: 'mean' | 'median' | 'max' | 'softmax' (logit-mean)

    Returns:
        seq_proba: (n_sequences, n_classes) aggregated scores per sequence
        seq_ids: list of unique sequence identifiers in aggregation order
    """
    window_to_seq = np.array(window_to_seq)
    seq_ids = list(pd.unique(window_to_seq))
    seq_to_idx = {sid: i for i, sid in enumerate(seq_ids)}
    n_classes = proba.shape[1]
    # collect list per sequence
    buckets: List[List[np.ndarray]] = [[] for _ in seq_ids]
    for w_idx, sid in enumerate(window_to_seq):
        buckets[seq_to_idx[sid]].append(proba[w_idx])

    seq_proba = np.zeros((len(seq_ids), n_classes), dtype=float)
    for i, arrs in enumerate(buckets):
        if not arrs:
            continue
        stack = np.stack(arrs, axis=0)
        # If stack is entirely non-finite, leave as zeros and continue
        if not np.isfinite(stack).any():
            continue
        if reducer == "mean":
            seq_proba[i] = np.nanmean(stack, axis=0)
        elif reducer == "median":
            seq_proba[i] = np.nanmedian(stack, axis=0)
        elif reducer == "max":
            seq_proba[i] = np.nanmax(stack, axis=0)
        elif reducer == "softmax":
            # average logits: log-mean-exp for numerical stability
            eps = 1e-12
            stack_clip = np.clip(stack, eps, 1.0)
            logits = np.log(stack_clip)
            m = np.max(logits, axis=0, keepdims=True)
            logmeanexp = m + np.log(np.mean(np.exp(logits - m), axis=0, keepdims=True))
            seq_proba[i] = np.exp(logmeanexp).ravel()
            # renormalize to sum=1 in case of numerical drift
            s = seq_proba[i].sum()
            if s > 0:
                seq_proba[i] /= s
        else:
            seq_proba[i] = np.nanmean(stack, axis=0)
        # Replace non-finite or degenerate rows with a uniform distribution
        if not np.isfinite(seq_proba[i]).all() or seq_proba[i].sum() == 0:
            seq_proba[i] = np.ones(n_classes, dtype=float) / n_classes
    return seq_proba, seq_ids


def sequence_level_identification_accuracy(
    y_true_windows: np.ndarray,
    proba_windows: np.ndarray,
    window_to_seq: List,
    classes: List,
    reducer: str = "mean"
) -> float:
    """Compute sequence-level (trial-level) identification accuracy by fusing
    window probabilities within each sequence and taking argmax.

    Args:
        y_true_windows: (n_windows,) true class per window
        proba_windows: (n_windows, n_classes) probabilities per window
        window_to_seq: sequence id per window (same length)
        classes: class labels aligned with proba columns
        reducer: aggregation method ('mean'|'median'|'max'|'softmax')

    Returns:
        float accuracy over sequences
    """
    seq_proba, seq_ids = aggregate_window_proba_to_sequences(proba_windows, window_to_seq, reducer)
    # determine true label per sequence by majority vote of window labels
    window_to_seq_arr = np.array(window_to_seq)
    correct = 0
    for i, sid in enumerate(seq_ids):
        mask = window_to_seq_arr == sid
        true_labels = y_true_windows[mask]
        # majority vote
        if len(true_labels) == 0:
            continue
        vals, counts = np.unique(true_labels, return_counts=True)
        seq_true = vals[np.argmax(counts)]
        pred_idx = int(np.argmax(seq_proba[i]))
        pred_label = classes[pred_idx]
        if pred_label == seq_true:
            correct += 1
    return correct / len(seq_ids) if len(seq_ids) > 0 else 0.0


def sequence_level_scores_for_verification(
    y_true_windows: np.ndarray,
    proba_windows: np.ndarray,
    window_to_seq: List,
    classes: List,
    reducer: str = "mean"
) -> Dict[str, Dict[str, np.ndarray]]:
    """Build per-user genuine/impostor score vectors at sequence level.

    For each sequence, compute the target-user score from the aggregated
    probabilities. Then split aggregated scores into genuine vs impostor
    by comparing the sequence's majority-vote label.
    """
    seq_proba, seq_ids = aggregate_window_proba_to_sequences(proba_windows, window_to_seq, reducer)
    window_to_seq_arr = np.array(window_to_seq)
    # determine true label per sequence by majority vote
    seq_true_labels: List = []
    for sid in seq_ids:
        mask = window_to_seq_arr == sid
        true_labels = y_true_windows[mask]
        vals, counts = np.unique(true_labels, return_counts=True)
        seq_true_labels.append(vals[np.argmax(counts)])

    per_user: Dict[str, Dict[str, List[float]]] = {str(u): {"genuine": [], "impostor": []} for u in classes}
    for i, seq_true in enumerate(seq_true_labels):
        for col_idx, user in enumerate(classes):
            score = float(seq_proba[i, col_idx])
            if user == seq_true:
                per_user[str(user)]["genuine"].append(score)
            else:
                per_user[str(user)]["impostor"].append(score)

    # convert lists to arrays
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for user, d in per_user.items():
        out[user] = {
            "genuine": np.array(d["genuine"], dtype=float),
            "impostor": np.array(d["impostor"], dtype=float)
        }
    return out

