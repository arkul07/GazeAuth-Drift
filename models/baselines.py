"""
Baseline Classification Models

This module contains non-temporal baseline classifiers (KNN, SVM) for
gaze-based continuous authentication.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from typing import Dict, List, Tuple, Any, Optional
import pickle


class BaselineClassifier:
    """
    Baseline classifier for gaze-based authentication using non-temporal models.
    """
    
    def __init__(self, model_type: str = 'knn', **kwargs):
        """
        Initialize the baseline classifier.
        
        Args:
            model_type (str): Type of model ('knn' or 'svm')
            **kwargs: Additional parameters for the specific model
        """
        self.model_type = model_type
        self.kwargs = kwargs
        self.scaler = StandardScaler()
        
        # Initialize model based on type
        if model_type == 'knn':
            n_neighbors = kwargs.get('n_neighbors', 5)
            metric = kwargs.get('metric', 'euclidean')
            self.model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
            print(f"Initialized KNN classifier (n_neighbors={n_neighbors}, metric={metric})")
        elif model_type == 'svm':
            kernel = kwargs.get('kernel', 'rbf')
            C = kwargs.get('C', 1.0)
            gamma = kwargs.get('gamma', 'scale')
            self.model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
            print(f"Initialized SVM classifier (kernel={kernel}, C={C}, gamma={gamma})")
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'knn' or 'svm'")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the baseline model.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
        """
        print(f"Training {self.model_type.upper()} on {len(X_train)} samples...")
        
        # Handle NaN values
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate training accuracy
        train_pred = self.model.predict(X_train_scaled)
        train_acc = accuracy_score(y_train, train_pred)
        
        print(f"✅ Training complete! Training accuracy: {train_acc:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Predicted labels
        """
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Prediction probabilities
        """
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get probabilities
        probabilities = self.model.predict_proba(X_scaled)
        return probabilities
    
    def save(self, filepath: str) -> None:
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
        """
        model_data = {
            'model_type': self.model_type,
            'model': self.model,
            'scaler': self.scaler,
            'kwargs': self.kwargs
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'BaselineClassifier':
        """
        Load model from file.
        
        Args:
            filepath: Path to load model from
            
        Returns:
            Loaded BaselineClassifier instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        classifier = cls(model_data['model_type'], **model_data['kwargs'])
        classifier.model = model_data['model']
        classifier.scaler = model_data['scaler']
        print(f"Model loaded from {filepath}")
        return classifier


def evaluate_baseline_performance(y_true: np.ndarray, y_pred: np.ndarray, 
                                 y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Evaluate baseline model performance.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        y_proba (np.ndarray): Prediction probabilities (optional)
        
    Returns:
        Dict[str, float]: Performance metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['f1_score_macro'] = f1_score(y_true, y_pred, average='macro')
    metrics['f1_score_weighted'] = f1_score(y_true, y_pred, average='weighted')
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm
    
    # Per-class accuracy
    unique_labels = np.unique(y_true)
    for label in unique_labels:
        mask = y_true == label
        if mask.sum() > 0:
            class_acc = accuracy_score(y_true[mask], y_pred[mask])
            metrics[f'accuracy_class_{label}'] = class_acc
    
    return metrics


def train_test_split_by_user(features_df: pd.DataFrame, 
                             test_size: float = 0.3,
                             random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/test by time period (not random).
    
    Uses early windows for training, later windows for testing.
    
    Args:
        features_df: DataFrame with extracted features
        test_size: Proportion for test set (default 0.3)
        random_state: Random seed
        
    Returns:
        train_df, test_df
    """
    train_dfs = []
    test_dfs = []
    
    for user_id, user_data in features_df.groupby('user_id'):
        # Sort by window_id if available, otherwise use index
        if 'window_id' in user_data.columns:
            user_data = user_data.sort_values('window_id')
        
        # Split based on time (first X% for training, rest for testing)
        n_train = int(len(user_data) * (1 - test_size))
        
        train_dfs.append(user_data.iloc[:n_train])
        test_dfs.append(user_data.iloc[n_train:])
    
    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    print(f"Split: {len(train_df)} training samples, {len(test_df)} test samples")
    
    return train_df, test_df


def prepare_features(features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare feature matrix and labels from features DataFrame.
    
    Args:
        features_df: DataFrame with extracted features
        
    Returns:
        X (features), y (labels)
    """
    # Identify feature columns (exclude metadata)
    metadata_cols = ['user_id', 'session', 'window_id', 'time_period', 'round', 'task']
    feature_cols = [col for col in features_df.columns if col not in metadata_cols]
    
    X = features_df[feature_cols].values
    y = features_df['user_id'].values
    
    print(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")
    
    return X, y
