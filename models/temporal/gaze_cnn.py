"""
Gaze CNN Model

1D CNN for gaze-based continuous authentication using temporal patterns.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Tuple, Optional
import math


class GazeCNN(nn.Module):
    """
    1D CNN for gaze authentication using temporal feature sequences.
    
    Architecture:
    - Input: (batch, seq_len, features)
    - Conv1D layers to extract temporal patterns
    - Global pooling
    - Fully connected layers for classification
    """
    
    def __init__(self, input_features: int, num_classes: int, seq_length: int = 10):
        """
        Args:
            input_features: Number of features per time step (e.g., 43)
            num_classes: Number of users to classify
            seq_length: Length of input sequences
        """
        super(GazeCNN, self).__init__()
        
        self.input_features = input_features
        self.num_classes = num_classes
        self.seq_length = seq_length
        
        # 1D Convolutional layers
        # Input: (batch, features, seq_len)
        self.conv1 = nn.Conv1d(input_features, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.3)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.3)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu3 = nn.ReLU()
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.fc_relu = nn.ReLU()
        self.fc_dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (batch_size, seq_length, input_features)
        
        Returns:
            logits: (batch_size, num_classes)
        """
        # Transpose to (batch, features, seq_len) for Conv1d
        x = x.transpose(1, 2)
        
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.squeeze(-1)  # (batch, 256)
        
        # Fully connected
        x = self.fc1(x)
        x = self.fc_relu(x)
        x = self.fc_dropout(x)
        x = self.fc2(x)
        
        return x


class GazeCNNClassifier:
    """Wrapper for GazeCNN with sklearn-like interface."""
    
    def __init__(self, seq_length: int = 10, epochs: int = 50, 
                 batch_size: int = 32, learning_rate: float = 0.001,
                 device: str = 'cpu'):
        """
        Args:
            seq_length: Number of consecutive windows per sequence
            epochs: Training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            device: 'cpu' or 'cuda'
        """
        self.seq_length = seq_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None
        self.classes_ = None
        
    def _create_sequences(self, X, y):
        """Create sequences from windowed features.

        For initial training: standard sliding window if enough windows.
        For fine-tune (when called with continue_training=True): if a user has fewer than seq_length
        windows, create a single padded sequence (repeat last window) to retain that user's contribution.
        """
        sequences = []
        labels = []
        unique_users = np.unique(y)
        for user in unique_users:
            user_mask = (y == user)
            user_X = X[user_mask]
            n = len(user_X)
            if n >= self.seq_length:
                for i in range(n - self.seq_length + 1):
                    seq = user_X[i:i + self.seq_length]
                    sequences.append(seq)
                    labels.append(user)
            else:
                # padding path only used in fine-tune; decided by caller removing sequences if not allowed
                pass
        if len(sequences) == 0:
            return np.empty((0, self.seq_length, X.shape[1]), dtype=np.float32), np.array(labels)
        seq_arr = np.asarray(sequences, dtype=np.float32)
        lab_arr = np.asarray(labels)
        return seq_arr, lab_arr

    def _create_sequences_with_padding(self, X, y):
        sequences = []
        labels = []
        unique_users = np.unique(y)
        for user in unique_users:
            user_mask = (y == user)
            user_X = X[user_mask]
            n = len(user_X)
            if n >= self.seq_length:
                for i in range(n - self.seq_length + 1):
                    seq = user_X[i:i + self.seq_length]
                    sequences.append(seq)
                    labels.append(user)
            elif n > 0:
                pad_needed = self.seq_length - n
                pad_block = np.repeat(user_X[-1][None, :], pad_needed, axis=0)
                seq = np.concatenate([user_X, pad_block], axis=0)
                sequences.append(seq)
                labels.append(user)
        if len(sequences) == 0:
            return np.empty((0, self.seq_length, X.shape[1]), dtype=np.float32), np.array(labels)
        seq_arr = np.asarray(sequences, dtype=np.float32)
        lab_arr = np.asarray(labels)
        return seq_arr, lab_arr
    
    def _sanitize(self, X: np.ndarray) -> np.ndarray:
        """Replace NaN/inf and clip extreme values via percentiles."""
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        # Clip each feature to 1st-99th percentile using nanpercentile for robustness
        try:
            p1 = np.nanpercentile(X, 1, axis=0)
            p99 = np.nanpercentile(X, 99, axis=0)
            X = np.clip(X, p1, p99)
        except Exception:
            pass
        # enforce finite
        X[~np.isfinite(X)] = 0.0
        return X

    def _scale(self, X: np.ndarray) -> np.ndarray:
        self.scaler_mean = X.mean(axis=0)
        var = ((X - self.scaler_mean) ** 2).mean(axis=0)
        self.scaler_std = np.sqrt(var + 1e-8)
        scaled = (X - self.scaler_mean) / self.scaler_std
        scaled[~np.isfinite(scaled)] = 0.0
        return scaled.astype(np.float32, copy=False)

    def _apply_scale(self, X: np.ndarray) -> np.ndarray:
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        scaled = (X - self.scaler_mean) / self.scaler_std
        scaled[~np.isfinite(scaled)] = 0.0
        return scaled.astype(np.float32, copy=False)

    def train(self, X, y, continue_training: bool = False):
        """Train the CNN model. If continue_training=True, reuse existing model weights for fine-tuning."""
        phase = "fine-tune" if continue_training and self.model is not None else "initial"
        print(f"Initialized CNN classifier ({phase}, seq_length={self.seq_length}, epochs={self.epochs})")

        # Scale or apply existing scaler
        if not continue_training or self.model is None:
            X_clean = self._sanitize(X)
            X_scaled = self._scale(X_clean)
        else:
            X_scaled = self._apply_scale(self._sanitize(X))

        # Create sequences (with padding if fine-tuning)
        if continue_training:
            X_seq, y_seq = self._create_sequences_with_padding(X_scaled, y)
        else:
            X_seq, y_seq = self._create_sequences(X_scaled, y)
        if len(X_seq) == 0:
            if continue_training:
                print(f"⚠️ Fine-tune skipped: no sequences could be formed (seq_length={self.seq_length}).")
                return
            else:
                raise ValueError(f"Not enough data to create sequences (need >= {self.seq_length})")
        print(f"Training CNN on {len(X_seq)} sequences...")

        # Class management
        if self.classes_ is None:
            self.classes_ = np.unique(y_seq)
        else:
            if continue_training:
                mask = np.isin(y_seq, self.classes_)
                if not np.all(mask):
                    y_seq = y_seq[mask]
                    X_seq = X_seq[mask]
        num_classes = len(self.classes_)
        label_to_idx = {label: idx for idx, label in enumerate(self.classes_)}
        y_idx = np.array([label_to_idx[label] for label in y_seq])

        # Init / reuse model
        input_features = X_seq.shape[2]
        if self.model is None:
            self.model = GazeCNN(input_features, num_classes, self.seq_length)
        self.model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        dataset = TensorDataset(torch.FloatTensor(X_seq), torch.LongTensor(y_idx))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                out = self.model(xb)
                loss = criterion(out, yb)
                if torch.isnan(loss):
                    print("  ⚠️ NaN loss encountered; skipping batch")
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
                _, pred = out.max(1)
                correct += (pred == yb).sum().item()
                total += xb.size(0)
            if total == 0:
                print("  ⚠️ No valid batches this epoch")
                continue
            if (epoch + 1) % max(1, self.epochs // 5) == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{self.epochs}: Loss={epoch_loss/total:.4f}, Acc={correct/total:.4f}")

        # Final training accuracy
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                out = self.model(xb)
                _, pred = out.max(1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
            train_acc = correct / max(1, total)
        print(f"✅ Training complete! Training accuracy: {train_acc:.4f}")
        
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained yet")
        X_scaled = self._apply_scale(self._sanitize(X))
        X_seq, _ = self._create_sequences(X_scaled, np.zeros(len(X)))
        if len(X_seq) == 0:
            return np.array([self.classes_[0]] * len(X))
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.as_tensor(X_seq, dtype=torch.float32, device=self.device)
            outputs = self.model(X_tensor)
            _, predicted_idx = outputs.max(1)
        predictions = self.classes_[predicted_idx.cpu().numpy()]
        # Majority vote per window across sequences that include it
        votes = [[] for _ in range(len(X))]
        for i in range(len(predictions)):
            for w in range(i, i + self.seq_length):
                if w < len(X):
                    votes[w].append(predictions[i])
        final = []
        for v in votes:
            if not v:
                final.append(self.classes_[0])
            else:
                vals, counts = np.unique(v, return_counts=True)
                final.append(vals[np.argmax(counts)])
        return np.array(final)
    
    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model not trained yet")
        X_scaled = self._apply_scale(self._sanitize(X))
        X_seq, _ = self._create_sequences(X_scaled, np.zeros(len(X)))
        if len(X_seq) == 0:
            return np.ones((len(X), len(self.classes_))) / len(self.classes_)
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.as_tensor(X_seq, dtype=torch.float32, device=self.device)
            outputs = self.model(X_tensor)
            probas_seq = torch.softmax(outputs, dim=1).cpu().numpy()
        # Aggregate probabilities via average vote per window
        agg = np.zeros((len(X), len(self.classes_)))
        counts = np.zeros(len(X))
        for i in range(len(probas_seq)):
            for w in range(i, i + self.seq_length):
                if w < len(X):
                    agg[w] += probas_seq[i]
                    counts[w] += 1
        counts[counts == 0] = 1
        agg /= counts[:, None]
        return agg
    
    def save(self, path: str):
        """Save model to disk."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        state = {
            'model_state': self.model.state_dict(),
            'scaler_mean': self.scaler_mean,
            'scaler_std': self.scaler_std,
            'classes_': self.classes_,
            'seq_length': self.seq_length,
            'input_features': self.model.input_features,
            'num_classes': self.model.num_classes
        }
        torch.save(state, path)
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu'):
        """Load model from disk."""
        state = torch.load(path, map_location=device)
        
        classifier = cls(seq_length=state['seq_length'], device=device)
        classifier.scaler_mean = state['scaler_mean']
        classifier.scaler_std = state['scaler_std']
        classifier.classes_ = state['classes_']
        
        classifier.model = GazeCNN(
            state['input_features'],
            state['num_classes'],
            state['seq_length']
        )
        classifier.model.load_state_dict(state['model_state'])
        classifier.model.to(device)
        classifier.model.eval()
        
        return classifier
