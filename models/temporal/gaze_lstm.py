"""
Gaze LSTM Model

LSTM for gaze-based continuous authentication using temporal sequences.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional


class GazeLSTM(nn.Module):
    """
    LSTM for gaze authentication using temporal feature sequences.
    
    Architecture:
    - Input: (batch, seq_len, features)
    - Bidirectional LSTM layers
    - Attention mechanism (optional)
    - Fully connected layers for classification
    """
    
    def __init__(self, input_features: int, num_classes: int, 
                 hidden_size: int = 128, num_layers: int = 2, 
                 dropout: float = 0.3):
        """
        Args:
            input_features: Number of features per time step
            num_classes: Number of users to classify
            hidden_size: Size of LSTM hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(GazeLSTM, self).__init__()
        
        self.input_features = input_features
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_features,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention layer
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)
        
    def attention_net(self, lstm_output):
        """
        Apply attention mechanism to LSTM outputs.
        
        Args:
            lstm_output: (batch, seq_len, hidden_size*2)
        
        Returns:
            context: (batch, hidden_size*2)
        """
        # Calculate attention weights
        attn_weights = torch.tanh(lstm_output)
        attn_weights = self.attention(attn_weights)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # Apply attention
        context = torch.sum(attn_weights * lstm_output, dim=1)  # (batch, hidden_size*2)
        
        return context
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (batch_size, seq_length, input_features)
        
        Returns:
            logits: (batch_size, num_classes)
        """
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)  # (batch, seq, hidden*2)
        
        # Apply attention
        context = self.attention_net(lstm_out)
        
        # Fully connected
        out = self.fc1(context)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class GazeLSTMClassifier:
    """Wrapper for GazeLSTM with sklearn-like interface."""
    
    def __init__(self, seq_length: int = 10, hidden_size: int = 128,
                 num_layers: int = 2, epochs: int = 50, 
                 batch_size: int = 32, learning_rate: float = 0.001,
                 device: str = 'cpu'):
        """
        Args:
            seq_length: Number of consecutive windows per sequence
            hidden_size: LSTM hidden state size
            num_layers: Number of LSTM layers
            epochs: Training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            device: 'cpu' or 'cuda'
        """
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None
        self.classes_ = None
        
    def _create_sequences(self, X, y):
        """Create sequences from windowed features (no padding)."""
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
        # Ensure a numeric, contiguous float32 array to avoid object dtype downstream
        if len(sequences) == 0:
            return np.empty((0, self.seq_length, X.shape[1]), dtype=np.float32), np.array(labels)
        return np.asarray(sequences, dtype=np.float32), np.asarray(labels)

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
        return np.asarray(sequences, dtype=np.float32), np.asarray(labels)
    
    def _sanitize(self, X: np.ndarray) -> np.ndarray:
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        try:
            p1 = np.nanpercentile(X, 1, axis=0)
            p99 = np.nanpercentile(X, 99, axis=0)
            X = np.clip(X, p1, p99)
        except Exception:
            pass
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
        phase = "fine-tune" if continue_training and self.model is not None else "initial"
        print(f"Initialized LSTM classifier ({phase}, seq_length={self.seq_length}, hidden={self.hidden_size}, layers={self.num_layers})")
        if not continue_training or self.model is None:
            X_clean = self._sanitize(X)
            X_scaled = self._scale(X_clean)
        else:
            X_scaled = self._apply_scale(self._sanitize(X))
        if continue_training:
            X_seq, y_seq = self._create_sequences_with_padding(X_scaled, y)
        else:
            X_seq, y_seq = self._create_sequences(X_scaled, y)
        if len(X_seq) == 0:
            if continue_training:
                print(f"⚠️ Fine-tune skipped: no sequences (seq_length={self.seq_length}).")
                return
            raise ValueError(f"Not enough data to create sequences (need >= {self.seq_length})")
        print(f"Training LSTM on {len(X_seq)} sequences...")
        if self.classes_ is None or not continue_training:
            self.classes_ = np.unique(y_seq)
        num_classes = len(self.classes_)
        label_to_idx = {label: idx for idx, label in enumerate(self.classes_)}
        y_idx = np.array([label_to_idx[label] for label in y_seq])
        input_features = X_seq.shape[2]
        if self.model is None:
            self.model = GazeLSTM(input_features, num_classes, hidden_size=self.hidden_size, num_layers=self.num_layers)
        self.model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # Enforce float32 dtype for stable tensor conversion
        X_seq = X_seq.astype(np.float32, copy=False)
        y_idx = y_idx.astype(np.int64, copy=False)
        dataset = TensorDataset(torch.from_numpy(X_seq), torch.from_numpy(y_idx))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model.train()
        final_correct = 0
        final_total = 0
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
            # accumulate for final accuracy over all epochs (last epoch sufficient but keep running tally)
            final_correct = correct
            final_total = total
            if total == 0:
                print("  ⚠️ No valid batches this epoch")
                continue
            if (epoch + 1) % max(1, self.epochs // 5) == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{self.epochs}: Loss={epoch_loss/total:.4f}, Acc={correct/total:.4f}")
        train_acc = final_correct / final_total if final_total > 0 else 0.0
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
            X_tensor = torch.from_numpy(np.asarray(X_seq, dtype=np.float32)).to(self.device)
            outputs = self.model(X_tensor)
            _, predicted_idx = outputs.max(1)
        predictions = self.classes_[predicted_idx.cpu().numpy()]
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
            X_tensor = torch.from_numpy(np.asarray(X_seq, dtype=np.float32)).to(self.device)
            outputs = self.model(X_tensor)
            probas_seq = torch.softmax(outputs, dim=1).cpu().numpy()
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
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'input_features': self.model.input_features,
            'num_classes': self.model.num_classes
        }
        torch.save(state, path)
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu'):
        """Load model from disk."""
        state = torch.load(path, map_location=device)
        
        classifier = cls(
            seq_length=state['seq_length'],
            hidden_size=state['hidden_size'],
            num_layers=state['num_layers'],
            device=device
        )
        classifier.scaler_mean = state['scaler_mean']
        classifier.scaler_std = state['scaler_std']
        classifier.classes_ = state['classes_']
        
        classifier.model = GazeLSTM(
            state['input_features'],
            state['num_classes'],
            hidden_size=state['hidden_size'],
            num_layers=state['num_layers']
        )
        classifier.model.load_state_dict(state['model_state'])
        classifier.model.to(device)
        classifier.model.eval()
        
        return classifier
