"""
Gaze LSTM Model

LSTM for gaze-based continuous authentication using temporal sequences.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
        """Create sequences from windowed features."""
        sequences = []
        labels = []
        
        # Group by user
        unique_users = np.unique(y)
        for user in unique_users:
            user_mask = (y == user)
            user_X = X[user_mask]
            
            # Create sequences
            for i in range(len(user_X) - self.seq_length + 1):
                seq = user_X[i:i + self.seq_length]
                sequences.append(seq)
                labels.append(user)
        
        return np.array(sequences), np.array(labels)
    
    def train(self, X, y):
        """Train the LSTM model."""
        print(f"Initialized LSTM classifier (seq_length={self.seq_length}, hidden={self.hidden_size}, layers={self.num_layers})")
        
        # Handle NaN
        X = np.nan_to_num(X, nan=0.0)
        
        # Scale features
        self.scaler_mean = np.mean(X, axis=0)
        self.scaler_std = np.std(X, axis=0)
        self.scaler_std[self.scaler_std == 0] = 1
        X_scaled = (X - self.scaler_mean) / self.scaler_std
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y)
        
        if len(X_seq) == 0:
            raise ValueError(f"Not enough data to create sequences (need at least {self.seq_length} samples)")
        
        print(f"Training LSTM on {len(X_seq)} sequences...")
        
        # Get classes
        self.classes_ = np.unique(y_seq)
        num_classes = len(self.classes_)
        
        # Map labels to indices
        label_to_idx = {label: idx for idx, label in enumerate(self.classes_)}
        y_idx = np.array([label_to_idx[label] for label in y_seq])
        
        # Create model
        input_features = X_seq.shape[2]
        self.model = GazeLSTM(
            input_features, 
            num_classes, 
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        )
        self.model.to(self.device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.LongTensor(y_idx).to(self.device)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == y_tensor).float().mean().item()
                print(f"  Epoch {epoch+1}/{self.epochs}: Loss={loss.item():.4f}, Acc={accuracy:.4f}")
        
        # Final accuracy
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y_tensor).float().mean().item()
        
        print(f"✅ Training complete! Training accuracy: {accuracy:.4f}")
        
    def predict(self, X):
        """Predict class labels."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Handle NaN and scale
        X = np.nan_to_num(X, nan=0.0)
        X_scaled = (X - self.scaler_mean) / self.scaler_std
        
        # Create sequences
        X_seq, _ = self._create_sequences(X_scaled, np.zeros(len(X)))
        
        if len(X_seq) == 0:
            # Not enough data, return first class
            return np.array([self.classes_[0]] * len(X))
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            outputs = self.model(X_tensor)
            _, predicted_idx = torch.max(outputs, 1)
            predicted_idx = predicted_idx.cpu().numpy()
        
        # Map back to original labels
        predictions = self.classes_[predicted_idx]
        
        # Expand predictions to match input length
        full_predictions = np.full(len(X), predictions[0])
        for i in range(len(predictions)):
            start_idx = i
            end_idx = min(i + self.seq_length, len(X))
            full_predictions[start_idx:end_idx] = predictions[i]
        
        return full_predictions
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Handle NaN and scale
        X = np.nan_to_num(X, nan=0.0)
        X_scaled = (X - self.scaler_mean) / self.scaler_std
        
        # Create sequences
        X_seq, _ = self._create_sequences(X_scaled, np.zeros(len(X)))
        
        if len(X_seq) == 0:
            # Not enough data, return uniform probabilities
            proba = np.ones((len(X), len(self.classes_))) / len(self.classes_)
            return proba
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            outputs = self.model(X_tensor)
            probas = torch.softmax(outputs, dim=1).cpu().numpy()
        
        # Expand probabilities to match input length
        full_probas = np.zeros((len(X), len(self.classes_)))
        for i in range(len(X)):
            seq_idx = min(i, len(probas) - 1)
            full_probas[i] = probas[seq_idx]
        
        return full_probas
    
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
