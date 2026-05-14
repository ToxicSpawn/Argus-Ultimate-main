"""Additional ML Architectures for Trading.

CNN1D, Attention, GNN, Transformer-XL, etc.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Optional, Any

logger = logging.getLogger(__name__)

PYTORCH_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    nn = object


class CNN1DPricePredictor(nn.Module):
    """1D CNN for sequential price data."""

    def __init__(
        self,
        input_channels: int = 10,
        sequence_length: int = 100,
        filters: list = [32, 64, 128],
        kernel_size: int = 3,
        output_size: int = 1,
    ):
        super().__init__()

        layers = []
        in_channels = input_channels

        for i, num_filters in enumerate(filters):
            layers.append(nn.Conv1d(
                in_channels,
                num_filters,
                kernel_size,
                padding=kernel_size // 2
            ))
            layers.append(nn.BatchNorm1d(num_filters))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(2))
            layers.append(nn.Dropout(0.2))
            in_channels = num_filters

        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(filters[-1] * (sequence_length // 2**len(filters)), output_size)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self._device)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self._device)
            predictions = self.forward(X_tensor)
            return predictions.cpu().numpy()

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
    ) -> dict:
        self.train()

        X_tensor = torch.FloatTensor(X_train).to(self._device)
        y_tensor = torch.FloatTensor(y_train).to(self._device)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.forward(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            losses.append(epoch_loss / len(dataloader))

            if epoch % 20 == 0:
                logger.info(f"CNN Epoch {epoch}: loss={losses[-1]:.4f}")

        return {"loss": losses}


class AttentionPricePredictor(nn.Module):
    """Self-Attention for price sequences."""

    def __init__(
        self,
        input_dim: int = 10,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        output_size: int = 1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.attention_weights = None

        self.fc = nn.Linear(d_model, output_size)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self._device)

    def forward(self, x, return_attention=False):
        x = self.input_proj(x)

        if return_attention:
            attention_weights = []
            for layer in self.transformer.layers:
                attn_output, attn_weights = layer.self_attn(x, x, x, need_weights=True)
                attention_weights.append(attn_weights)
            self.attention_weights = attention_weights

        x = self.transformer(x)
        x = x[:, -1, :]
        return self.fc(x)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self._device)
            predictions = self.forward(X_tensor)
            return predictions.cpu().numpy()


class TemporalCNN(nn.Module):
    """Temporal Convolutional Network (TCN)."""

    def __init__(
        self,
        input_channels: int = 10,
        num_channels: list = [32, 64, 128],
        kernel_size: int = 3,
        output_size: int = 1,
    ):
        super().__init__()

        layers = []
        in_channels = input_channels

        for i, out_channels in enumerate(num_channels):
            layers.append(nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=kernel_size - 1
            ))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))

            if i < len(num_channels) - 1:
                layers.append(nn.MaxPool1d(2))

            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_size)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self._device)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv_layers(x)
        x = x.mean(dim=2)
        return self.fc(x)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self._device)
            predictions = self.forward(X_tensor)
            return predictions.cpu().numpy()


class GraphNeuralNetwork(nn.Module):
    """Graph Neural Network for market regime prediction."""

    def __init__(
        self,
        num_nodes: int = 10,
        node_features: int = 5,
        hidden_dim: int = 64,
        output_size: int = 1,
    ):
        super().__init__()

        self.node_encoder = nn.Linear(node_features, hidden_dim)

        self.message_passing_layers = nn.ModuleList([
            nn.Linear(hidden_dim * 2, hidden_dim)
            for _ in range(3)
        ])

        self.output = nn.Linear(hidden_dim, output_size)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self._device)

    def forward(self, x, edge_index):
        x = self.node_encoder(x)

        for layer in self.message_passing_layers:
            source = x[edge_index[0]]
            target = x[edge_index[1]]

            messages = layer(torch.cat([source, target], dim=-1))

            x = x + torch.scatter_add(
                torch.zeros_like(x),
                1,
                edge_index[1],
                messages
            )

        return self.output(x.mean(dim=0, keepdim=True))

    def predict(self, node_features: np.ndarray, edges: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(node_features).to(self._device)
            edge_index = torch.LongTensor(edges.T).to(self._device)
            predictions = self.forward(x, edge_index)
            return predictions.cpu().numpy()


class TransformerXL(nn.Module):
    """Transformer-XL for longer sequences."""

    def __init__(
        self,
        mem_len: int = 128,
        d_model: int = 64,
        n_head: int = 4,
        n_layer: int = 3,
        d_head: int = 64,
        d_inner: int = 256,
        output_size: int = 1,
    ):
        super().__init__()

        self.mem_len = mem_len
        self.d_model = d_model

        self.input_proj = nn.Linear(1, d_model)

        self.pos_emb = nn.Embedding(1000, d_model)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                d_ff=d_inner,
                batch_first=True,
            )
            for _ in range(n_layer)
        ])

        self.output = nn.Linear(d_model, output_size)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = None
        self.to(self._device)

    def forward(self, x, memory=None):
        batch_size, seq_len, feature_dim = x.shape

        x = x.mean(dim=-1, keepdim=True)
        x = self.input_proj(x)

        position = torch.arange(seq_len, device=self._device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.pos_emb(position)

        if memory is not None:
            x = torch.cat([memory, x], dim=1)

        new_memory = x[:, -self.mem_len:, :]

        for layer in self.layers:
            x = layer(x)

        return self.output(x[:, -1, :]), new_memory

    def predict(self, X: np.ndarray, memory: np.ndarray = None) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self._device)
            mem_tensor = torch.FloatTensor(memory).to(self._device) if memory is not None else None
            predictions, new_mem = self.forward(X_tensor, mem_tensor)
            return predictions.cpu().numpy()


def create_model(model_type: str, **kwargs) -> nn.Module:
    """Factory function to create ML models."""

    models = {
        "cnn1d": CNN1DPricePredictor,
        "attention": AttentionPricePredictor,
        "tcn": TemporalCNN,
        "transformer_xl": TransformerXL,
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")

    return models[model_type](**kwargs)


def get_available_models() -> dict:
    return {
        "available": PYTORCH_AVAILABLE,
        "models": [
            "CNN1D - 1D Convolutional Network",
            "Attention - Self-Attention Mechanism",
            "TCN - Temporal Convolutional Network",
            "TransformerXL - Extended Memory Transformer",
            "GraphNets - (when edge data available)",
        ]
    }