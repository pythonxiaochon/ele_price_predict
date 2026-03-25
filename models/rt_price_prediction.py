"""
实时价格预测模块
Real-Time Price Prediction Module

Algorithm : LSTM (Long Short-Term Memory) or GRU (Gated Recurrent Unit) — PyTorch
Target    : Real-time clearing price for the next 15-min slot (multi-step optional)
Features  : Real-time unit output, tie-line power, day-ahead price, load, weather,
            plus the day-ahead / real-time spread forecast.
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    warnings.warn(
        "PyTorch is not installed. RTPricePredictor will raise ImportError on use."
    )

from utils.metrics import evaluate_all
from data.preprocess import add_time_features, add_lag_features, add_rolling_features


# ---------------------------------------------------------------------------
# Neural network architectures
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class _LSTMNet(nn.Module):
        """Stacked LSTM for sequence-to-one regression."""

        def __init__(
            self,
            input_size: int,
            hidden_size: int = 64,
            num_layers: int = 2,
            dropout: float = 0.2,
            output_size: int = 1,
        ) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, output_size),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])   # last time step

    class _GRUNet(nn.Module):
        """Stacked GRU for sequence-to-one regression."""

        def __init__(
            self,
            input_size: int,
            hidden_size: int = 64,
            num_layers: int = 2,
            dropout: float = 0.2,
            output_size: int = 1,
        ) -> None:
            super().__init__()
            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, output_size),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            out, _ = self.gru(x)
            return self.fc(out[:, -1, :])


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def _build_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build sliding-window sequences for LSTM/GRU input.

    Args:
        X: Feature matrix of shape (N, n_features).
        y: Target vector of shape (N,).
        seq_len: Number of time steps per sequence.

    Returns:
        Tuple (X_seq, y_seq) with shapes (N-seq_len, seq_len, n_features)
        and (N-seq_len,).
    """
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i - seq_len: i])
        y_seq.append(y[i])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


# ---------------------------------------------------------------------------
# Main predictor class
# ---------------------------------------------------------------------------

class RTPricePredictor:
    """Real-time clearing price predictor using LSTM or GRU (PyTorch).

    Args:
        model_type: ``'lstm'`` or ``'gru'``.
        seq_len: Number of 15-min steps used as the input sequence window.
        hidden_size: Number of hidden units in each recurrent layer.
        num_layers: Number of stacked recurrent layers.
        dropout: Dropout probability applied between layers.
        lr: Learning rate for the Adam optimiser.
        batch_size: Mini-batch size during training.
        epochs: Maximum number of training epochs.
        patience: Early-stopping patience (epochs without validation improvement).
        device: ``'cpu'``, ``'cuda'``, or ``'auto'`` to select automatically.
    """

    TARGET = "rt_price"
    FEATURE_COLS = [
        "da_price",
        "load_mw",
        "unit_output",
        "tie_line_mw",
        "temperature",
        "humidity",
        "irradiance",
        "time_slot",
        "hour",
        "is_workday",
        "is_holiday",
        "is_weekend",
    ]

    def __init__(
        self,
        model_type: str = "lstm",
        seq_len: int = 96,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-3,
        batch_size: int = 256,
        epochs: int = 50,
        patience: int = 10,
        device: str = "auto",
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for RTPricePredictor. "
                "Install it with: pip install torch"
            )
        self.model_type = model_type
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._model: Optional[nn.Module] = None
        self._feature_names: List[str] = []
        self._x_mean: Optional[np.ndarray] = None
        self._x_std: Optional[np.ndarray] = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        self.train_history: List[Dict] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_features(self, df: pd.DataFrame, spread_pred: Optional[pd.Series] = None) -> pd.DataFrame:
        """Engineer features from raw DataFrame.

        Optionally incorporates an externally provided spread forecast.
        """
        feat = add_time_features(df)
        feat = add_lag_features(feat, [self.TARGET, "da_price"], [1, 4, 96])
        feat = add_rolling_features(feat, [self.TARGET], [4, 96], stats=["mean", "std"])

        if spread_pred is not None:
            feat["spread_pred"] = spread_pred.reindex(feat.index)

        feat.dropna(inplace=True)
        return feat

    def _get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        base = [c for c in df.columns if c != self.TARGET]
        # Remove raw price columns that would leak future info
        leakage = {"rt_price"}
        return [c for c in base if c not in leakage]

    def _normalise(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, fit: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if fit:
            self._x_mean = X.mean(axis=0)
            self._x_std = X.std(axis=0) + 1e-9
            if y is not None:
                self._y_mean = float(y.mean())
                self._y_std = float(y.std()) + 1e-9
        X_norm = (X - self._x_mean) / self._x_std
        y_norm = ((y - self._y_mean) / self._y_std) if y is not None else None
        return X_norm, y_norm

    def _denormalise_y(self, y_norm: np.ndarray) -> np.ndarray:
        return y_norm * self._y_std + self._y_mean

    def _build_model(self, input_size: int) -> "nn.Module":
        cls = _LSTMNet if self.model_type == "lstm" else _GRUNet
        return cls(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        spread_pred: Optional[pd.Series] = None,
        val_fraction: float = 0.1,
    ) -> "RTPricePredictor":
        """Train the LSTM/GRU model on historical real-time price data.

        Args:
            df: DataFrame with DateTimeIndex (15-min) containing at least
                ``rt_price``, ``da_price``, ``unit_output``, ``tie_line_mw``,
                ``load_mw``, and weather columns.
            spread_pred: Optional Series of predicted day-ahead/RT spread values
                (output from SpreadAnalyzer.predict).
            val_fraction: Fraction of training data held out for early stopping.

        Returns:
            self
        """
        feat = self._build_features(df, spread_pred)
        feature_cols = self._get_feature_cols(feat)
        self._feature_names = feature_cols

        X_raw = feat[feature_cols].values.astype(np.float32)
        y_raw = feat[self.TARGET].values.astype(np.float32)

        X_norm, y_norm = self._normalise(X_raw, y_raw, fit=True)

        split = int(len(X_norm) * (1 - val_fraction))
        X_tr, X_val = X_norm[:split], X_norm[split:]
        y_tr, y_val = y_norm[:split], y_norm[split:]

        X_seq_tr, y_seq_tr = _build_sequences(X_tr, y_tr, self.seq_len)
        X_seq_val, y_seq_val = _build_sequences(X_val, y_val, self.seq_len)

        tr_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_seq_tr),
                torch.from_numpy(y_seq_tr).unsqueeze(-1),
            ),
            batch_size=self.batch_size,
            shuffle=False,
        )
        val_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_seq_val),
                torch.from_numpy(y_seq_val).unsqueeze(-1),
            ),
            batch_size=self.batch_size,
            shuffle=False,
        )

        self._model = self._build_model(X_seq_tr.shape[2])
        optimiser = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0
        self.train_history = []

        for epoch in range(1, self.epochs + 1):
            # ---- Train ----
            self._model.train()
            train_losses = []
            for X_batch, y_batch in tr_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimiser.zero_grad()
                pred = self._model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                optimiser.step()
                train_losses.append(loss.item())

            # ---- Validate ----
            self._model.eval()
            val_losses = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    pred = self._model(X_batch)
                    val_losses.append(criterion(pred, y_batch).item())

            train_loss = float(np.mean(train_losses))
            val_loss = float(np.mean(val_losses))
            self.train_history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

            # ---- Early stopping ----
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    self._model.load_state_dict(best_state)
                    break

        return self

    def predict(
        self,
        df: pd.DataFrame,
        spread_pred: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Predict real-time prices.

        Args:
            df: DataFrame with the same schema as training data.
            spread_pred: Optional spread forecast Series.

        Returns:
            Series of predicted real-time prices aligned to the input index.
        """
        if self._model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        feat = self._build_features(df, spread_pred)
        X_raw = feat[self._feature_names].values.astype(np.float32)
        X_norm, _ = self._normalise(X_raw)

        X_seq, _ = _build_sequences(X_norm, np.zeros(len(X_norm)), self.seq_len)
        seq_index = feat.index[self.seq_len:]

        self._model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X_seq).to(self.device)
            preds_norm = self._model(X_tensor).cpu().numpy().squeeze()

        preds = self._denormalise_y(preds_norm)
        return pd.Series(preds, index=seq_index, name="rt_price_pred")

    def evaluate(
        self,
        df: pd.DataFrame,
        spread_pred: Optional[pd.Series] = None,
    ) -> Dict:
        """Evaluate the model on a hold-out DataFrame.

        Args:
            df: DataFrame containing ground-truth ``rt_price`` values.
            spread_pred: Optional spread forecast Series.

        Returns:
            Dict of evaluation metrics (mape, smape, rmse, mae, r2).
        """
        preds = self.predict(df, spread_pred)
        feat = self._build_features(df, spread_pred)
        # Align index: only overlap between predictions and available ground truth
        common_idx = preds.index.intersection(feat.index)
        y_true = feat.loc[common_idx, self.TARGET].values
        y_pred = preds.loc[common_idx].values
        return evaluate_all(y_true, y_pred)
