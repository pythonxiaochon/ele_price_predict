"""
日前—实时价差分析模块
Day-Ahead / Real-Time Spread Analysis Module

Algorithm : Random Forest Regressor (sklearn)
Purpose   : Analyse and predict the spread between day-ahead bidding prices
            and real-time clearing prices; identify abnormal arbitrage windows.

Key outputs:
    - Spread statistics (mean, std, skew, kurtosis, percentiles)
    - Arbitrage window flags (spread outside ±1.5 σ)
    - Trained Random Forest that predicts the spread for each 15-min slot
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from utils.metrics import evaluate_all
from data.preprocess import add_time_features, add_lag_features, add_rolling_features


class SpreadAnalyzer:
    """Day-ahead vs. real-time price spread analyser and predictor.

    Columns required in the input DataFrame:
        da_price   : Day-ahead clearing price (¥/MWh).
        rt_price   : Real-time clearing price (¥/MWh).
        load_mw    : Grid load (MW) — used as a feature.
        temperature: Ambient temperature (°C).

    Args:
        model_type: ``'rf'`` (Random Forest) or ``'ridge'`` (Ridge regression).
        n_estimators: Number of trees when using Random Forest.
        arbitrage_threshold_sigma: Spread magnitude in standard deviations above
            which a time slot is flagged as an abnormal arbitrage window.
        random_state: Random seed for reproducibility.
    """

    SPREAD_COL = "spread"
    DA_COL = "da_price"
    RT_COL = "rt_price"

    def __init__(
        self,
        model_type: str = "rf",
        n_estimators: int = 200,
        arbitrage_threshold_sigma: float = 1.5,
        random_state: int = 42,
    ) -> None:
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.arbitrage_threshold_sigma = arbitrage_threshold_sigma
        self.random_state = random_state

        self._model = None
        self._scaler = StandardScaler()
        self._feature_names: List[str] = []
        self._spread_stats: Dict = {}

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compute_spread(df: pd.DataFrame) -> pd.Series:
        """Compute the price spread: rt_price − da_price.

        Args:
            df: DataFrame containing ``da_price`` and ``rt_price`` columns.

        Returns:
            Series named ``'spread'``.
        """
        return (df["rt_price"] - df["da_price"]).rename("spread")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_spread(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[self.SPREAD_COL] = self.compute_spread(out)
        return out

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feat = self._add_spread(df)
        feat = add_time_features(feat)
        feat = add_lag_features(feat, [self.SPREAD_COL, self.DA_COL], [4, 96])
        feat = add_rolling_features(
            feat,
            [self.SPREAD_COL],
            windows=[4, 96],
            stats=["mean", "std"],
        )
        feat.dropna(inplace=True)
        return feat

    def _get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        exclude = {self.SPREAD_COL, self.RT_COL}
        return [c for c in df.columns if c not in exclude]

    # ------------------------------------------------------------------
    # Statistics & arbitrage detection
    # ------------------------------------------------------------------

    def compute_statistics(self, df: pd.DataFrame) -> Dict:
        """Compute spread statistics over the full DataFrame.

        Args:
            df: DataFrame with ``da_price`` and ``rt_price`` columns.

        Returns:
            Dict with keys: mean, std, median, skewness, kurtosis, q5, q25,
            q75, q95, positive_fraction, negative_fraction.
        """
        spread = self.compute_spread(df)
        stats = {
            "mean": float(spread.mean()),
            "std": float(spread.std()),
            "median": float(spread.median()),
            "skewness": float(spread.skew()),
            "kurtosis": float(spread.kurt()),
            "q5": float(spread.quantile(0.05)),
            "q25": float(spread.quantile(0.25)),
            "q75": float(spread.quantile(0.75)),
            "q95": float(spread.quantile(0.95)),
            "positive_fraction": float((spread > 0).mean()),
            "negative_fraction": float((spread < 0).mean()),
        }
        self._spread_stats = stats
        return stats

    def identify_arbitrage_windows(
        self,
        df: pd.DataFrame,
        threshold_sigma: Optional[float] = None,
    ) -> pd.DataFrame:
        """Flag time slots where the price spread is abnormally large.

        A slot is flagged as an arbitrage opportunity when:
            |spread| > mean + threshold_sigma * std

        Args:
            df: DataFrame with ``da_price`` and ``rt_price`` columns.
            threshold_sigma: Override the instance-level threshold.

        Returns:
            DataFrame copy with additional columns:
                ``spread``          : Raw price spread.
                ``is_arbitrage``    : Boolean flag.
                ``spread_zscore``   : Standardised spread value.
        """
        sigma = threshold_sigma if threshold_sigma is not None else self.arbitrage_threshold_sigma

        out = df.copy()
        out[self.SPREAD_COL] = self.compute_spread(out)

        stats = self.compute_statistics(df)
        mean_s, std_s = stats["mean"], stats["std"]

        out["spread_zscore"] = (out[self.SPREAD_COL] - mean_s) / (std_s + 1e-9)
        out["is_arbitrage"] = out["spread_zscore"].abs() > sigma
        return out

    def hourly_spread_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the average spread profile by time-slot (0–95).

        Args:
            df: DataFrame with ``da_price`` and ``rt_price`` columns and
                a 15-min DateTimeIndex.

        Returns:
            DataFrame indexed by time_slot (0–95) with columns:
                mean_spread, std_spread, q25_spread, q75_spread.
        """
        out = df.copy()
        out[self.SPREAD_COL] = self.compute_spread(out)
        out["time_slot"] = out.index.hour * 4 + out.index.minute // 15

        profile = (
            out.groupby("time_slot")[self.SPREAD_COL]
            .agg(
                mean_spread="mean",
                std_spread="std",
                q25_spread=lambda x: x.quantile(0.25),
                q75_spread=lambda x: x.quantile(0.75),
            )
        )
        return profile

    # ------------------------------------------------------------------
    # Model training & prediction
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "SpreadAnalyzer":
        """Fit the spread prediction model.

        Args:
            df: DataFrame with DateTimeIndex (15-min) containing at least
                ``da_price``, ``rt_price``, and ``load_mw``.

        Returns:
            self
        """
        feat = self._build_features(df)
        feature_cols = self._get_feature_cols(feat)
        self._feature_names = feature_cols

        X = self._scaler.fit_transform(feat[feature_cols].values)
        y = feat[self.SPREAD_COL].values

        if self.model_type == "rf":
            self._model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=10,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=self.random_state,
            )
        else:
            self._model = Ridge(alpha=1.0)

        self._model.fit(X, y)
        self.compute_statistics(df)
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Predict the day-ahead/real-time price spread.

        Args:
            df: DataFrame with the same schema as training data.

        Returns:
            Series of predicted spread values aligned to the input index.
        """
        if self._model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        feat = self._build_features(df)
        X = self._scaler.transform(feat[self._feature_names].values)
        preds = self._model.predict(X)
        return pd.Series(preds, index=feat.index, name="spread_pred")

    def cross_validate(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
    ) -> List[Dict]:
        """Time-series cross-validation using sklearn's TimeSeriesSplit.

        Args:
            df: Full DataFrame with DateTimeIndex (15-min).
            n_splits: Number of CV folds.

        Returns:
            List of metric dicts (one per fold).
        """
        feat = self._build_features(df)
        feature_cols = self._get_feature_cols(feat)

        X = self._scaler.fit_transform(feat[feature_cols].values)
        y = feat[self.SPREAD_COL].values

        tscv = TimeSeriesSplit(n_splits=n_splits)
        results = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            if self.model_type == "rf":
                model = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=10,
                    min_samples_leaf=5,
                    n_jobs=-1,
                    random_state=self.random_state,
                )
            else:
                model = Ridge(alpha=1.0)

            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)
            metrics = evaluate_all(y_te, preds)
            metrics["fold"] = fold + 1
            results.append(metrics)

        if not results:
            raise ValueError("No CV folds were completed. Check that the dataset is large enough.")
        self._model = model  # noqa: F821
        self._feature_names = feature_cols
        return results

    def feature_importance(self) -> Optional[pd.DataFrame]:
        """Return feature importances (only available for Random Forest).

        Returns:
            DataFrame with columns ['feature', 'importance'], or None for Ridge.
        """
        if self._model is None:
            raise RuntimeError("Model is not fitted.")
        if not hasattr(self._model, "feature_importances_"):
            warnings.warn("Feature importance is not available for this model type.")
            return None
        imp = self._model.feature_importances_
        return (
            pd.DataFrame({"feature": self._feature_names, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
