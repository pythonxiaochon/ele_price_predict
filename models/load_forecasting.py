"""
负荷预测模块
Load Forecasting Module

Algorithm : LightGBM (with XGBoost as optional fallback)
Target    : Day-ahead 96-point (15-min interval) load curve
Validation: Sliding-window (walk-forward) cross-validation
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    _LGBM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LGBM_AVAILABLE = False
    warnings.warn("lightgbm not installed. Falling back to XGBoost.")

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _XGB_AVAILABLE = False

from utils.metrics import evaluate_all
from data.preprocess import add_time_features, add_lag_features, add_rolling_features

# Default LightGBM hyper-parameters
_DEFAULT_LGBM_PARAMS: Dict = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "n_estimators": 500,
    "early_stopping_rounds": 30,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}

# Default XGBoost hyper-parameters (fallback)
_DEFAULT_XGB_PARAMS: Dict = {
    "objective": "reg:squarederror",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "early_stopping_rounds": 30,
    "verbosity": 0,
    "random_state": 42,
    "n_jobs": -1,
}


class LoadForecaster:
    """Day-ahead load forecaster using LightGBM (or XGBoost as fallback).

    Workflow:
        1. Call ``fit(df)`` with a raw DataFrame containing at least a
           ``load_mw`` column and weather columns.
        2. Call ``predict(df)`` to generate a 96-point day-ahead load curve.
        3. Call ``cross_validate(df)`` for sliding-window CV evaluation.

    Args:
        model_type: ``'lgbm'`` or ``'xgb'``. Auto-selects based on availability.
        params: Model hyper-parameters dict (merged with defaults).
        lag_steps: Lag periods (in 15-min steps) to create as features.
        roll_windows: Rolling window sizes (in 15-min steps) for statistical features.
        weather_cols: Weather column names present in the input DataFrame.
    """

    TARGET = "load_mw"
    DEFAULT_WEATHER = ["temperature", "humidity", "irradiance"]

    def __init__(
        self,
        model_type: str = "lgbm",
        params: Optional[Dict] = None,
        lag_steps: Optional[List[int]] = None,
        roll_windows: Optional[List[int]] = None,
        weather_cols: Optional[List[str]] = None,
    ) -> None:
        if model_type == "lgbm" and not _LGBM_AVAILABLE:
            model_type = "xgb"
        if model_type == "xgb" and not _XGB_AVAILABLE:
            raise ImportError("Neither lightgbm nor xgboost is installed.")
        self.model_type = model_type

        default_params = _DEFAULT_LGBM_PARAMS if model_type == "lgbm" else _DEFAULT_XGB_PARAMS
        self.params = {**default_params, **(params or {})}

        self.lag_steps = lag_steps or [4, 8, 96, 192, 288]   # 1h, 2h, 1d, 2d, 3d
        self.roll_windows = roll_windows or [4, 96]           # 1h, 1d
        self.weather_cols = weather_cols or self.DEFAULT_WEATHER

        self._model = None
        self._feature_names: List[str] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from raw DataFrame."""
        feat = add_time_features(df)
        feat = add_lag_features(feat, [self.TARGET], self.lag_steps)
        feat = add_lag_features(feat, self.weather_cols, [96])   # same time yesterday
        feat = add_rolling_features(feat, [self.TARGET], self.roll_windows, stats=["mean", "std"])
        feat.dropna(inplace=True)
        return feat

    def _get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        """Return the list of predictor column names (excludes target)."""
        exclude = {
            self.TARGET, "da_price", "rt_price",
            "unit_output", "tie_line_mw",
        }
        return [c for c in df.columns if c not in exclude]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        val_fraction: float = 0.1,
    ) -> "LoadForecaster":
        """Train the model on historical data.

        Args:
            df: DataFrame with DateTimeIndex (15-min) containing at least
                ``load_mw`` and weather columns.
            val_fraction: Fraction of training data held out for early stopping.

        Returns:
            self
        """
        feat = self._build_features(df)
        feature_cols = self._get_feature_cols(feat)
        self._feature_names = feature_cols

        X = feat[feature_cols]
        y = feat[self.TARGET].values

        split = int(len(X) * (1 - val_fraction))
        X_tr, X_val = X.iloc[:split], X.iloc[split:]
        y_tr, y_val = y[:split], y[split:]

        if self.model_type == "lgbm":
            params = {k: v for k, v in self.params.items()
                      if k not in ("early_stopping_rounds",)}
            n_estimators = params.pop("n_estimators", 500)
            early_stop = self.params.get("early_stopping_rounds", 30)
            self._model = lgb.LGBMRegressor(n_estimators=n_estimators, **params)
            self._model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(early_stop, verbose=False),
                           lgb.log_evaluation(period=-1)],
            )
        else:
            params = dict(self.params)
            early_stop = params.pop("early_stopping_rounds", 30)
            self._model = xgb.XGBRegressor(**params)
            self._model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Generate load predictions.

        Args:
            df: DataFrame with the same schema as training data.

        Returns:
            Series of predicted load values aligned to the DataFrame index.
        """
        if self._model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        feat = self._build_features(df)
        X = feat[self._feature_names]
        preds = self._model.predict(X)
        return pd.Series(preds, index=feat.index, name="load_mw_pred")

    def predict_day_ahead(self, df: pd.DataFrame, date: str) -> pd.Series:
        """Predict the full 96-point day-ahead load curve for a given date.

        Args:
            df: Full historical DataFrame (used to derive lag/rolling features).
            date: Target date string (e.g. '2024-03-15').

        Returns:
            Series of 96 predicted load values with 15-min DateTimeIndex.
        """
        target_date = pd.Timestamp(date).date()
        feat = self._build_features(df)
        day_mask = feat.index.date == target_date
        if not day_mask.any():
            raise ValueError(f"No data for date {date} after feature engineering.")
        X = feat.loc[day_mask, self._feature_names]
        preds = self._model.predict(X)
        return pd.Series(preds, index=feat.index[day_mask], name="load_mw_pred")

    def cross_validate(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        test_days: int = 7,
    ) -> List[Dict]:
        """Sliding-window cross-validation.

        Each fold trains on all data before the test window and evaluates on
        the next *test_days* of data, sliding the window forward.

        Args:
            df: Full DataFrame with DateTimeIndex (15-min).
            n_splits: Number of CV folds.
            test_days: Number of days in each test window.

        Returns:
            List of metric dicts, one per fold.
        """
        results = []
        # Build feature-engineered frame once to preserve lag continuity
        feat = self._build_features(df)
        feature_cols = self._get_feature_cols(feat)
        self._feature_names = feature_cols

        total_days = (feat.index.max() - feat.index.min()).days
        available_days = total_days - test_days * n_splits
        if available_days < test_days:
            raise ValueError(
                "Not enough data for the requested number of CV splits."
            )

        min_date = feat.index.min()

        for fold in range(n_splits):
            test_start = min_date + pd.Timedelta(
                days=available_days + fold * test_days
            )
            test_end = test_start + pd.Timedelta(days=test_days)

            train_mask = feat.index < test_start
            test_mask = (feat.index >= test_start) & (feat.index < test_end)

            if train_mask.sum() < 96 or test_mask.sum() < 96:
                continue

            X_tr = feat.loc[train_mask, feature_cols]
            y_tr = feat.loc[train_mask, self.TARGET].values
            X_te = feat.loc[test_mask, feature_cols]
            y_te = feat.loc[test_mask, self.TARGET].values

            split = int(len(X_tr) * 0.9)
            X_tv, X_val = X_tr.iloc[:split], X_tr.iloc[split:]
            y_tv, y_val = y_tr[:split], y_tr[split:]

            if self.model_type == "lgbm":
                params = {k: v for k, v in self.params.items()
                          if k not in ("early_stopping_rounds",)}
                n_est = params.pop("n_estimators", 500)
                early_stop = self.params.get("early_stopping_rounds", 30)
                model = lgb.LGBMRegressor(n_estimators=n_est, **params)
                model.fit(
                    X_tv, y_tv,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(early_stop, verbose=False),
                               lgb.log_evaluation(period=-1)],
                )
            else:
                params = dict(self.params)
                params.pop("early_stopping_rounds", None)
                model = xgb.XGBRegressor(**params)
                model.fit(X_tv, y_tv, eval_set=[(X_val, y_val)], verbose=False)

            preds = model.predict(X_te)
            metrics = evaluate_all(y_te, preds)
            metrics["fold"] = fold + 1
            metrics["test_start"] = str(test_start.date())
            metrics["test_end"] = str(test_end.date())
            results.append(metrics)

        # Save last fold's model as the fitted model
        if not results:
            raise ValueError("No CV folds were completed. Check that the dataset is large enough.")
        self._model = model  # noqa: F821
        return results

    def feature_importance(self) -> pd.DataFrame:
        """Return a DataFrame of feature importances sorted descending.

        Returns:
            DataFrame with columns ['feature', 'importance'].
        """
        if self._model is None:
            raise RuntimeError("Model is not fitted.")
        if self.model_type == "lgbm":
            imp = self._model.feature_importances_
        else:
            imp = self._model.feature_importances_
        return (
            pd.DataFrame({"feature": self._feature_names, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
