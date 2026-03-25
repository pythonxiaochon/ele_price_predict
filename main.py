"""
完整预测流入口
Main Entry Point — Full Prediction Pipeline

Usage:
    python main.py [--start 2024-01-01] [--end 2024-03-31] [--no-rt]

Steps executed:
    1. Generate (or load) synthetic sample data at 15-min granularity.
    2. Preprocess: align index, engineer features.
    3. Train LoadForecaster → evaluate with sliding-window CV.
    4. Train SpreadAnalyzer → compute statistics and identify arbitrage windows.
    5. (Optional) Train RTPricePredictor → evaluate on hold-out period.
    6. Print a consolidated metrics report.
"""

import argparse
import warnings
from typing import Optional

import pandas as pd

from data.preprocess import (
    generate_sample_data,
    align_15min_index,
    train_test_split_ts,
)
from models.load_forecasting import LoadForecaster
from models.spread_analysis import SpreadAnalyzer

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="江西电力市场分析与预测框架 — 完整预测流"
    )
    parser.add_argument(
        "--start", default="2024-01-01", help="Data start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", default="2024-03-31", help="Data end date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--cv-splits", type=int, default=3, help="Number of cross-validation folds"
    )
    parser.add_argument(
        "--no-rt", action="store_true", help="Skip real-time price prediction (no PyTorch required)"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pretty printing helpers
# ---------------------------------------------------------------------------

def _print_section(title: str) -> None:
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def _print_metrics(metrics: dict, prefix: str = "") -> None:
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {prefix}{k:20s}: {v:.4f}")
        else:
            print(f"  {prefix}{k:20s}: {v}")


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def run_load_forecasting(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_splits: int = 3,
) -> pd.Series:
    """Train and evaluate the load forecasting module.

    Args:
        train_df: Training DataFrame with 15-min DateTimeIndex.
        test_df: Hold-out test DataFrame.
        n_splits: Number of CV folds.

    Returns:
        Series of test-set load predictions.
    """
    _print_section("负荷预测 (Load Forecasting — LightGBM)")

    forecaster = LoadForecaster()

    print("\n[1/2] Running sliding-window cross-validation …")
    # CV needs a contiguous dataset; pass the full train set
    cv_results = forecaster.cross_validate(train_df, n_splits=n_splits, test_days=5)
    for fold in cv_results:
        print(f"  Fold {fold['fold']} | MAPE={fold['mape']:.2f}% | "
              f"RMSE={fold['rmse']:.2f} | R²={fold['r2']:.4f} | "
              f"Test: {fold['test_start']} → {fold['test_end']}")

    print("\n[2/2] Training final model on full training set …")
    forecaster.fit(train_df)

    preds = forecaster.predict(test_df)
    # Evaluate on aligned actuals
    feat = forecaster._build_features(test_df)
    common = preds.index.intersection(feat.index)
    from utils.metrics import evaluate_all
    metrics = evaluate_all(feat.loc[common, "load_mw"].values, preds.loc[common].values)
    print("\n  Hold-out Test Metrics:")
    _print_metrics(metrics)

    return preds


def run_spread_analysis(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_splits: int = 3,
) -> pd.Series:
    """Train and evaluate the spread analysis module.

    Args:
        train_df: Training DataFrame.
        test_df: Hold-out test DataFrame.
        n_splits: Number of CV folds.

    Returns:
        Series of test-set spread predictions (used by RT module).
    """
    _print_section("日前—实时价差分析 (Spread Analysis — Random Forest)")

    analyzer = SpreadAnalyzer()

    print("\n[1/3] Computing spread statistics …")
    stats = analyzer.compute_statistics(train_df)
    print(f"  Mean spread  : {stats['mean']:+.2f} ¥/MWh")
    print(f"  Std spread   : {stats['std']:.2f} ¥/MWh")
    print(f"  Positive frac: {stats['positive_fraction']*100:.1f}%")
    print(f"  Q5 / Q95     : {stats['q5']:+.1f} / {stats['q95']:+.1f} ¥/MWh")

    print("\n[2/3] Identifying arbitrage windows …")
    arb_df = analyzer.identify_arbitrage_windows(train_df)
    n_arb = arb_df["is_arbitrage"].sum()
    pct_arb = n_arb / len(arb_df) * 100
    print(f"  Abnormal arbitrage slots: {n_arb:,} / {len(arb_df):,} ({pct_arb:.1f}%)")

    print("\n[3/3] Training spread prediction model with time-series CV …")
    cv_results = analyzer.cross_validate(train_df, n_splits=n_splits)
    for fold in cv_results:
        print(f"  Fold {fold['fold']} | MAPE={fold['mape']:.2f}% | "
              f"RMSE={fold['rmse']:.2f} | R²={fold['r2']:.4f}")

    analyzer.fit(train_df)
    spread_preds = analyzer.predict(test_df)

    print("\n  Hourly Spread Profile (first 8 slots):")
    profile = analyzer.hourly_spread_profile(train_df)
    print(profile.head(8).to_string())

    return spread_preds


def run_rt_price_prediction(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    spread_preds: Optional[pd.Series] = None,
) -> None:
    """Train and evaluate the real-time price prediction module.

    Args:
        train_df: Training DataFrame.
        test_df: Hold-out test DataFrame.
        spread_preds: Optional spread forecast Series from SpreadAnalyzer.
    """
    _print_section("实时价格预测 (RT Price Prediction — LSTM/GRU)")

    try:
        from models.rt_price_prediction import RTPricePredictor
    except ImportError as exc:
        print(f"  Skipping RT prediction: {exc}")
        return

    print("\n  Training LSTM model (this may take a few minutes) …")
    predictor = RTPricePredictor(
        model_type="lstm",
        seq_len=48,         # 12-hour lookback window
        hidden_size=32,
        num_layers=1,
        epochs=20,
        patience=5,
        batch_size=128,
    )

    predictor.fit(train_df, spread_pred=spread_preds)
    metrics = predictor.evaluate(test_df, spread_pred=spread_preds)

    print("\n  Hold-out Test Metrics:")
    _print_metrics(metrics)

    print("\n  Training history (last 5 epochs):")
    for rec in predictor.train_history[-5:]:
        print(f"    Epoch {rec['epoch']:3d} | "
              f"train_loss={rec['train_loss']:.6f} | "
              f"val_loss={rec['val_loss']:.6f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    _print_section("江西电力市场分析与预测框架")
    print(f"  Data period  : {args.start} → {args.end}")
    print(f"  CV folds     : {args.cv_splits}")
    print(f"  RT prediction: {'disabled' if args.no_rt else 'enabled'}")

    # ------------------------------------------------------------------
    # Step 1: Generate / load data
    # ------------------------------------------------------------------
    _print_section("Step 1: Data Generation & Preprocessing")
    print("\n  Generating synthetic 15-min Jiangxi market data …")
    df = generate_sample_data(start=args.start, end=args.end)
    df = align_15min_index(df)
    print(f"  Total records: {len(df):,} (15-min slots across {(df.index.max()-df.index.min()).days+1} days)")
    print(f"  Columns      : {list(df.columns)}")

    # ------------------------------------------------------------------
    # Step 2: Train / test split (last 7 days as test)
    # ------------------------------------------------------------------
    train_df, test_df = train_test_split_ts(df, test_days=7)
    print(f"\n  Train: {train_df.index.min()} → {train_df.index.max()} ({len(train_df):,} rows)")
    print(f"  Test : {test_df.index.min()} → {test_df.index.max()} ({len(test_df):,} rows)")

    # ------------------------------------------------------------------
    # Step 3: Load Forecasting
    # ------------------------------------------------------------------
    run_load_forecasting(train_df, test_df, n_splits=args.cv_splits)

    # ------------------------------------------------------------------
    # Step 4: Spread Analysis
    # ------------------------------------------------------------------
    spread_preds = run_spread_analysis(train_df, test_df, n_splits=args.cv_splits)

    # ------------------------------------------------------------------
    # Step 5: Real-time Price Prediction (optional)
    # ------------------------------------------------------------------
    if not args.no_rt:
        run_rt_price_prediction(train_df, test_df, spread_preds=spread_preds)

    _print_section("Pipeline Complete")
    print("\n  All modules executed successfully.")
    print("  Refer to individual module outputs above for detailed metrics.\n")


if __name__ == "__main__":
    main()
