"""
数据预处理模块
Data Preprocessing Module

Responsibilities:
    - Generate synthetic 15-min interval sample data for development/testing.
    - Enforce strict 15-minute DateTimeIndex alignment across all DataFrames.
    - Engineer time-based, lag, and rolling features used by the three core models.
"""

import numpy as np
import pandas as pd
from typing import Optional, List


# ---------------------------------------------------------------------------
# Index utilities
# ---------------------------------------------------------------------------

def align_15min_index(
    df: pd.DataFrame,
    start: Optional[str] = None,
    end: Optional[str] = None,
    fill_method: str = "ffill",
) -> pd.DataFrame:
    """Re-index a DataFrame to a strict 15-minute frequency.

    Args:
        df: Input DataFrame with a DateTimeIndex (or convertible index).
        start: Start timestamp string (defaults to df.index.min()).
        end: End timestamp string (defaults to df.index.max()).
        fill_method: Strategy for filling gaps ('ffill', 'bfill', or 'interpolate').

    Returns:
        DataFrame with a complete 15-minute DateTimeIndex and no NaN gaps.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    start_ts = pd.Timestamp(start) if start else df.index.min()
    end_ts = pd.Timestamp(end) if end else df.index.max()
    full_index = pd.date_range(start=start_ts, end=end_ts, freq="15min")
    df = df.reindex(full_index)

    if fill_method == "interpolate":
        df = df.interpolate(method="time")
    elif fill_method == "ffill":
        df = df.ffill().bfill()
    elif fill_method == "bfill":
        df = df.bfill().ffill()

    return df


# ---------------------------------------------------------------------------
# Sample data generation
# ---------------------------------------------------------------------------

def generate_sample_data(
    start: str = "2024-01-01",
    end: str = "2024-03-31",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic 15-minute power market data for Jiangxi province.

    Columns produced:
        load_mw      : Grid load in MW (sine-based daily pattern + noise).
        temperature  : Ambient temperature in °C.
        humidity     : Relative humidity (%).
        irradiance   : Solar irradiance (W/m²).
        da_price     : Day-ahead clearing price (¥/MWh).
        rt_price     : Real-time clearing price (¥/MWh).
        unit_output  : Real-time unit output utilisation ratio [0, 1].
        tie_line_mw  : Inter-regional tie-line power flow (MW).

    Args:
        start: Start date string (inclusive).
        end: End date string (inclusive).
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with a 15-minute DateTimeIndex.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range(start=start, end=end, freq="15min")
    n = len(index)

    # Fractional hour of day [0, 24)
    hour_frac = np.array(index.hour, dtype=float) + np.array(index.minute, dtype=float) / 60.0
    day_of_year = np.array(index.dayofyear, dtype=float)

    # -- Load: base + daily sine + seasonal component + noise
    daily_load = (
        8000
        + 3000 * np.sin(2 * np.pi * (hour_frac - 6) / 24)
        + 500 * np.sin(2 * np.pi * day_of_year / 365)
        + rng.normal(0, 200, n)
    ).clip(3000, 14000)

    # -- Temperature: seasonal + diurnal + noise
    temperature = (
        15
        + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        + 5 * np.sin(2 * np.pi * (hour_frac - 14) / 24)
        + rng.normal(0, 1, n)
    )

    # -- Humidity: 40–90 % range
    humidity = (
        65
        + 20 * np.sin(2 * np.pi * day_of_year / 365 + np.pi)
        + rng.normal(0, 5, n)
    ).clip(30, 95)

    # -- Solar irradiance: 0 at night, peaks at noon
    irradiance = np.where(
        (hour_frac >= 6) & (hour_frac <= 18),
        800 * np.sin(np.pi * (hour_frac - 6) / 12) + rng.normal(0, 50, n),
        0.0,
    ).clip(0)

    # -- Day-ahead price: load-correlated with random spread
    da_price = (
        250
        + 0.02 * daily_load
        + 30 * np.sin(2 * np.pi * (hour_frac - 8) / 24)
        + rng.normal(0, 15, n)
    ).clip(100, 600)

    # -- Real-time price: day-ahead price + spread noise
    spread = rng.normal(0, 20, n) + 10 * np.sin(2 * np.pi * hour_frac / 24)
    rt_price = (da_price + spread).clip(80, 650)

    # -- Unit output utilisation
    unit_output = (
        0.6
        + 0.2 * np.sin(2 * np.pi * (hour_frac - 8) / 24)
        + rng.normal(0, 0.05, n)
    ).clip(0.1, 1.0)

    # -- Tie-line power flow
    tie_line_mw = (
        500
        + 200 * np.sin(2 * np.pi * hour_frac / 12)
        + rng.normal(0, 50, n)
    )

    df = pd.DataFrame(
        {
            "load_mw": daily_load,
            "temperature": temperature,
            "humidity": humidity,
            "irradiance": irradiance,
            "da_price": da_price,
            "rt_price": rt_price,
            "unit_output": unit_output,
            "tie_line_mw": tie_line_mw,
        },
        index=index,
    )
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

_CHINESE_HOLIDAYS_2024 = {
    # New Year
    "2024-01-01",
    # Spring Festival
    "2024-02-10", "2024-02-11", "2024-02-12", "2024-02-13",
    "2024-02-14", "2024-02-15", "2024-02-16", "2024-02-17",
    # Tomb-sweeping Day
    "2024-04-04", "2024-04-05", "2024-04-06",
    # Labour Day
    "2024-05-01", "2024-05-02", "2024-05-03", "2024-05-04", "2024-05-05",
    # Dragon Boat Festival
    "2024-06-08", "2024-06-09", "2024-06-10",
    # Mid-autumn Festival
    "2024-09-15", "2024-09-16", "2024-09-17",
    # National Day / Golden Week
    "2024-10-01", "2024-10-02", "2024-10-03", "2024-10-04",
    "2024-10-05", "2024-10-06", "2024-10-07",
}


def add_time_features(df: pd.DataFrame, holiday_dates: Optional[set] = None) -> pd.DataFrame:
    """Add calendar and date-type feature columns.

    New columns added (all integer-encoded):
        hour, minute, dayofweek, month, quarter,
        time_slot (0-95, 15-min slot within the day),
        is_weekend (0/1),
        is_holiday (0/1),
        is_workday (0/1, complement of weekend + holiday).

    Args:
        df: Input DataFrame with a DateTimeIndex.
        holiday_dates: Set of date strings ('YYYY-MM-DD') considered holidays.
                       Defaults to a built-in 2024 Chinese holiday calendar.

    Returns:
        DataFrame with additional time feature columns (copy).
    """
    if holiday_dates is None:
        holiday_dates = _CHINESE_HOLIDAYS_2024

    holiday_set = {pd.Timestamp(d).date() for d in holiday_dates}
    out = df.copy()
    idx = out.index

    out["hour"] = idx.hour
    out["minute"] = idx.minute
    out["dayofweek"] = idx.dayofweek          # 0=Monday … 6=Sunday
    out["month"] = idx.month
    out["quarter"] = idx.quarter
    out["time_slot"] = idx.hour * 4 + idx.minute // 15   # 0–95
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    out["is_holiday"] = idx.date
    out["is_holiday"] = out["is_holiday"].apply(lambda d: int(d in holiday_set))
    out["is_workday"] = ((out["is_weekend"] == 0) & (out["is_holiday"] == 0)).astype(int)

    return out


def add_lag_features(
    df: pd.DataFrame,
    columns: List[str],
    lags: List[int],
) -> pd.DataFrame:
    """Add lag features for specified columns.

    Args:
        df: Input DataFrame with a DateTimeIndex at 15-min frequency.
        columns: Column names to create lags for.
        lags: List of lag steps (in number of 15-min periods).
              E.g., [4, 96, 192] corresponds to 1 hour, 1 day, 2 days ago.

    Returns:
        DataFrame with additional lag columns (copy).
    """
    out = df.copy()
    for col in columns:
        for lag in lags:
            out[f"{col}_lag{lag}"] = out[col].shift(lag)
    return out


def add_rolling_features(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int],
    stats: List[str] = ("mean", "std"),
) -> pd.DataFrame:
    """Add rolling window statistics for specified columns.

    Args:
        df: Input DataFrame with a DateTimeIndex at 15-min frequency.
        columns: Column names to compute rolling statistics for.
        windows: Window sizes in number of 15-min periods.
        stats: Which statistics to compute ('mean', 'std', 'min', 'max').

    Returns:
        DataFrame with additional rolling columns (copy).
    """
    out = df.copy()
    for col in columns:
        for w in windows:
            roller = out[col].rolling(window=w, min_periods=1)
            for stat in stats:
                out[f"{col}_roll{w}_{stat}"] = getattr(roller, stat)()
    return out


# ---------------------------------------------------------------------------
# Train / test split helper
# ---------------------------------------------------------------------------

def train_test_split_ts(
    df: pd.DataFrame,
    test_days: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time-series DataFrame into train and test sets.

    The test set consists of the last *test_days* calendar days.

    Args:
        df: Input DataFrame with a DateTimeIndex.
        test_days: Number of days to reserve for testing.

    Returns:
        Tuple of (train_df, test_df).
    """
    split_ts = df.index.max() - pd.Timedelta(days=test_days)
    return df[df.index <= split_ts], df[df.index > split_ts]
