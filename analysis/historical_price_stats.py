"""
历史价格时段统计模块
Historical Price Statistics by Time Period

Reads:
    data/整合型 Excel.xlsx              : Historical 15-min records with
                                           day-ahead and real-time prices.
    data/25年10月-26年2月日前现货出清.xls : Day-ahead clearing details.
    data/25年10月-26年2月实时现货出清.xls : Real-time clearing details.

Key outputs (when run as __main__):
    - Console: period-level price statistics table.
    - Figures (saved to analysis/outputs/):
        price_period_boxplot.png   : DA vs RT price boxplot by period.
        spread_distribution.png   : DA–RT spread histogram by period.

Public API:
    load_all_data()             → (df_integrated, df_da, df_rt)
    compute_period_price_stats()→ stats DataFrame (period × metric)
    compute_spread_stats()      → spread DataFrame
    plot_price_period_boxplot() → Figure
    plot_spread_distribution()  → Figure
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from analysis.time_period_labeling import add_period_label

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _REPO_ROOT / "data"
_OUTPUT_DIR = _REPO_ROOT / "analysis" / "outputs"

HISTORY_FILE = _DATA_DIR / "整合型 Excel.xlsx"
DA_CLEARING_FILE = _DATA_DIR / "25年10月-26年2月日前现货出清.xls"
RT_CLEARING_FILE = _DATA_DIR / "25年10月-26年2月实时现货出清.xls"

# Column name constants — integrated Excel
_SLOT_COL = "时段序号"
_DATE_COL = "日期"
_DA_PRICE_COL = "日前出清价格（元 / 兆瓦时）"
_RT_PRICE_COL = "实时出清价格（元/兆瓦时）"
_LOAD_COL = "全省实时负荷"
_BID_SPACE_COL = "竞价空间（MW）"

# Column name constants — DA/RT clearing files
_CLEAR_DATE_COL = "日期"
_CLEAR_SLOT_COL = "时刻点"           # time string e.g. "00:15"
_CLEAR_DA_PRICE = "出清均价"
_CLEAR_DA_BID_PRICE = "申报均价"
_CLEAR_TOTAL_VOL = "出清总电量"
_CLEAR_THERMAL_VOL = "火电出清电量"
_CLEAR_WIND_VOL = "风电出清电量"
_CLEAR_SOLAR_VOL = "光伏出清电量"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _time_str_to_slot(time_str: str) -> int:
    """Convert a HH:MM time string to a 1-based 15-min slot index.

    E.g. '00:15' → 1, '00:30' → 2, '23:45' → 95.
    For 5-min resolution (real-time) data, the slot is rounded to the
    nearest 15-min boundary.
    """
    h, m = map(int, str(time_str).strip().split(":"))
    # Round to nearest 15-min period
    slot = (h * 60 + m) // 15 + 1
    return min(max(slot, 1), 96)


def load_all_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and pre-process all three source files.

    Returns:
        df_integrated : Integrated Excel with period labels.
        df_da         : Day-ahead clearing with slot numbers and period labels.
        df_rt         : Real-time clearing with slot numbers and period labels.
    """
    # -- Integrated Excel -------------------------------------------------
    df_int = pd.read_excel(HISTORY_FILE)
    df_int[_DATE_COL] = pd.to_datetime(df_int[_DATE_COL])
    df_int = df_int.sort_values([_DATE_COL, _SLOT_COL]).reset_index(drop=True)
    df_int = add_period_label(df_int, slot_col=_SLOT_COL)

    # -- Day-ahead clearing -----------------------------------------------
    df_da = pd.read_excel(DA_CLEARING_FILE)
    df_da[_CLEAR_DATE_COL] = pd.to_datetime(df_da[_CLEAR_DATE_COL])
    df_da["slot"] = df_da[_CLEAR_SLOT_COL].apply(_time_str_to_slot)
    df_da = add_period_label(df_da, slot_col="slot")
    df_da = df_da.sort_values([_CLEAR_DATE_COL, "slot"]).reset_index(drop=True)

    # -- Real-time clearing -----------------------------------------------
    df_rt = pd.read_excel(RT_CLEARING_FILE)
    df_rt[_CLEAR_DATE_COL] = pd.to_datetime(df_rt[_CLEAR_DATE_COL])
    df_rt["slot"] = df_rt[_CLEAR_SLOT_COL].apply(_time_str_to_slot)
    df_rt = add_period_label(df_rt, slot_col="slot")
    df_rt = df_rt.sort_values([_CLEAR_DATE_COL, "slot"]).reset_index(drop=True)

    return df_int, df_da, df_rt


# ---------------------------------------------------------------------------
# Statistics computation
# ---------------------------------------------------------------------------

def _describe_series(s: pd.Series) -> Dict:
    """Return a dict of standard descriptive statistics for a numeric Series."""
    s = s.dropna()
    if s.empty:
        return {k: np.nan for k in ("mean", "std", "min", "q25", "median", "q75", "q90", "q95", "max")}
    return {
        "mean":   float(s.mean()),
        "std":    float(s.std()),
        "min":    float(s.min()),
        "q25":    float(s.quantile(0.25)),
        "median": float(s.median()),
        "q75":    float(s.quantile(0.75)),
        "q90":    float(s.quantile(0.90)),
        "q95":    float(s.quantile(0.95)),
        "max":    float(s.max()),
    }


def compute_period_price_stats(df_integrated: pd.DataFrame) -> pd.DataFrame:
    """Compute day-ahead and real-time price statistics by time period.

    Args:
        df_integrated: Output of load_all_data()[0].

    Returns:
        DataFrame with MultiIndex (time_period, price_type) and stat columns.
    """
    periods = ["低谷", "平段", "高峰"]
    rows = []

    for period in periods:
        mask = df_integrated["time_period"] == period
        for price_col, price_label in [
            (_DA_PRICE_COL, "日前出清价格"),
            (_RT_PRICE_COL, "实时出清价格"),
        ]:
            if price_col not in df_integrated.columns:
                continue
            sub = df_integrated.loc[mask, price_col]
            row = {"时段": period, "价格类型": price_label}
            row.update(_describe_series(sub))
            rows.append(row)

    return pd.DataFrame(rows)


def compute_spread_stats(df_integrated: pd.DataFrame) -> pd.DataFrame:
    """Compute DA–RT price spread statistics by time period.

    Spread = 日前出清价格 − 实时出清价格

    Args:
        df_integrated: Output of load_all_data()[0].

    Returns:
        DataFrame with one row per period containing spread statistics plus
        positive/negative fraction columns.
    """
    periods = ["低谷", "平段", "高峰"]
    rows = []

    for period in periods:
        mask = df_integrated["time_period"] == period
        if _DA_PRICE_COL not in df_integrated.columns or _RT_PRICE_COL not in df_integrated.columns:
            break
        sub_da = df_integrated.loc[mask, _DA_PRICE_COL]
        sub_rt = df_integrated.loc[mask, _RT_PRICE_COL]
        spread = sub_da - sub_rt

        row = {"时段": period}
        row.update(_describe_series(spread))
        valid = spread.dropna()
        row["正价差占比(%)"] = float((valid > 0).mean() * 100) if len(valid) > 0 else np.nan
        row["负价差占比(%)"] = float((valid < 0).mean() * 100) if len(valid) > 0 else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def compute_clearing_volume_stats(df_da: pd.DataFrame, df_rt: pd.DataFrame) -> pd.DataFrame:
    """Compute clearing volume statistics for key generation types by period.

    Args:
        df_da: Day-ahead clearing DataFrame from load_all_data()[1].
        df_rt: Real-time clearing DataFrame from load_all_data()[2].

    Returns:
        DataFrame with mean clearing volumes by period and market type.
    """
    periods = ["低谷", "平段", "高峰"]
    gen_types = {
        "火电出清电量": "火电",
        "风电出清电量": "风电",
        "光伏出清电量": "光伏",
        "出清总电量": "总计",
    }

    rows = []
    for period in periods:
        for df, market in [(df_da, "日前"), (df_rt, "实时")]:
            mask = df["time_period"] == period
            row = {"时段": period, "市场": market}
            for col, label in gen_types.items():
                if col in df.columns:
                    row[f"{label}均值(MWh)"] = float(df.loc[mask, col].mean())
            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _ensure_output_dir() -> None:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_price_period_boxplot(
    df_integrated: pd.DataFrame,
    save: bool = True,
    show: bool = False,
) -> "plt.Figure":  # type: ignore[name-defined]
    """Box-plot of DA vs RT prices for each time period.

    Args:
        df_integrated: Output of load_all_data()[0].
        save: Save to analysis/outputs/price_period_boxplot.png.
        show: Display figure interactively.

    Returns:
        matplotlib Figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    periods = ["低谷", "平段", "高峰"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=False)

    for ax, period in zip(axes, periods):
        mask = df_integrated["time_period"] == period
        data, labels = [], []
        for col, lbl in [(_DA_PRICE_COL, "日前价格"), (_RT_PRICE_COL, "实时价格")]:
            if col in df_integrated.columns:
                series = df_integrated.loc[mask, col].dropna()
                if not series.empty:
                    data.append(series.values)
                    labels.append(lbl)

        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            colours = ["#4292c6", "#ef6548"]
            for patch, c in zip(bp["boxes"], colours[:len(data)]):
                patch.set_facecolor(c)
                patch.set_alpha(0.65)

        ax.set_title(f"{period}", fontsize=12, fontweight="bold")
        ax.set_ylabel("价格 (元/MWh)", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("历史各时段日前/实时出清价格分布", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save:
        _ensure_output_dir()
        fig.savefig(_OUTPUT_DIR / "price_period_boxplot.png", dpi=150)
    if show:
        plt.show()

    return fig


def plot_spread_distribution(
    df_integrated: pd.DataFrame,
    save: bool = True,
    show: bool = False,
) -> "Optional[plt.Figure]":  # type: ignore[name-defined]
    """Histogram of DA–RT spread, faceted by time period.

    Args:
        df_integrated: Output of load_all_data()[0].
        save: Save to analysis/outputs/spread_distribution.png.
        show: Display interactively.

    Returns:
        matplotlib Figure, or None if required price columns are missing.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if _DA_PRICE_COL not in df_integrated.columns or _RT_PRICE_COL not in df_integrated.columns:
        return None

    df_integrated = df_integrated.copy()
    df_integrated["spread"] = df_integrated[_DA_PRICE_COL] - df_integrated[_RT_PRICE_COL]

    periods = ["低谷", "平段", "高峰"]
    colours = ["#6baed6", "#74c476", "#fd8d3c"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, period, colour in zip(axes, periods, colours):
        data = df_integrated.loc[df_integrated["time_period"] == period, "spread"].dropna()
        if not data.empty:
            ax.hist(data, bins=40, color=colour, alpha=0.75, edgecolor="white")
            ax.axvline(data.mean(), color="red", linestyle="--", linewidth=1.5,
                       label=f"均值 {data.mean():+.1f}")
            ax.axvline(0, color="black", linestyle="-", linewidth=0.8)
            ax.legend(fontsize=8)

        ax.set_title(f"{period} 价差分布", fontsize=11, fontweight="bold")
        ax.set_xlabel("日前−实时价差 (元/MWh)", fontsize=9)
        ax.set_ylabel("频次", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("历史各时段日前–实时价差分布直方图", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save:
        _ensure_output_dir()
        fig.savefig(_OUTPUT_DIR / "spread_distribution.png", dpi=150)
    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Main — standalone execution
# ---------------------------------------------------------------------------

def main() -> None:
    """Run historical price statistics analysis and print results."""
    print("=" * 60)
    print("  历史价格时段统计分析")
    print("=" * 60)

    df_int, df_da, df_rt = load_all_data()
    print(f"\n  整合型数据: {len(df_int):,} 行  ({df_int[_DATE_COL].min().date()} → {df_int[_DATE_COL].max().date()})")
    print(f"  日前出清:   {len(df_da):,} 行")
    print(f"  实时出清:   {len(df_rt):,} 行")

    print("\n  各时段价格统计:")
    price_stats = compute_period_price_stats(df_int)
    _FMT = {c: lambda x: f"{x:,.2f}" for c in price_stats.columns if c not in ("时段", "价格类型")}
    print(price_stats.to_string(index=False, formatters=_FMT))

    print("\n  各时段日前–实时价差统计:")
    spread_stats = compute_spread_stats(df_int)
    _SFMT = {
        c: (lambda x: f"{x:,.2f}") if c not in ("时段",) else str
        for c in spread_stats.columns
    }
    print(spread_stats.to_string(index=False, formatters=_SFMT))

    print("\n  各时段出清电量均值:")
    vol_stats = compute_clearing_volume_stats(df_da, df_rt)
    print(vol_stats.to_string(index=False))

    print("\n  生成价格图表 …")
    plot_price_period_boxplot(df_int, save=True)
    plot_spread_distribution(df_int, save=True)
    print(f"  图表已保存至 {_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
