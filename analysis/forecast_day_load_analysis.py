"""
预测日负荷分析模块 — 2026-03-01
Forecast-Day Load Analysis Module — 2026-03-01

Reads:
    data/真实负荷.xlsx          : 96-point load for 2026-03-01.
    data/整合型 Excel.xlsx      : Historical load (2025-10 → 2026-02).

Outputs (when run as __main__):
    - Console: load statistics, per-period summary, historical comparison.
    - Figures (saved to analysis/outputs/):
        load_curve_20260301.png  : 96-point load curve with period colouring.
        load_period_boxplot.png  : Historical vs forecast load by period.

Public API:
    load_forecast_day()          → (df_96, stats_dict)
    load_historical_march_data() → df_hist (March records from history)
    compare_with_history()       → comparison DataFrame
    plot_load_curve()            → matplotlib Figure
    plot_period_comparison()     → matplotlib Figure
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from analysis.time_period_labeling import add_period_label, PERIOD_SLOTS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _REPO_ROOT / "data"
_OUTPUT_DIR = _REPO_ROOT / "analysis" / "outputs"

FORECAST_LOAD_FILE = _DATA_DIR / "真实负荷.xlsx"
HISTORY_FILE = _DATA_DIR / "整合型 Excel.xlsx"

# Column aliases
_SLOT_COL = "时段序号"
_LOAD_COL_FORECAST = "全省实时负荷（MW）"
_LOAD_COL_HIST = "全省实时负荷"
_DATE_COL = "日期"
_DATE_COL_FORECAST = "预测日"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_forecast_day() -> Tuple[pd.DataFrame, Dict]:
    """Load and analyse the 2026-03-01 forecast-day load data.

    Returns:
        df_96: DataFrame with columns ['时段序号', '全省实时负荷（MW）',
               'time_period'].  Index is 0-based (row 0 = slot 1).
        stats: Dict with keys: max_load, min_load, mean_load, peak_valley_diff,
               peak_slot, valley_slot, morning_peak_load, evening_peak_load,
               dual_peak_diff, per_period (nested dict per period).
    """
    df = pd.read_excel(FORECAST_LOAD_FILE)
    df = df[[_DATE_COL_FORECAST, _SLOT_COL, _LOAD_COL_FORECAST]].copy()
    df = df.sort_values(_SLOT_COL).reset_index(drop=True)
    df = add_period_label(df, slot_col=_SLOT_COL)

    load = df[_LOAD_COL_FORECAST]

    peak_idx = load.idxmax()
    valley_idx = load.idxmin()

    # Morning peak heuristic: max load in slots 33-52 (08:00-13:00)
    morning_mask = df[_SLOT_COL].between(33, 52)
    morning_peak = load[morning_mask].max() if morning_mask.any() else np.nan
    # Evening peak: max load in slots 65-88 (16:00-22:00)
    evening_mask = df[_SLOT_COL].between(65, 88)
    evening_peak = load[evening_mask].max() if evening_mask.any() else np.nan

    per_period: Dict = {}
    for period in ("低谷", "平段", "高峰"):
        mask = df["time_period"] == period
        sub = load[mask]
        per_period[period] = {
            "mean": float(sub.mean()),
            "max": float(sub.max()),
            "min": float(sub.min()),
            "count": int(mask.sum()),
        }

    stats: Dict = {
        "max_load": float(load.max()),
        "min_load": float(load.min()),
        "mean_load": float(load.mean()),
        "peak_valley_diff": float(load.max() - load.min()),
        "peak_slot": int(df.loc[peak_idx, _SLOT_COL]),
        "valley_slot": int(df.loc[valley_idx, _SLOT_COL]),
        "morning_peak_load": float(morning_peak),
        "evening_peak_load": float(evening_peak),
        "dual_peak_diff": float(evening_peak - morning_peak)
            if not (np.isnan(morning_peak) or np.isnan(evening_peak)) else np.nan,
        "per_period": per_period,
    }

    return df, stats


def load_historical_data() -> pd.DataFrame:
    """Load full historical data from 整合型 Excel.xlsx.

    Returns:
        DataFrame with columns from the history file plus 'time_period' label.
    """
    df = pd.read_excel(HISTORY_FILE)
    df[_DATE_COL] = pd.to_datetime(df[_DATE_COL])
    df = df.sort_values([_DATE_COL, _SLOT_COL]).reset_index(drop=True)
    df = add_period_label(df, slot_col=_SLOT_COL)
    return df


def load_historical_march_data() -> pd.DataFrame:
    """Return only the March records from the history file (month == 3).

    Returns:
        Filtered DataFrame (March-only historical data with period labels).
    """
    df = load_historical_data()
    return df[df[_DATE_COL].dt.month == 3].copy()


def compare_with_history(
    df_forecast: pd.DataFrame,
    df_hist: pd.DataFrame,
) -> pd.DataFrame:
    """Compare forecast-day per-period loads with historical March averages.

    Args:
        df_forecast: Output DataFrame from load_forecast_day().
        df_hist: Historical March DataFrame from load_historical_march_data().

    Returns:
        DataFrame with columns: ['time_period', 'forecast_mean_MW',
        'hist_mean_MW', 'hist_std_MW', 'diff_MW', 'diff_pct'].
    """
    forecast_agg = (
        df_forecast.groupby("time_period")[_LOAD_COL_FORECAST]
        .mean()
        .rename("forecast_mean_MW")
    )

    # Historical March data may be empty if no March records exist in history
    if df_hist.empty:
        hist_agg = pd.DataFrame(
            columns=["time_period", "hist_mean_MW", "hist_std_MW"]
        ).set_index("time_period")
    else:
        hist_agg = (
            df_hist.groupby("time_period")[_LOAD_COL_HIST]
            .agg(hist_mean_MW="mean", hist_std_MW="std")
        )

    comp = forecast_agg.to_frame().join(hist_agg, how="left")
    comp["diff_MW"] = comp["forecast_mean_MW"] - comp["hist_mean_MW"]
    comp["diff_pct"] = (comp["diff_MW"] / comp["hist_mean_MW"] * 100).round(2)
    comp = comp.reset_index()
    # Reorder periods logically
    order = {"低谷": 0, "平段": 1, "高峰": 2}
    comp["_order"] = comp["time_period"].map(order)
    comp = comp.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return comp


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _ensure_output_dir() -> None:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_load_curve(
    df: pd.DataFrame,
    save: bool = True,
    show: bool = False,
) -> "plt.Figure":  # type: ignore[name-defined]
    """Plot the 96-point forecast-day load curve with period colouring.

    Args:
        df: Output of load_forecast_day() with 'time_period' column.
        save: If True, save PNG to analysis/outputs/.
        show: If True, display the figure interactively.

    Returns:
        matplotlib Figure object.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    period_colours = {"低谷": "#6baed6", "平段": "#74c476", "高峰": "#fd8d3c"}

    fig, ax = plt.subplots(figsize=(14, 5))
    slots = df[_SLOT_COL].values
    load = df[_LOAD_COL_FORECAST].values

    # Background shading by period
    for i, row in df.iterrows():
        colour = period_colours.get(row["time_period"], "#cccccc")
        ax.axvspan(row[_SLOT_COL] - 0.5, row[_SLOT_COL] + 0.5,
                   alpha=0.15, color=colour, linewidth=0)

    ax.plot(slots, load, color="#333333", linewidth=1.8, zorder=3)
    ax.set_xlabel("时段序号 (1–96)", fontsize=11)
    ax.set_ylabel("全省实时负荷 (MW)", fontsize=11)
    ax.set_title("2026-03-01 全省负荷曲线（时段颜色标注）", fontsize=13, fontweight="bold")

    # X-axis hour labels
    hour_ticks = list(range(1, 97, 4))
    hour_labels = [f"{(i-1)//4:02d}:00" for i in hour_ticks]
    ax.set_xticks(hour_ticks)
    ax.set_xticklabels(hour_labels, rotation=45, fontsize=8)

    legend_handles = [
        mpatches.Patch(color=c, alpha=0.4, label=lbl)
        for lbl, c in period_colours.items()
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save:
        _ensure_output_dir()
        fig.savefig(_OUTPUT_DIR / "load_curve_20260301.png", dpi=150)

    if show:
        plt.show()

    return fig


def plot_period_comparison(
    df_forecast: pd.DataFrame,
    df_hist: pd.DataFrame,
    save: bool = True,
    show: bool = False,
) -> "plt.Figure":  # type: ignore[name-defined]
    """Box-plot comparing historical vs forecast-day load by period.

    Args:
        df_forecast: Output of load_forecast_day().
        df_hist: Output of load_historical_march_data() (may be empty).
        save: If True, save PNG to analysis/outputs/.
        show: If True, display interactively.

    Returns:
        matplotlib Figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    periods = ["低谷", "平段", "高峰"]
    fig, axes = plt.subplots(1, len(periods), figsize=(14, 5), sharey=False)

    for ax, period in zip(axes, periods):
        hist_data = (
            df_hist.loc[df_hist["time_period"] == period, _LOAD_COL_HIST].dropna()
            if not df_hist.empty else pd.Series(dtype=float)
        )
        fc_data = df_forecast.loc[
            df_forecast["time_period"] == period, _LOAD_COL_FORECAST
        ].dropna()

        data_list = []
        labels = []
        if not hist_data.empty:
            data_list.append(hist_data.values)
            labels.append("历史3月")
        if not fc_data.empty:
            data_list.append(fc_data.values)
            labels.append("2026-03-01")

        if data_list:
            bp = ax.boxplot(data_list, labels=labels, patch_artist=True)
            colours = ["#6baed6", "#fd8d3c"]
            for patch, colour in zip(bp["boxes"], colours[:len(data_list)]):
                patch.set_facecolor(colour)
                patch.set_alpha(0.6)

        ax.set_title(f"{period}", fontsize=12, fontweight="bold")
        ax.set_ylabel("负荷 (MW)", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("历史3月 vs 2026-03-01 各时段负荷对比", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save:
        _ensure_output_dir()
        fig.savefig(_OUTPUT_DIR / "load_period_boxplot.png", dpi=150)

    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Main — standalone execution
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full forecast-day load analysis and print results."""
    print("=" * 60)
    print("  预测日负荷分析 — 2026-03-01")
    print("=" * 60)

    df_forecast, stats = load_forecast_day()
    df_hist_march = load_historical_march_data()

    print(f"\n  最高负荷  : {stats['max_load']:>10,.2f} MW  (时段 {stats['peak_slot']})")
    print(f"  最低负荷  : {stats['min_load']:>10,.2f} MW  (时段 {stats['valley_slot']})")
    print(f"  平均负荷  : {stats['mean_load']:>10,.2f} MW")
    print(f"  峰谷差    : {stats['peak_valley_diff']:>10,.2f} MW")
    print(f"  上午高峰  : {stats['morning_peak_load']:>10,.2f} MW  (时段 33-52)")
    print(f"  傍晚高峰  : {stats['evening_peak_load']:>10,.2f} MW  (时段 65-88)")
    print(f"  双峰差    : {stats['dual_peak_diff']:>10,.2f} MW")

    print("\n  各时段负荷统计:")
    print(f"  {'时段':<8}{'均值(MW)':>12}{'最大(MW)':>12}{'最小(MW)':>12}{'时段数':>8}")
    for period, pdata in stats["per_period"].items():
        print(
            f"  {period:<8}{pdata['mean']:>12,.2f}"
            f"{pdata['max']:>12,.2f}{pdata['min']:>12,.2f}"
            f"{pdata['count']:>8}"
        )

    print("\n  与历史3月数据对比:")
    comp = compare_with_history(df_forecast, df_hist_march)
    if comp["hist_mean_MW"].notna().any():
        print(
            comp.to_string(
                index=False,
                float_format=lambda x: f"{x:,.2f}",
            )
        )
    else:
        print("  (历史数据中无3月记录，跳过对比)")

    print("\n  生成图表 …")
    plot_load_curve(df_forecast, save=True)
    plot_period_comparison(df_forecast, df_hist_march, save=True)
    print(f"  图表已保存至 {_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
