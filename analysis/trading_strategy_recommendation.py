"""
发电侧竞标策略建议模块 — 2026-03-01
Generator-Side Bidding Strategy Recommendation — 2026-03-01

Integrates results from:
    - forecast_day_load_analysis     : 2026-03-01 load profile
    - historical_price_stats         : Historical price distribution by period
    - correlation_marginal_analysis  : Correlation and marginal analysis

Outputs (when run as __main__):
    - Console: quantitative bidding price recommendations, output guidance,
               arbitrage windows and risk warnings.
    - Figures (saved to analysis/outputs/):
        bidding_strategy_table.png   : Visual summary table.
        risk_matrix.png              : Risk matrix chart.

Public API:
    build_price_recommendation()   → DataFrame (period × quantile prices)
    build_storage_arbitrage()      → DataFrame (charge/discharge windows)
    build_risk_matrix()            → DataFrame (risk × probability × impact)
    print_strategy_report()        → None (prints full console report)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from analysis.time_period_labeling import add_period_label, PERIOD_SLOTS
from analysis.historical_price_stats import (
    load_all_data,
    compute_period_price_stats,
    compute_spread_stats,
    compute_clearing_volume_stats,
)
from analysis.forecast_day_load_analysis import load_forecast_day

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT_DIR = _REPO_ROOT / "analysis" / "outputs"

# Integrated Excel column names
_RT_PRICE_COL = "实时出清价格（元/兆瓦时）"
_DA_PRICE_COL = "日前出清价格（元 / 兆瓦时）"
_LOAD_COL = "全省实时负荷"
_BID_SPACE_COL = "竞价空间（MW）"
_SLOT_COL = "时段序号"
_DATE_COL = "日期"
_LOAD_COL_FORECAST = "全省实时负荷（MW）"


# ---------------------------------------------------------------------------
# Price recommendation engine
# ---------------------------------------------------------------------------

def build_price_recommendation(
    df_integrated: pd.DataFrame,
    quantiles: tuple = (0.75, 0.90, 0.95),
) -> pd.DataFrame:
    """Build period-level bidding price recommendations for generators.

    For each time period, the recommended bid price is the historical
    DA-price quantile at the specified percentile levels.  The rationale
    is that a generator should target prices above the historical median to
    improve revenue, anchored to the distribution of winning bids.

    Args:
        df_integrated: Integrated Excel DataFrame with period labels.
        quantiles: Quantile levels to compute (e.g. 0.75, 0.90, 0.95).

    Returns:
        DataFrame with columns: ['时段', 'hist_da_mean', 'hist_rt_mean',
        'mean_spread', 'q75_da', 'q90_da', 'q95_da',
        'rec_low_price', 'rec_mid_price', 'rec_high_price',
        'notes'] where rec_* prices are in ¥/MWh.
    """
    periods = ["低谷", "平段", "高峰"]
    rows = []

    for period in periods:
        mask = df_integrated["time_period"] == period
        da_sub = df_integrated.loc[mask, _DA_PRICE_COL].dropna()
        rt_sub = df_integrated.loc[mask, _RT_PRICE_COL].dropna()
        spread = (da_sub - rt_sub).dropna()

        hist_da_mean = float(da_sub.mean()) if not da_sub.empty else np.nan
        hist_rt_mean = float(rt_sub.mean()) if not rt_sub.empty else np.nan
        mean_spread = float(spread.mean()) if not spread.empty else np.nan

        q_vals = {
            f"q{int(q*100)}_da": float(da_sub.quantile(q)) if not da_sub.empty else np.nan
            for q in quantiles
        }

        # Recommended prices:
        # - Conservative (75th pct): capture majority of the market
        # - Moderate (90th pct): aggressive but still within winning range
        # - Aggressive (95th pct): reserve for scarcity events
        rec_low = q_vals.get("q75_da", np.nan)
        rec_mid = q_vals.get("q90_da", np.nan)
        rec_high = q_vals.get("q95_da", np.nan)

        # Notes per period
        if period == "低谷":
            notes = "低谷时段供大于求，建议适度降价保量，优先保证出清"
        elif period == "平段":
            notes = "平段适中申报，锁定稳定收益，避免被低价新能源排挤"
        else:  # 高峰
            notes = "高峰时段负荷高、竞价空间大，是申报高价的核心窗口"

        row = {
            "时段": period,
            "hist_da_mean(¥/MWh)": round(hist_da_mean, 2),
            "hist_rt_mean(¥/MWh)": round(hist_rt_mean, 2),
            "mean_spread(¥/MWh)": round(mean_spread, 2),
            **{k: round(v, 2) for k, v in q_vals.items()},
            "rec_conservative(¥/MWh)": round(rec_low, 2),
            "rec_moderate(¥/MWh)": round(rec_mid, 2),
            "rec_aggressive(¥/MWh)": round(rec_high, 2),
            "策略备注": notes,
        }
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Output guidance per period
# ---------------------------------------------------------------------------

def build_output_guidance(
    df_forecast: pd.DataFrame,
    df_integrated: pd.DataFrame,
) -> pd.DataFrame:
    """Build period-level output guidance for 2026-03-01.

    Args:
        df_forecast: Output of load_forecast_day()[0].
        df_integrated: Integrated Excel DataFrame.

    Returns:
        DataFrame with columns: ['时段', '预测负荷均值(MW)', '历史负荷均值(MW)',
        '负荷偏差(%)', '竞价空间均值(MW)', '推荐出力策略'].
    """
    periods = ["低谷", "平段", "高峰"]
    rows = []

    for period in periods:
        mask_fc = df_forecast["time_period"] == period
        mask_hist = df_integrated["time_period"] == period

        fc_load = df_forecast.loc[mask_fc, _LOAD_COL_FORECAST].mean() \
            if mask_fc.any() else np.nan
        hist_load = df_integrated.loc[mask_hist, _LOAD_COL].mean() \
            if mask_hist.any() else np.nan
        bid_space = df_integrated.loc[mask_hist, _BID_SPACE_COL].mean() \
            if _BID_SPACE_COL in df_integrated.columns and mask_hist.any() else np.nan

        diff_pct = (fc_load - hist_load) / hist_load * 100 \
            if (not np.isnan(fc_load) and not np.isnan(hist_load) and hist_load != 0) \
            else np.nan

        if period == "低谷":
            strategy = "降低出力 / 充电储能，避免低价出清拉低均价"
        elif period == "平段":
            strategy = "维持额定出力，稳健申报，确保日前中标"
        else:  # 高峰
            strategy = "满发并申报高价，把握峰时溢价窗口"

        rows.append({
            "时段": period,
            "预测负荷均值(MW)": round(fc_load, 1) if not np.isnan(fc_load) else np.nan,
            "历史负荷均值(MW)": round(hist_load, 1) if not np.isnan(hist_load) else np.nan,
            "负荷偏差(%)": round(diff_pct, 2) if not np.isnan(diff_pct) else np.nan,
            "竞价空间均值(MW)": round(bid_space, 1) if not np.isnan(bid_space) else np.nan,
            "推荐出力策略": strategy,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Storage / pumped-hydro arbitrage windows
# ---------------------------------------------------------------------------

def build_storage_arbitrage(df_integrated: pd.DataFrame) -> pd.DataFrame:
    """Identify charge (low-price) and discharge (high-price) windows.

    Args:
        df_integrated: Integrated Excel DataFrame with period labels.

    Returns:
        DataFrame with recommended charge/discharge windows, expected
        spread, and slot ranges.
    """
    periods_charge = ["低谷"]
    periods_discharge = ["高峰"]

    rows = []

    for period in periods_charge:
        mask = df_integrated["time_period"] == period
        rt_sub = df_integrated.loc[mask, _RT_PRICE_COL].dropna()
        slots = sorted(PERIOD_SLOTS.get(period, []))
        rows.append({
            "操作": "充电",
            "推荐时段": period,
            "时段序号范围": f"{min(slots)}–{max(slots)}" if slots else "-",
            "时间范围": _slots_to_time_range(slots),
            "预期RT均价(¥/MWh)": round(float(rt_sub.mean()), 2) if not rt_sub.empty else np.nan,
            "RT价格Q25(¥/MWh)": round(float(rt_sub.quantile(0.25)), 2) if not rt_sub.empty else np.nan,
            "RT价格Q75(¥/MWh)": round(float(rt_sub.quantile(0.75)), 2) if not rt_sub.empty else np.nan,
            "策略说明": "低谷廉价电力充电，降低综合用能成本",
        })

    for period in periods_discharge:
        mask = df_integrated["time_period"] == period
        rt_sub = df_integrated.loc[mask, _RT_PRICE_COL].dropna()
        slots = sorted(PERIOD_SLOTS.get(period, []))
        rows.append({
            "操作": "放电",
            "推荐时段": period,
            "时段序号范围": f"{min(slots)}–{max(slots)}" if slots else "-",
            "时间范围": _slots_to_time_range(slots),
            "预期RT均价(¥/MWh)": round(float(rt_sub.mean()), 2) if not rt_sub.empty else np.nan,
            "RT价格Q25(¥/MWh)": round(float(rt_sub.quantile(0.25)), 2) if not rt_sub.empty else np.nan,
            "RT价格Q75(¥/MWh)": round(float(rt_sub.quantile(0.75)), 2) if not rt_sub.empty else np.nan,
            "策略说明": "高峰高价放电，获取峰谷价差套利收益",
        })

    df = pd.DataFrame(rows)

    # Compute arbitrage spread between discharge mean and charge mean
    charge_rows = df[df["操作"] == "充电"]["预期RT均价(¥/MWh)"]
    discharge_rows = df[df["操作"] == "放电"]["预期RT均价(¥/MWh)"]
    if not charge_rows.empty and not discharge_rows.empty:
        spread = float(discharge_rows.mean()) - float(charge_rows.mean())
        df.attrs["expected_arbitrage_spread"] = round(spread, 2)

    return df


def _slots_to_time_range(slots: list) -> str:
    """Convert a list of 1-based slots to human-readable time ranges."""
    if not slots:
        return "-"
    ranges = []
    start = slots[0]
    prev = slots[0]
    for s in slots[1:]:
        if s != prev + 1:
            ranges.append(f"{_slot_to_time(start)}–{_slot_to_time(prev + 1)}")
            start = s
        prev = s
    ranges.append(f"{_slot_to_time(start)}–{_slot_to_time(prev + 1)}")
    return "、".join(ranges)


def _slot_to_time(slot: int) -> str:
    """Convert 1-based slot to HH:MM string (start of slot)."""
    minute = (slot - 1) * 15
    h, m = divmod(minute, 60)
    return f"{h:02d}:{m:02d}"


# ---------------------------------------------------------------------------
# Risk matrix
# ---------------------------------------------------------------------------

def build_risk_matrix() -> pd.DataFrame:
    """Build a qualitative risk matrix for the 2026-03-01 trading day.

    Returns:
        DataFrame with columns: ['风险类型', '发生概率', '价格影响',
        '风险等级', '应对措施'].
    """
    risks = [
        {
            "风险类型": "风电出力大幅上升",
            "发生概率": "中等",
            "价格影响": "实时价格下行 20–60 ¥/MWh",
            "风险等级": "中",
            "应对措施": "高峰时段调低实时申报价格至均值附近，避免不中标",
        },
        {
            "风险类型": "风电出力大幅下降",
            "发生概率": "低",
            "价格影响": "实时价格上行 30–80 ¥/MWh",
            "风险等级": "中",
            "应对措施": "保留容量参与实时市场，高价补充出清",
        },
        {
            "风险类型": "负荷超预期（极端天气）",
            "发生概率": "低",
            "价格影响": "高峰价格上行 50–150 ¥/MWh",
            "风险等级": "高（收益方向）",
            "应对措施": "提前锁定高峰申报价，充分利用竞价空间",
        },
        {
            "风险类型": "跨省联络线大量外送",
            "发生概率": "中等",
            "价格影响": "压低省内价格 10–40 ¥/MWh",
            "风险等级": "中",
            "应对措施": "关注联络线计划，低谷段降低申报价，减少竞争损失",
        },
        {
            "风险类型": "日前–实时价差反转",
            "发生概率": "低",
            "价格影响": "实时价高于日前 50+ ¥/MWh",
            "风险等级": "中（实时市场机会）",
            "应对措施": "在实时市场保留一定剩余容量，参与实时高价段",
        },
        {
            "风险类型": "竞价空间骤降",
            "发生概率": "低",
            "价格影响": "高价申报中标率下降",
            "风险等级": "低–中",
            "应对措施": "监控竞价空间公告，动态调整申报量",
        },
    ]
    return pd.DataFrame(risks)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _ensure_output_dir() -> None:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_bidding_strategy_table(
    price_rec: pd.DataFrame,
    output_guidance: pd.DataFrame,
    save: bool = True,
    show: bool = False,
) -> "plt.Figure":  # type: ignore[name-defined]
    """Render a visual strategy summary table.

    Args:
        price_rec: Output of build_price_recommendation().
        output_guidance: Output of build_output_guidance().
        save: Save to analysis/outputs/bidding_strategy_table.png.
        show: Display interactively.

    Returns:
        matplotlib Figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    axes[0].axis("off")
    axes[1].axis("off")

    # --- Price recommendation table ---
    cols_price = [
        "时段",
        "hist_da_mean(¥/MWh)",
        "hist_rt_mean(¥/MWh)",
        "q75_da(¥/MWh)",
        "q90_da(¥/MWh)",
        "rec_aggressive(¥/MWh)",
    ]
    available_price = [c for c in cols_price if c in price_rec.columns]
    tbl_data_price = price_rec[available_price].values.tolist()
    t1 = axes[0].table(
        cellText=[[str(round(v, 1)) if isinstance(v, float) else str(v) for v in row]
                  for row in tbl_data_price],
        colLabels=available_price,
        cellLoc="center",
        loc="center",
    )
    t1.auto_set_font_size(False)
    t1.set_fontsize(9)
    t1.scale(1, 1.8)
    axes[0].set_title("2026-03-01 发电侧竞标价格建议表", fontsize=12, fontweight="bold", pad=20)

    # --- Output guidance table ---
    cols_out = [
        "时段",
        "预测负荷均值(MW)",
        "历史负荷均值(MW)",
        "负荷偏差(%)",
        "竞价空间均值(MW)",
        "推荐出力策略",
    ]
    available_out = [c for c in cols_out if c in output_guidance.columns]
    tbl_data_out = output_guidance[available_out].values.tolist()
    t2 = axes[1].table(
        cellText=[[str(round(v, 1)) if isinstance(v, float) else str(v) for v in row]
                  for row in tbl_data_out],
        colLabels=available_out,
        cellLoc="center",
        loc="center",
    )
    t2.auto_set_font_size(False)
    t2.set_fontsize(9)
    t2.scale(1, 1.8)
    axes[1].set_title("2026-03-01 各时段出力指导", fontsize=12, fontweight="bold", pad=20)

    # Row colouring by period
    colours = {"低谷": "#deebf7", "平段": "#e5f5e0", "高峰": "#fee8c8"}
    for tbl in (t1, t2):
        for (row, col), cell in tbl.get_celld().items():
            if row == 0:
                cell.set_facecolor("#cccccc")
                cell.set_text_props(fontweight="bold")
            else:
                period = tbl_data_price[row - 1][0] \
                    if tbl is t1 else tbl_data_out[row - 1][0]
                cell.set_facecolor(colours.get(str(period), "white"))

    plt.suptitle("江西省电力现货市场 — 2026-03-01 发电侧竞标策略",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save:
        _ensure_output_dir()
        fig.savefig(_OUTPUT_DIR / "bidding_strategy_table.png", dpi=150,
                    bbox_inches="tight")
    if show:
        plt.show()

    return fig


def plot_risk_matrix(
    risk_df: pd.DataFrame,
    save: bool = True,
    show: bool = False,
) -> "plt.Figure":  # type: ignore[name-defined]
    """Render the risk matrix as a formatted table.

    Args:
        risk_df: Output of build_risk_matrix().
        save: Save to analysis/outputs/risk_matrix.png.
        show: Display interactively.

    Returns:
        matplotlib Figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")

    tbl = ax.table(
        cellText=risk_df.values.tolist(),
        colLabels=risk_df.columns.tolist(),
        cellLoc="left",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 2)
    tbl.auto_set_column_width(list(range(len(risk_df.columns))))

    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4d4d4d")
            cell.set_text_props(color="white", fontweight="bold")
        elif "高" in str(risk_df.iloc[row - 1]["风险等级"]) and "低" not in str(risk_df.iloc[row - 1]["风险等级"]):
            cell.set_facecolor("#fee8c8")
        elif "低" in str(risk_df.iloc[row - 1]["风险等级"]):
            cell.set_facecolor("#e5f5e0")
        else:
            cell.set_facecolor("#fff7bc")

    ax.set_title("2026-03-01 交易风险矩阵", fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()

    if save:
        _ensure_output_dir()
        fig.savefig(_OUTPUT_DIR / "risk_matrix.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Main — full strategy report
# ---------------------------------------------------------------------------

def print_strategy_report() -> None:
    """Print a complete quantitative bidding strategy report for 2026-03-01."""
    separator = "=" * 65

    print(separator)
    print("  江西省电力现货市场 — 2026-03-01 发电侧竞标策略建议")
    print(separator)

    df_int, df_da, df_rt = load_all_data()
    df_forecast, fc_stats = load_forecast_day()

    # ---- Price recommendations ------------------------------------------
    price_rec = build_price_recommendation(df_int)
    print("\n【一】时段级申报价格建议 (¥/MWh)")
    print("-" * 65)
    for _, row in price_rec.iterrows():
        print(
            f"  {row['时段']:4s}  "
            f"历史DA均值={row['hist_da_mean(¥/MWh)']:>7.2f}  "
            f"历史RT均值={row['hist_rt_mean(¥/MWh)']:>7.2f}  "
            f"价差={row['mean_spread(¥/MWh)']:>+7.2f}"
        )
        for label, col in [
            ("保守(Q75)", "rec_conservative(¥/MWh)"),
            ("适中(Q90)", "rec_moderate(¥/MWh)"),
            ("激进(Q95)", "rec_aggressive(¥/MWh)"),
        ]:
            if col in row.index:
                print(f"         → {label}: {row[col]:>8.2f} ¥/MWh")
        print(f"         备注: {row['策略备注']}")

    # ---- Output guidance ------------------------------------------------
    output_guidance = build_output_guidance(df_forecast, df_int)
    print("\n【二】各时段出力指导")
    print("-" * 65)
    print(output_guidance.to_string(index=False))

    # ---- Load stats for forecast day ------------------------------------
    print("\n【三】预测日负荷特征")
    print("-" * 65)
    print(f"  最高负荷  : {fc_stats['max_load']:>10,.2f} MW  (时段 {fc_stats['peak_slot']})")
    print(f"  最低负荷  : {fc_stats['min_load']:>10,.2f} MW  (时段 {fc_stats['valley_slot']})")
    print(f"  峰谷差    : {fc_stats['peak_valley_diff']:>10,.2f} MW")
    print(f"  高峰均负荷: {fc_stats['per_period'].get('高峰', {}).get('mean', 0):>10,.2f} MW")

    # ---- Bidding space --------------------------------------------------
    bid_space_peak = df_int.loc[
        df_int["time_period"] == "高峰", "竞价空间（MW）"
    ].mean() if "竞价空间（MW）" in df_int.columns else np.nan
    bid_space_valley = df_int.loc[
        df_int["time_period"] == "低谷", "竞价空间（MW）"
    ].mean() if "竞价空间（MW）" in df_int.columns else np.nan

    print("\n【四】竞价空间利用")
    print("-" * 65)
    if not np.isnan(bid_space_peak):
        print(f"  高峰平均竞价空间: {bid_space_peak:>8,.1f} MW")
        print(f"  低谷平均竞价空间: {bid_space_valley:>8,.1f} MW")
        print("  建议: 高峰时段充分利用竞价空间申报高价，低谷段降量保价")

    # ---- Storage arbitrage ---------------------------------------------
    arb_df = build_storage_arbitrage(df_int)
    print("\n【五】储能/抽蓄套利建议")
    print("-" * 65)
    print(arb_df.to_string(index=False))
    if "expected_arbitrage_spread" in arb_df.attrs:
        print(f"\n  预期峰谷套利空间: {arb_df.attrs['expected_arbitrage_spread']:+.2f} ¥/MWh")

    # ---- Risk matrix ---------------------------------------------------
    risk_df = build_risk_matrix()
    print("\n【六】风险提示矩阵")
    print("-" * 65)
    for _, r in risk_df.iterrows():
        print(f"  [{r['风险等级']}] {r['风险类型']}")
        print(f"         概率: {r['发生概率']}  影响: {r['价格影响']}")
        print(f"         应对: {r['应对措施']}")

    print(f"\n{separator}")
    print("  生成可视化图表 …")
    plot_bidding_strategy_table(price_rec, output_guidance, save=True)
    plot_risk_matrix(risk_df, save=True)
    print(f"  图表已保存至 {_REPO_ROOT / 'analysis' / 'outputs'}")
    print(separator)


def main() -> None:
    """Entry point — run the full bidding strategy report."""
    print_strategy_report()


if __name__ == "__main__":
    main()
