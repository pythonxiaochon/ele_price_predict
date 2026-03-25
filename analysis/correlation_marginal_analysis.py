"""
相关性与边际分析模块
Correlation and Marginal Analysis Module

Analyses:
    1. Pearson / Spearman correlations: load → RT price, temperature → price,
       wind speed → price, all computed per time period.
    2. New-energy marginal impact: (wind + solar) total output vs RT price,
       piecewise linear regression, price sensitivity in ¥/MWh per GWh.
    3. Thermal-power clearing characteristics by load quantile.

Reads:
    data/整合型 Excel.xlsx          : All 15-min records.
    data/25年10月-26年2月日前现货出清.xls
    data/25年10月-26年2月实时现货出清.xls

Outputs (when run as __main__):
    - Console: correlation matrix, regression coefficients.
    - Figures (saved to analysis/outputs/):
        correlation_matrix.png      : Heatmap of correlation coefficients.
        load_price_scatter.png      : Load vs RT price by period.
        renewables_price_scatter.png: Renewables vs RT price + regression.
        thermal_load_quantile.png   : Thermal clearing by load quantile.

Public API:
    load_merged_data()            → merged DataFrame
    compute_correlations()        → correlation DataFrame
    renewables_marginal_impact()  → regression dict
    thermal_clearing_by_quantile()→ DataFrame
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from analysis.time_period_labeling import add_period_label
from analysis.historical_price_stats import load_all_data

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT_DIR = _REPO_ROOT / "analysis" / "outputs"

# Integrated Excel column names
_SLOT_COL = "时段序号"
_DATE_COL = "日期"
_RT_PRICE_COL = "实时出清价格（元/兆瓦时）"
_DA_PRICE_COL = "日前出清价格（元 / 兆瓦时）"
_LOAD_COL = "全省实时负荷"
_TEMP_COL = "全省平均气温"
_WIND_COL = "风速"
_SOLAR_COL = "光伏出清电量"
_THERMAL_COL = "火电出清电量"
_HYDRO_COL = "水电出清电量"
_TIE_LINE_COL = "跨省联络线功率（MW）"
_BID_SPACE_COL = "竞价空间（MW）"
_TOTAL_VOL_COL = "总出清电量"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_merged_data() -> pd.DataFrame:
    """Load and merge the integrated Excel with DA/RT clearing details.

    The wind-power column comes from the clearing files (风电出清电量), while
    the solar column is already in the integrated file (光伏出清电量).

    Returns:
        Merged DataFrame with period labels and a 'renewables_mwh' column
        (wind + solar clearing volume, in MWh).
    """
    df_int, df_da, df_rt = load_all_data()

    # Aggregate RT clearing wind/solar to 15-min slots to match integrated file
    rt_agg = (
        df_rt.groupby([_DATE_COL, "slot"])
        .agg(rt_wind_mwh=("风电出清电量", "sum"), rt_solar_mwh=("光伏出清电量", "sum"))
        .reset_index()
        .rename(columns={"slot": _SLOT_COL})
    )
    rt_agg[_DATE_COL] = pd.to_datetime(rt_agg[_DATE_COL])

    merged = df_int.merge(rt_agg, on=[_DATE_COL, _SLOT_COL], how="left")

    # Wind from RT clearing; solar from integrated file
    merged["renewables_mwh"] = (
        merged["rt_wind_mwh"].fillna(0) + merged[_SOLAR_COL].fillna(0)
    )

    return merged


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

def compute_correlations(
    df: pd.DataFrame,
    target_col: str = _RT_PRICE_COL,
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute Pearson and Spearman correlations between features and target.

    Correlations are computed globally and per time period.

    Args:
        df: Merged DataFrame with 'time_period' column.
        target_col: Dependent variable (default: RT clearing price).
        feature_cols: Feature columns to correlate.  Defaults to a standard set.

    Returns:
        DataFrame with columns: ['特征', '全局_pearson', '全局_spearman',
        '低谷_pearson', '低谷_spearman', '平段_pearson', '平段_spearman',
        '高峰_pearson', '高峰_spearman'].
    """
    if feature_cols is None:
        feature_cols = [
            _LOAD_COL, _TEMP_COL, _WIND_COL,
            _BID_SPACE_COL, _TIE_LINE_COL, "renewables_mwh",
        ]
        feature_cols = [c for c in feature_cols if c in df.columns]

    periods = ["低谷", "平段", "高峰"]
    rows = []

    for feat in feature_cols:
        row = {"特征": feat}
        sub_all = df[[feat, target_col]].dropna()

        # Global correlations
        if len(sub_all) > 2:
            row["全局_pearson"] = float(sub_all[feat].corr(sub_all[target_col]))
            row["全局_spearman"] = float(
                scipy_stats.spearmanr(sub_all[feat], sub_all[target_col])[0]
            )
        else:
            row["全局_pearson"] = np.nan
            row["全局_spearman"] = np.nan

        # Per-period correlations
        for period in periods:
            mask = df["time_period"] == period
            sub = df.loc[mask, [feat, target_col]].dropna()
            if len(sub) > 2:
                row[f"{period}_pearson"] = float(sub[feat].corr(sub[target_col]))
                row[f"{period}_spearman"] = float(
                    scipy_stats.spearmanr(sub[feat], sub[target_col])[0]
                )
            else:
                row[f"{period}_pearson"] = np.nan
                row[f"{period}_spearman"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Renewables marginal impact
# ---------------------------------------------------------------------------

def renewables_marginal_impact(
    df: pd.DataFrame,
    n_segments: int = 3,
) -> Dict:
    """Estimate the marginal price impact of renewables using piecewise OLS.

    For each time period, fit *n_segments* linear regression segments on
    (renewables_mwh → RT price) after sorting by renewables output.

    Args:
        df: Merged DataFrame with 'renewables_mwh' and RT price columns.
        n_segments: Number of piecewise segments (quantile-based breakpoints).

    Returns:
        Dict keyed by period ('低谷', '平段', '高峰') each containing:
            'overall_slope'  : Global OLS slope (¥/MWh per MWh of renewables).
            'overall_r2'     : R² of global OLS.
            'segments'       : List of dicts with 'range', 'slope', 'intercept'.
            'price_per_gwh'  : Estimated price drop per GWh of renewables (¥/MWh).
    """
    from sklearn.linear_model import LinearRegression  # local import

    periods = ["低谷", "平段", "高峰"]
    results: Dict = {}

    for period in periods:
        mask = df["time_period"] == period
        sub = df.loc[mask, ["renewables_mwh", _RT_PRICE_COL]].dropna()
        sub = sub.sort_values("renewables_mwh").reset_index(drop=True)

        if len(sub) < 10:
            results[period] = {
                "overall_slope": np.nan,
                "overall_r2": np.nan,
                "segments": [],
                "price_per_gwh": np.nan,
            }
            continue

        X = sub["renewables_mwh"].values.reshape(-1, 1)
        y = sub[_RT_PRICE_COL].values

        # Global OLS
        model_global = LinearRegression().fit(X, y)
        y_pred = model_global.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        # Piecewise by quantile breakpoints
        breakpoints = [
            sub["renewables_mwh"].quantile(q)
            for q in np.linspace(0, 1, n_segments + 1)
        ]
        segments = []
        for i in range(n_segments):
            lo, hi = breakpoints[i], breakpoints[i + 1]
            seg_mask = (sub["renewables_mwh"] >= lo) & (sub["renewables_mwh"] <= hi)
            seg = sub[seg_mask]
            if len(seg) < 3:
                continue
            Xs = seg["renewables_mwh"].values.reshape(-1, 1)
            ys = seg[_RT_PRICE_COL].values
            m = LinearRegression().fit(Xs, ys)
            segments.append({
                "range": (float(lo), float(hi)),
                "slope": float(m.coef_[0]),
                "intercept": float(m.intercept_),
            })

        results[period] = {
            "overall_slope": float(model_global.coef_[0]),
            "overall_r2": float(r2),
            "segments": segments,
            # Convert slope from ¥/MWh per MWh → ¥/MWh per GWh
            "price_per_gwh": float(model_global.coef_[0]) * 1000,
        }

    return results


# ---------------------------------------------------------------------------
# Thermal clearing by load quantile
# ---------------------------------------------------------------------------

def thermal_clearing_by_quantile(
    df: pd.DataFrame,
    quantiles: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Summarise thermal clearing volume and price at load quantile buckets.

    Args:
        df: Merged DataFrame.
        quantiles: Load quantile breakpoints.  Defaults to [0.25, 0.50, 0.75, 0.90].

    Returns:
        DataFrame with columns: ['负荷分位', '负荷下限(MW)', '负荷上限(MW)',
        '火电均量(MWh)', '火电均价(¥/MWh)', '记录数'].
    """
    if quantiles is None:
        quantiles = [0.25, 0.50, 0.75, 0.90]

    load = df[_LOAD_COL].dropna()
    breakpoints = [load.quantile(0)] + [load.quantile(q) for q in quantiles] + [load.quantile(1)]

    rows = []
    labels = ["Q0–Q25", "Q25–Q50", "Q50–Q75", "Q75–Q90", "Q90–Q100"]
    for i in range(len(breakpoints) - 1):
        lo, hi = breakpoints[i], breakpoints[i + 1]
        label = labels[i] if i < len(labels) else f"Q{i}"
        mask = (df[_LOAD_COL] >= lo) & (df[_LOAD_COL] <= hi)
        sub = df[mask]
        rows.append({
            "负荷分位": label,
            "负荷下限(MW)": round(lo, 1),
            "负荷上限(MW)": round(hi, 1),
            "火电均量(MWh)": round(sub[_THERMAL_COL].mean(), 2) if _THERMAL_COL in sub else np.nan,
            "DA均价(¥/MWh)": round(sub[_DA_PRICE_COL].mean(), 2) if _DA_PRICE_COL in sub else np.nan,
            "RT均价(¥/MWh)": round(sub[_RT_PRICE_COL].mean(), 2) if _RT_PRICE_COL in sub else np.nan,
            "记录数": int(mask.sum()),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _ensure_output_dir() -> None:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_correlation_matrix(
    corr_df: pd.DataFrame,
    save: bool = True,
    show: bool = False,
) -> "plt.Figure":  # type: ignore[name-defined]
    """Heatmap of Pearson correlation coefficients (global + per period).

    Args:
        corr_df: Output of compute_correlations().
        save: Save to analysis/outputs/correlation_matrix.png.
        show: Display interactively.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pearson_cols = [c for c in corr_df.columns if c.endswith("_pearson")]
    matrix = corr_df.set_index("特征")[pearson_cols].astype(float)
    matrix.columns = [c.replace("_pearson", "") for c in pearson_cols]

    fig, ax = plt.subplots(figsize=(10, max(4, len(matrix) * 0.8)))
    im = ax.imshow(matrix.values, cmap="RdYlGn", aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.04)

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, fontsize=10)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=9)

    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            val = matrix.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                        color="black")

    ax.set_title("Pearson相关系数矩阵（实时出清价格为目标）", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save:
        _ensure_output_dir()
        fig.savefig(_OUTPUT_DIR / "correlation_matrix.png", dpi=150)
    if show:
        plt.show()

    return fig


def plot_load_price_scatter(
    df: pd.DataFrame,
    save: bool = True,
    show: bool = False,
) -> "plt.Figure":  # type: ignore[name-defined]
    """Scatter plot of load vs RT price, coloured by period.

    Args:
        df: Merged DataFrame with 'time_period'.
        save: Save to analysis/outputs/load_price_scatter.png.
        show: Display interactively.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    period_colours = {"低谷": "#6baed6", "平段": "#74c476", "高峰": "#fd8d3c"}

    fig, ax = plt.subplots(figsize=(9, 6))

    for period, colour in period_colours.items():
        mask = df["time_period"] == period
        sub = df[mask].dropna(subset=[_LOAD_COL, _RT_PRICE_COL])
        if sub.empty:
            continue
        ax.scatter(sub[_LOAD_COL], sub[_RT_PRICE_COL],
                   c=colour, alpha=0.25, s=6, label=period)

        # OLS trend line
        x = sub[_LOAD_COL].values
        y = sub[_RT_PRICE_COL].values
        if len(x) > 2:
            coef = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, np.polyval(coef, x_line), color=colour, linewidth=1.5)

    ax.set_xlabel("全省实时负荷 (MW)", fontsize=11)
    ax.set_ylabel("实时出清价格 (元/MWh)", fontsize=11)
    ax.set_title("负荷–实时价格散点图（时段着色）", fontsize=13, fontweight="bold")
    handles = [mpatches.Patch(color=c, label=p) for p, c in period_colours.items()]
    ax.legend(handles=handles, fontsize=10)
    ax.grid(linestyle="--", alpha=0.3)
    plt.tight_layout()

    if save:
        _ensure_output_dir()
        fig.savefig(_OUTPUT_DIR / "load_price_scatter.png", dpi=150)
    if show:
        plt.show()

    return fig


def plot_renewables_price_scatter(
    df: pd.DataFrame,
    impact_results: Dict,
    save: bool = True,
    show: bool = False,
) -> "plt.Figure":  # type: ignore[name-defined]
    """Scatter plot of renewables output vs RT price with regression lines.

    Args:
        df: Merged DataFrame with 'renewables_mwh' and 'time_period'.
        impact_results: Output of renewables_marginal_impact().
        save: Save to analysis/outputs/renewables_price_scatter.png.
        show: Display interactively.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    periods = ["低谷", "平段", "高峰"]
    colours = ["#6baed6", "#74c476", "#fd8d3c"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, period, colour in zip(axes, periods, colours):
        mask = df["time_period"] == period
        sub = df[mask].dropna(subset=["renewables_mwh", _RT_PRICE_COL])

        if not sub.empty:
            ax.scatter(sub["renewables_mwh"], sub[_RT_PRICE_COL],
                       c=colour, alpha=0.25, s=6)

        res = impact_results.get(period, {})
        slope = res.get("overall_slope", np.nan)
        r2 = res.get("overall_r2", np.nan)
        price_per_gwh = res.get("price_per_gwh", np.nan)

        if not sub.empty and not np.isnan(slope):
            x_range = np.linspace(sub["renewables_mwh"].min(),
                                  sub["renewables_mwh"].max(), 100)
            # Use overall_slope + mean intercept approximation
            y_mean = sub[_RT_PRICE_COL].mean()
            x_mean = sub["renewables_mwh"].mean()
            y_line = slope * (x_range - x_mean) + y_mean
            ax.plot(x_range, y_line, "r-", linewidth=1.8,
                    label=f"斜率={slope:.3f}\nR²={r2:.3f}")
            ax.legend(fontsize=8)

        subtitle = ""
        if not np.isnan(price_per_gwh):
            subtitle = f"\n新能源增加1GWh → 价格变化 {price_per_gwh:+.2f} ¥/MWh"
        ax.set_title(f"{period}{subtitle}", fontsize=10, fontweight="bold")
        ax.set_xlabel("新能源出清电量 (MWh)", fontsize=9)
        ax.set_ylabel("实时出清价格 (元/MWh)", fontsize=9)
        ax.grid(linestyle="--", alpha=0.3)

    fig.suptitle("新能源出力–实时价格散点图（分段线性回归）", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save:
        _ensure_output_dir()
        fig.savefig(_OUTPUT_DIR / "renewables_price_scatter.png", dpi=150)
    if show:
        plt.show()

    return fig


def plot_thermal_load_quantile(
    quantile_df: pd.DataFrame,
    save: bool = True,
    show: bool = False,
) -> "plt.Figure":  # type: ignore[name-defined]
    """Bar chart of thermal clearing volume at each load quantile.

    Args:
        quantile_df: Output of thermal_clearing_by_quantile().
        save: Save to analysis/outputs/thermal_load_quantile.png.
        show: Display interactively.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(10, 5))

    x = range(len(quantile_df))
    bars = ax1.bar(x, quantile_df["火电均量(MWh)"], color="#fc8d59", alpha=0.75,
                   label="火电均量(MWh)")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(quantile_df["负荷分位"], fontsize=10)
    ax1.set_ylabel("火电出清电量均值 (MWh)", fontsize=10)
    ax1.legend(loc="upper left", fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(list(x), quantile_df["RT均价(¥/MWh)"], "b-o", linewidth=1.8, label="RT均价")
    ax2.plot(list(x), quantile_df["DA均价(¥/MWh)"], "g--s", linewidth=1.5, label="DA均价")
    ax2.set_ylabel("价格 (元/MWh)", fontsize=10)
    ax2.legend(loc="upper right", fontsize=9)

    ax1.set_title("不同负荷分位下的火电出清电量与价格", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    if save:
        _ensure_output_dir()
        fig.savefig(_OUTPUT_DIR / "thermal_load_quantile.png", dpi=150)
    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Main — standalone execution
# ---------------------------------------------------------------------------

def main() -> None:
    """Run correlation and marginal analysis and print results."""
    print("=" * 60)
    print("  相关性与边际分析")
    print("=" * 60)

    df = load_merged_data()
    print(f"\n  合并数据: {len(df):,} 行")

    print("\n  相关性分析（目标: 实时出清价格）:")
    corr_df = compute_correlations(df)
    print(corr_df.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    print("\n  新能源边际价格影响:")
    impact = renewables_marginal_impact(df)
    for period, res in impact.items():
        slope = res.get("overall_slope", np.nan)
        r2 = res.get("overall_r2", np.nan)
        ppg = res.get("price_per_gwh", np.nan)
        print(
            f"  {period}: 斜率={slope:+.4f} ¥/MWh per MWh  "
            f"R²={r2:.3f}  "
            f"→ 新能源增加1GWh价格变化 {ppg:+.2f} ¥/MWh"
        )

    print("\n  不同负荷分位下的火电出清特征:")
    quantile_df = thermal_clearing_by_quantile(df)
    print(quantile_df.to_string(index=False))

    print("\n  生成分析图表 …")
    plot_correlation_matrix(corr_df, save=True)
    plot_load_price_scatter(df, save=True)
    plot_renewables_price_scatter(df, impact, save=True)
    plot_thermal_load_quantile(quantile_df, save=True)
    print(f"  图表已保存至 {_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
