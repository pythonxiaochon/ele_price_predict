"""
2026年3月1日全省96点负荷预测
Provincial 96-Point Load Forecasting for 2026-03-01

算法：LightGBM 主模型 + 残差滚动修正
Architecture: LightGBM primary model with rolling residual correction

用法:
    python predict_march_load.py

可选滚动修正:
    在脚本末尾调用 rolling_correction(new_actual_df) 传入新的真实数据即可。
"""

import warnings
import os
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")           # 无界面环境使用非交互后端
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error
from data.preprocess import align_15min_index

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. 常量与配置
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = "data"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 预测目标日期
TARGET_DATE = pd.Timestamp("2026-03-01")

# 异常时期（春节前后及恢复期，此段数据负荷严重偏低，不用于主模型训练）
# 2026年春节：1月28日，延长至春节前1周（1月21日）及节后恢复期（2月28日）
HOLIDAY_EXCLUDE_START = pd.Timestamp("2026-01-21")
HOLIDAY_EXCLUDE_END = pd.Timestamp("2026-02-28")  # 排除至2月底（含）

# LightGBM 超参数
LGBM_PARAMS: Dict = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "n_estimators": 800,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}

# 用于分位数回归（置信区间）
LGBM_Q10_PARAMS: Dict = {**LGBM_PARAMS, "objective": "quantile", "alpha": 0.10}
LGBM_Q90_PARAMS: Dict = {**LGBM_PARAMS, "objective": "quantile", "alpha": 0.90}

# 江西省2026年3月峰谷平尖峰时段定义（时段序号1-96，每15分钟一个时段）
# 尖峰: 10:00-12:00, 18:00-21:00 → 槽位41-48, 73-84
# 峰:   08:00-10:00, 12:00-18:00, 21:00-22:00 → 槽位33-40, 49-72, 85-88
# 谷:   00:00-08:00, 22:00-24:00 → 槽位1-32, 89-96
# 平:   其余（此处无剩余，按实际规则调整）
# 注：实际执行以电网公司官方文件为准，此处按常见规则定义
def get_period_label(slot: int) -> str:
    """根据时段序号（1-96）返回峰谷平尖峰标签。

    Args:
        slot: 时段序号，1-96（对应00:00-23:45，步长15分钟）。

    Returns:
        峰谷平尖峰标签字符串：'尖峰' / '峰' / '平' / '谷'
    """
    # 转换为小时（slot 1 = 00:00-00:15，slot 96 = 23:45-24:00）
    hour = (slot - 1) * 15 / 60  # 浮点小时，如 slot=41 → 10.0h

    if (10.0 <= hour < 12.0) or (18.0 <= hour < 21.0):
        return "尖峰"
    elif (8.0 <= hour < 10.0) or (12.0 <= hour < 18.0) or (21.0 <= hour < 22.0):
        return "峰"
    elif (0.0 <= hour < 8.0) or (22.0 <= hour < 24.0):
        return "谷"
    else:
        return "平"


# 江西省2026年节假日（春节、元旦等，3月无法定节假日但3月8日妇女节不放假）
HOLIDAYS_2026 = {
    "2026-01-01",  # 元旦
    "2026-01-28", "2026-01-29", "2026-01-30", "2026-01-31",
    "2026-02-01", "2026-02-02", "2026-02-03",  # 春节
    "2026-04-05",  # 清明
    "2026-05-01", "2026-05-02", "2026-05-03",  # 劳动节
    "2026-06-19",  # 端午
    "2026-09-20", "2026-09-21", "2026-09-22",  # 中秋
    "2026-10-01", "2026-10-02", "2026-10-03", "2026-10-04",
    "2026-10-05", "2026-10-06", "2026-10-07",  # 国庆
}
HOLIDAYS_2025 = {
    "2025-01-01",  # 元旦
    "2025-01-28", "2025-01-29", "2025-01-30", "2025-01-31",
    "2025-02-01", "2025-02-02", "2025-02-03", "2025-02-04",  # 春节
    "2025-04-04", "2025-04-05", "2025-04-06",  # 清明
    "2025-05-01", "2025-05-02", "2025-05-03", "2025-05-04", "2025-05-05",  # 劳动节
    "2025-05-31",  # 端午
    "2025-10-01", "2025-10-02", "2025-10-03", "2025-10-04",
    "2025-10-05", "2025-10-06", "2025-10-07",  # 国庆
}
ALL_HOLIDAYS = HOLIDAYS_2025 | HOLIDAYS_2026


# ─────────────────────────────────────────────────────────────────────────────
# 2. 数据读取函数
# ─────────────────────────────────────────────────────────────────────────────

def load_historical_data() -> pd.DataFrame:
    """读取整合型Excel.xlsx中的历史负荷、气温、风速数据（2025-10至2026-02）。

    Returns:
        DataFrame，以DatetimeIndex（15分钟频率）为索引，包含负荷、气温、风速等列。
    """
    path = os.path.join(DATA_DIR, "整合型 Excel.xlsx")
    df = pd.read_excel(path, sheet_name="Sheet1")

    # 构建时间戳：日期 + 时段序号（1-96 对应 00:00:00 - 23:45:00）
    df["datetime"] = pd.to_datetime(df["日期"]) + pd.to_timedelta(
        (df["时段序号"] - 1) * 15, unit="min"
    )
    df = df.set_index("datetime").sort_index()

    # 重命名关键列，保持一致性
    df = df.rename(columns={
        "全省实时负荷": "load_mw",
        "全省平均气温": "temperature",
        "风速": "wind_speed",
        "竞价空间（MW）": "bid_space",
        "实时出清价格（元/兆瓦时）": "rt_price",
        "日前出清价格（元 / 兆瓦时）": "da_price",
    })

    # 保留核心列（其余列作为可选扩展特征）
    core_cols = ["load_mw", "temperature", "wind_speed", "bid_space", "rt_price", "da_price",
                 "时段序号"]
    available = [c for c in core_cols if c in df.columns]
    df = df[available].copy()

    # 过滤明显异常值（负荷为负或极端值）
    df = df[df["load_mw"] > 0].copy()

    # 补全缺失时段（使用前向填充）：确保15分钟等间隔，让滞后特征定位正确
    df = align_15min_index(df, fill_method="ffill")
    # 补全后再次过滤极端异常值（填充可能引入）
    df = df[df["load_mw"] > 0].copy()

    print(f"[数据] 历史数据加载完毕: {df.index.min()} → {df.index.max()}, 共 {len(df)} 行")
    return df


def load_forecast_data() -> pd.DataFrame:
    """读取系统负荷预测文件，提取官方预测值作为特征。

    Returns:
        DataFrame，以DatetimeIndex（15分钟频率）为索引，包含官方预测负荷值。
    """
    path = os.path.join(DATA_DIR, "25年10月-26年2月系统负荷预测.xls")
    df = pd.read_excel(path)

    # 构建时间戳：执行日期 + 时刻点字符串（如 "00:15"）
    # 处理特殊值 "24:00"：替换为次日 "00:00"
    time_str = df["时刻点"].astype(str)
    date_str = df["执行日期"].astype(str)

    # 对 "24:00" 处理：转换为次日 00:00（在合并前替换为 "00:00"，日期加1天）
    mask_24 = time_str == "24:00"
    combined = date_str + " " + time_str.where(~mask_24, "00:00")
    dt_series = pd.to_datetime(combined)
    dt_series = dt_series.where(~mask_24.values, dt_series + pd.Timedelta(days=1))
    df["datetime"] = dt_series
    df = df.set_index("datetime").sort_index()
    df = df.rename(columns={"预测值": "official_forecast"})
    df = df[["official_forecast"]].copy()

    # 时刻点是当前时段末（00:15代表第1时段），与历史数据时刻点（时段开始）对齐
    # 历史数据: slot1 → 00:00, 预测数据: slot1 → 00:15，需向前平移15分钟
    df.index = df.index - pd.Timedelta(minutes=15)

    print(f"[数据] 官方预测数据加载完毕: {df.index.min()} → {df.index.max()}, 共 {len(df)} 行")
    return df


def load_real_load() -> pd.DataFrame:
    """读取2026-03-01真实负荷数据（用于验证和滚动修正）。

    Returns:
        DataFrame，以DatetimeIndex（15分钟频率）为索引，包含实际负荷列。
    """
    path = os.path.join(DATA_DIR, "真实负荷.xlsx")
    df = pd.read_excel(path)

    df["datetime"] = pd.to_datetime(df["预测日"]) + pd.to_timedelta(
        (df["时段序号"] - 1) * 15, unit="min"
    )
    df = df.set_index("datetime").sort_index()
    df = df.rename(columns={"全省实时负荷（MW）": "actual_load"})
    df = df[["actual_load"]].copy()

    print(f"[数据] 真实负荷数据加载完毕: {df.index.min()} → {df.index.max()}, 共 {len(df)} 行")
    return df


def load_weather_forecast() -> pd.DataFrame:
    """读取2026-03-01天气预报数据（气温、风速）。

    Returns:
        DataFrame，以DatetimeIndex（15分钟频率）为索引，包含气温和风速列。
    """
    path = os.path.join(DATA_DIR, "预测气象输入.xlsx")
    df = pd.read_excel(path)

    df["datetime"] = pd.to_datetime(df["预测日"]) + pd.to_timedelta(
        (df["时段序号"] - 1) * 15, unit="min"
    )
    df = df.set_index("datetime").sort_index()
    df = df.rename(columns={"气温(℃)": "temperature", "风速(m/s)": "wind_speed"})
    df = df[["temperature", "wind_speed"]].copy()

    print(f"[数据] 气象预报数据加载完毕: {df.index.min()} → {df.index.max()}, 共 {len(df)} 行")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. 特征工程
# ─────────────────────────────────────────────────────────────────────────────

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加日历特征：时段、小时、星期、月份、节假日、峰谷平标签。

    Args:
        df: 具有 DatetimeIndex 的 DataFrame。

    Returns:
        添加了日历特征列的新 DataFrame。
    """
    out = df.copy()
    idx = out.index

    out["time_slot"] = idx.hour * 4 + idx.minute // 15      # 0-95
    out["slot_num"] = out["time_slot"] + 1                   # 1-96（与原始数据对应）
    out["hour"] = idx.hour
    out["minute"] = idx.minute
    out["dayofweek"] = idx.dayofweek                          # 0=周一 … 6=周日
    out["month"] = idx.month
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)

    holiday_set = {pd.Timestamp(d).date() for d in ALL_HOLIDAYS}
    out["is_holiday"] = out.index.to_series().apply(lambda d: int(d.date() in holiday_set))
    out["is_workday"] = ((out["is_weekend"] == 0) & (out["is_holiday"] == 0)).astype(int)

    # 峰谷平尖峰标签（2026年3月江西省规则）
    period_map = {"尖峰": 3, "峰": 2, "平": 1, "谷": 0}
    out["period_label"] = out["slot_num"].apply(get_period_label)
    out["period_code"] = out["period_label"].map(period_map)

    # 一天中的正弦/余弦编码（捕捉时间周期性）
    out["time_sin"] = np.sin(2 * np.pi * out["time_slot"] / 96)
    out["time_cos"] = np.cos(2 * np.pi * out["time_slot"] / 96)

    # 一周中的正弦/余弦编码
    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7)

    return out


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加负荷和气象的滞后特征（1h/1d/2d/3d/7d）。

    Args:
        df: 包含 load_mw、temperature、wind_speed 列的 DataFrame。

    Returns:
        添加了滞后特征的 DataFrame。
    """
    out = df.copy()

    # 负荷滞后：前1个时段、前1小时(4)、前1天(96)、前2天、前3天、前7天
    load_lags = [1, 4, 96, 192, 288, 672]
    for lag in load_lags:
        out[f"load_lag{lag}"] = out["load_mw"].shift(lag)

    # 气温滞后：前1天、前7天
    for lag in [96, 672]:
        out[f"temp_lag{lag}"] = out["temperature"].shift(lag)

    # 风速滞后：前1天
    out["wind_lag96"] = out["wind_speed"].shift(96)

    # 滚动统计：过去1小时和过去1天的均值/标准差
    for window in [4, 96]:
        roller = out["load_mw"].rolling(window=window, min_periods=1)
        out[f"load_roll{window}_mean"] = roller.mean()
        out[f"load_roll{window}_std"] = roller.std().fillna(0)

    # 同时段前7天负荷均值（同一时段的历史规律）
    out["load_same_slot_7d_mean"] = (
        out.groupby("time_slot")["load_mw"]
        .transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    )

    return out


def add_forecast_features(
    df: pd.DataFrame, forecast_df: pd.DataFrame
) -> pd.DataFrame:
    """合并官方预测值，并计算日前预测偏差特征及残差目标列。

    残差预测思路：
        模型预测目标改为 load_residual = load_mw - official_forecast，
        而非直接预测绝对负荷值。这样可消除负荷水平绝对值带来的偏差，
        使模型更鲁棒地处理春节恢复期后的负荷跃升问题。

    Args:
        df: 历史负荷 DataFrame（已含 load_mw）。
        forecast_df: 官方预测 DataFrame（含 official_forecast）。

    Returns:
        合并了预测偏差特征的 DataFrame。
    """
    out = df.join(forecast_df, how="left")

    # 日前预测残差：实际负荷 - 官方预测（作为训练目标，通常在 ±3000 MW 以内）
    out["load_residual"] = out["load_mw"] - out["official_forecast"]

    # 滞后残差特征（反映最近几天官方预测的系统性偏差）
    out["residual_lag96"] = out["load_residual"].shift(96)     # 昨日同时段残差
    out["residual_lag192"] = out["load_residual"].shift(192)   # 前天同时段残差
    out["residual_lag288"] = out["load_residual"].shift(288)   # 大前天残差

    # 官方预测的环比比率（捕捉负荷水平的阶跃变化）
    # 当比值远偏离1时（如春节恢复期），提示模型负荷将发生大幅变化
    lag96_safe = out["load_lag96"].replace(0, np.nan)
    out["forecast_to_lag96_ratio"] = (out["official_forecast"] / lag96_safe).fillna(1.0)
    out["forecast_to_lag96_ratio"] = out["forecast_to_lag96_ratio"].clip(0.3, 3.0)

    # 官方预测滞后（作为额外特征）
    out["official_forecast_lag96"] = out["official_forecast"].shift(96)

    return out


def build_feature_matrix(
    hist_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
) -> pd.DataFrame:
    """完整特征工程流水线：日历 → 滞后 → 预测偏差 → 残差特征。

    Args:
        hist_df: 历史负荷数据（DatetimeIndex，15分钟频率）。
        forecast_df: 官方负荷预测数据。

    Returns:
        特征完整的 DataFrame，已删除含 NaN 的初始行。
    """
    df = add_calendar_features(hist_df)
    df = add_lag_features(df)
    df = add_forecast_features(df, forecast_df)

    # 删除由滞后特征产生的 NaN 行（前7天数据）
    df = df.dropna(subset=[c for c in df.columns if "lag" in c or "roll" in c])

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. 模型训练
# ─────────────────────────────────────────────────────────────────────────────

# 用于训练的特征列（不含目标列 load_residual 和泄漏列）
# 
# 关键设计决策：移除所有滞后负荷特征（load_lag*）
# 理由：3月1日的所有滞后特征均指向春节恢复期（1月21日-2月28日），
# 该时段负荷极度异常（3000-9000 MW），远低于正常运行水平（14000-20000 MW），
# 会严重干扰模型预测。仅保留以下可靠特征：
#   (1) 官方预测（正确反映当日负荷水平）
#   (2) 日历/时段特征（峰谷平尖峰规律）
#   (3) 气象特征（温度、风速）
#   (4) 官方预测的滞后和比率（捕捉预测系统性偏差）
FEATURE_COLS = [
    "time_slot", "hour", "minute", "dayofweek", "month",
    "is_weekend", "is_holiday", "is_workday",
    "period_code",
    "time_sin", "time_cos", "dow_sin", "dow_cos",
    "temperature", "wind_speed",
    "temp_lag96", "temp_lag672",
    "official_forecast",
    "official_forecast_lag96",
]

# 训练目标列
TARGET_COL = "load_residual"  # 预测残差，再加上 official_forecast 得到最终负荷


def get_available_features(df: pd.DataFrame) -> List[str]:
    """返回在当前 DataFrame 中实际存在的特征列。"""
    return [c for c in FEATURE_COLS if c in df.columns]


def train_lgbm_model(
    feat_df: pd.DataFrame,
    params: Dict,
    val_fraction: float = 0.1,
) -> lgb.LGBMRegressor:
    """训练 LightGBM 回归模型（预测 load_residual = load_mw - official_forecast）。

    Args:
        feat_df: 含所有特征和目标列 load_residual 的 DataFrame。
        params: LightGBM 超参数字典。
        val_fraction: 用于早停验证集的比例（按时间顺序从训练末尾取）。

    Returns:
        已训练的 LGBMRegressor 实例。
    """
    feature_cols = get_available_features(feat_df)
    X = feat_df[feature_cols]
    # 训练目标：残差（实际负荷 - 官方预测），约在 ±3000 MW 范围
    y = feat_df[TARGET_COL].values

    split = int(len(X) * (1 - val_fraction))
    X_tr, X_val = X.iloc[:split], X.iloc[split:]
    y_tr, y_val = y[:split], y[split:]

    p = {k: v for k, v in params.items() if k != "n_estimators"}
    n_est = params.get("n_estimators", 800)

    model = lgb.LGBMRegressor(n_estimators=n_est, **p)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    return model


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """计算 MAPE 和 MAE 指标。

    Args:
        y_true: 真实值数组。
        y_pred: 预测值数组。

    Returns:
        包含 MAPE 和 MAE 的字典。
    """
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"MAPE(%)": round(mape, 3), "MAE(MW)": round(mae, 2), "RMSE(MW)": round(rmse, 2)}


# ─────────────────────────────────────────────────────────────────────────────
# 5. 滑动窗口交叉验证
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate(
    feat_df: pd.DataFrame,
    n_splits: int = 5,
    test_days: int = 7,
) -> Tuple[List[Dict], float, float]:
    """滑动窗口时间序列交叉验证。

    Args:
        feat_df: 已完成特征工程的 DataFrame。
        n_splits: 折数。
        test_days: 每折测试窗口天数。

    Returns:
        (cv_results, mean_mape, mean_mae) 元组。
    """
    feature_cols = get_available_features(feat_df)
    results = []

    total_days = (feat_df.index.max() - feat_df.index.min()).days
    available_days = total_days - test_days * n_splits
    if available_days < 7:
        print("[警告] 数据量不足以支持所有折，将减少折数")
        n_splits = max(1, total_days // test_days - 1)

    min_date = feat_df.index.min()

    for fold in range(n_splits):
        test_start = min_date + pd.Timedelta(days=available_days + fold * test_days)
        test_end = test_start + pd.Timedelta(days=test_days)

        train_mask = feat_df.index < test_start
        test_mask = (feat_df.index >= test_start) & (feat_df.index < test_end)

        if train_mask.sum() < 96 * 7 or test_mask.sum() < 96:
            continue

        X_tr = feat_df.loc[train_mask, feature_cols]
        y_tr = feat_df.loc[train_mask, TARGET_COL].values   # 训练目标：残差
        X_te = feat_df.loc[test_mask, feature_cols]
        y_te_residual = feat_df.loc[test_mask, TARGET_COL].values
        y_te_actual = feat_df.loc[test_mask, "load_mw"].values
        official_te = feat_df.loc[test_mask, "official_forecast"].values

        split = int(len(X_tr) * 0.9)
        p = {k: v for k, v in LGBM_PARAMS.items() if k != "n_estimators"}
        n_est = LGBM_PARAMS.get("n_estimators", 800)
        model = lgb.LGBMRegressor(n_estimators=n_est, **p)
        model.fit(
            X_tr.iloc[:split], y_tr[:split],
            eval_set=[(X_tr.iloc[split:], y_tr[split:])],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(period=-1)],
        )
        # 残差预测 → 转换回实际负荷值（残差 + 官方预测）
        pred_residuals = model.predict(X_te)
        preds_load = pred_residuals + official_te
        metrics = compute_metrics(y_te_actual, preds_load)
        metrics["fold"] = fold + 1
        metrics["test_start"] = str(test_start.date())
        metrics["test_end"] = str(test_end.date())
        results.append(metrics)
        print(f"  折{fold+1} | MAPE={metrics['MAPE(%)']:.2f}% | "
              f"MAE={metrics['MAE(MW)']:.1f} MW | "
              f"{metrics['test_start']} → {metrics['test_end']}")

    if results:
        mean_mape = float(np.mean([r["MAPE(%)"] for r in results]))
        mean_mae = float(np.mean([r["MAE(MW)"] for r in results]))
    else:
        mean_mape = mean_mae = float("nan")

    return results, mean_mape, mean_mae


# ─────────────────────────────────────────────────────────────────────────────
# 6. 预测2026年3月1日
# ─────────────────────────────────────────────────────────────────────────────

def build_prediction_row(
    hist_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    weather_forecast_df: pd.DataFrame,
    official_forecast_march1: pd.Series,
) -> pd.DataFrame:
    """为2026-03-01构建预测特征矩阵（使用滞后历史数据+气象预报）。

    思路：将3月1日的气象预报和官方预测拼接到历史数据末尾，再统一做滞后特征，
    从而让3月1日的滞后特征指向2月28日及之前的真实历史数据。

    Args:
        hist_df: 历史负荷数据（截至2026-02-28）。
        forecast_df: 官方预测数据。
        weather_forecast_df: 2026-03-01 气象预报数据。
        official_forecast_march1: 2026-03-01 官方负荷预测（Series，index为datetime）。

    Returns:
        仅含2026-03-01 96行的特征 DataFrame（load_mw列为 NaN，其余为特征值）。
    """
    # 创建3月1日占位行（负荷未知，用 NaN 占位）
    march1_index = pd.date_range("2026-03-01 00:00", periods=96, freq="15min")
    march1_df = pd.DataFrame(index=march1_index)
    march1_df["load_mw"] = np.nan
    march1_df["时段序号"] = range(1, 97)

    # 填入气象预报
    march1_df["temperature"] = weather_forecast_df["temperature"].values
    march1_df["wind_speed"] = weather_forecast_df["wind_speed"].values

    # 其余列（bid_space, rt_price, da_price）填 NaN，不用于预测
    for col in ["bid_space", "rt_price", "da_price"]:
        if col in hist_df.columns:
            march1_df[col] = np.nan

    # 拼接历史数据和3月1日占位数据
    combined_df = pd.concat([hist_df, march1_df], axis=0)
    combined_df = combined_df[~combined_df.index.duplicated(keep="last")]
    combined_df = combined_df.sort_index()

    # 填充官方预测（历史 + 3月1日）
    combined_forecast = pd.concat([
        forecast_df,
        official_forecast_march1.rename("official_forecast").to_frame(),
    ]).sort_index()
    combined_forecast = combined_forecast[~combined_forecast.index.duplicated(keep="last")]

    # 特征工程（同历史数据流程）
    feat = add_calendar_features(combined_df)
    feat = add_lag_features(feat)
    feat = add_forecast_features(feat, combined_forecast)

    # 只返回3月1日的行
    march1_feat = feat[feat.index.date == TARGET_DATE.date()].copy()

    # 对预测期可能存在的少量 NaN 特征进行向前填充（用前一时段值补充）
    feature_cols_to_fill = get_available_features(march1_feat)
    march1_feat[feature_cols_to_fill] = (
        march1_feat[feature_cols_to_fill].ffill().bfill()
    )

    return march1_feat


def predict_march1(
    main_model: lgb.LGBMRegressor,
    q10_model: lgb.LGBMRegressor,
    q90_model: lgb.LGBMRegressor,
    march1_feat: pd.DataFrame,
) -> pd.DataFrame:
    """使用训练好的三个模型对3月1日进行点预测和区间预测。

    模型预测的是 load_residual = load_mw - official_forecast，
    最终负荷 = official_forecast + predicted_residual。

    Args:
        main_model: 主回归模型（均值预测残差）。
        q10_model: 10%分位数模型（置信区间下界残差）。
        q90_model: 90%分位数模型（置信区间上界残差）。
        march1_feat: 3月1日特征矩阵（96行）。

    Returns:
        含预测值和置信区间的 DataFrame。
    """
    feature_cols = get_available_features(march1_feat)
    official_fc = march1_feat["official_forecast"].values

    X = march1_feat[feature_cols]
    # 模型输出为残差，需加回官方预测得到绝对负荷
    pred_residual_main = main_model.predict(X)
    pred_residual_q10 = q10_model.predict(X)
    pred_residual_q90 = q90_model.predict(X)

    pred_main = pred_residual_main + official_fc
    pred_q10 = pred_residual_q10 + official_fc
    pred_q90 = pred_residual_q90 + official_fc

    result = pd.DataFrame({
        "datetime": march1_feat.index,
        "时段序号": march1_feat["slot_num"].astype(int),
        "时间": march1_feat.index.strftime("%H:%M"),
        "峰谷标签": march1_feat["period_label"],
        "官方预测(MW)": np.round(official_fc, 2),
        "预测残差(MW)": np.round(pred_residual_main, 2),
        "预测负荷(MW)": np.round(pred_main, 2),
        "置信下界_P10(MW)": np.round(pred_q10, 2),
        "置信上界_P90(MW)": np.round(pred_q90, 2),
    })
    result = result.set_index("datetime")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 7. 滚动修正函数
# ─────────────────────────────────────────────────────────────────────────────

def rolling_correction(
    prediction_df: pd.DataFrame,
    new_actual_df: pd.DataFrame,
    hist_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    weather_forecast_df: pd.DataFrame,
    official_forecast_march1: pd.Series,
    correction_method: str = "residual_smoothing",
) -> pd.DataFrame:
    """滚动修正：当真实数据到达后，对尚未发生的时段预测进行修正。

    支持两种修正策略：
    - "residual_smoothing": 基于已知时段的残差，通过平滑外推修正未来时段预测。
    - "retrain": 将新真实数据加入训练集，重新训练模型后重新预测。

    Args:
        prediction_df: 初始预测结果 DataFrame（含"预测负荷(MW)"列）。
        new_actual_df: 新到真实负荷数据（DataFrame，含"actual_load"列，
                        index为datetime，只需包含已知时段即可）。
        hist_df: 原始历史负荷数据。
        forecast_df: 官方预测数据。
        weather_forecast_df: 气象预报数据。
        official_forecast_march1: 官方3月1日负荷预测。
        correction_method: 修正策略，"residual_smoothing" 或 "retrain"。

    Returns:
        修正后的预测 DataFrame（格式同 prediction_df）。
    """
    corrected = prediction_df.copy()
    known_idx = new_actual_df.index.intersection(prediction_df.index)

    if len(known_idx) == 0:
        print("[滚动修正] 无已知真实数据，跳过修正。")
        return corrected

    # 计算已知时段的残差（真实 - 预测）
    actual_values = new_actual_df.loc[known_idx, "actual_load"]
    predicted_values = prediction_df.loc[known_idx, "预测负荷(MW)"]
    residuals = actual_values.values - predicted_values.values

    print(f"[滚动修正] 已接收 {len(known_idx)} 个时段的真实数据")
    print(f"  已知时段平均残差: {residuals.mean():.2f} MW")
    print(f"  已知时段残差标准差: {residuals.std():.2f} MW")

    # 未来时段（尚未有真实数据的时段）
    future_idx = prediction_df.index.difference(known_idx)

    if len(future_idx) == 0:
        print("[滚动修正] 所有时段均已有真实数据，无需预测修正。")
        # 仍将真实值填入预测列，用于完整性
        corrected.loc[known_idx, "预测负荷(MW)"] = actual_values.values
        return corrected

    if correction_method == "residual_smoothing":
        # 策略1：残差平滑外推
        # 思路：取最近N个已知残差的加权均值作为未来修正量，权重随时间衰减
        n_recent = min(len(residuals), 16)  # 最近4小时（16个时段）
        recent_residuals = residuals[-n_recent:]
        weights = np.exp(-np.arange(n_recent - 1, -1, -1) * 0.1)  # 指数加权
        weights /= weights.sum()
        correction_value = float(np.dot(weights, recent_residuals))

        print(f"  残差平滑修正量: {correction_value:.2f} MW（加权近期残差）")

        # 对未来时段施加修正（修正量随时间衰减）
        decay = np.exp(-np.arange(len(future_idx)) * 0.02)  # 衰减因子
        corrections = correction_value * decay

        corrected.loc[future_idx, "预测负荷(MW)"] = (
            prediction_df.loc[future_idx, "预测负荷(MW)"].values + corrections
        )
        corrected.loc[future_idx, "置信下界_P10(MW)"] = (
            prediction_df.loc[future_idx, "置信下界_P10(MW)"].values + corrections
        )
        corrected.loc[future_idx, "置信上界_P90(MW)"] = (
            prediction_df.loc[future_idx, "置信上界_P90(MW)"].values + corrections
        )

    elif correction_method == "retrain":
        # 策略2：重新训练（将新真实数据加入历史集）
        print("[滚动修正] 重新训练模型（策略：retrain）...")
        new_hist_data = new_actual_df.copy().rename(columns={"actual_load": "load_mw"})
        # 补充气象数据
        new_hist_data["temperature"] = weather_forecast_df["temperature"]
        new_hist_data["wind_speed"] = weather_forecast_df["wind_speed"]
        for col in ["bid_space", "rt_price", "da_price"]:
            if col in hist_df.columns:
                new_hist_data[col] = np.nan
        new_hist_data["时段序号"] = range(1, len(new_hist_data) + 1)

        updated_hist = pd.concat([hist_df, new_hist_data]).sort_index()
        updated_hist = updated_hist[~updated_hist.index.duplicated(keep="last")]

        updated_feat = build_feature_matrix(updated_hist, forecast_df)
        updated_feat = updated_feat.dropna(subset=["load_mw"])

        new_main = train_lgbm_model(updated_feat, LGBM_PARAMS)
        new_q10 = train_lgbm_model(updated_feat, LGBM_Q10_PARAMS)
        new_q90 = train_lgbm_model(updated_feat, LGBM_Q90_PARAMS)

        new_march1_feat = build_prediction_row(
            updated_hist, forecast_df, weather_forecast_df, official_forecast_march1
        )
        corrected = predict_march1(new_main, new_q10, new_q90, new_march1_feat)
        # 将已知真实值填回
        corrected.loc[known_idx, "预测负荷(MW)"] = actual_values.values

    # 更新已知时段为真实值
    corrected.loc[known_idx, "预测负荷(MW)"] = actual_values.values
    print(f"[滚动修正] 修正完成，未来 {len(future_idx)} 个时段已调整。")

    return corrected


# ─────────────────────────────────────────────────────────────────────────────
# 8. 可视化
# ─────────────────────────────────────────────────────────────────────────────

PERIOD_COLORS = {"尖峰": "#d62728", "峰": "#ff7f0e", "平": "#2ca02c", "谷": "#1f77b4"}


def plot_prediction(
    prediction_df: pd.DataFrame,
    actual_df: Optional[pd.DataFrame] = None,
    corrected_df: Optional[pd.DataFrame] = None,
    save_path: str = "output/march1_load_prediction.png",
) -> None:
    """绘制96点负荷预测曲线图（含置信区间和真实值对比）。

    Args:
        prediction_df: 初始预测结果 DataFrame。
        actual_df: 真实负荷 DataFrame（可选，用于对比）。
        corrected_df: 滚动修正后预测 DataFrame（可选）。
        save_path: 图片保存路径。
    """
    fig, axes = plt.subplots(2, 1, figsize=(18, 12), height_ratios=[3, 1])

    ax1 = axes[0]
    times = prediction_df.index

    # 置信区间阴影
    ax1.fill_between(
        times,
        prediction_df["置信下界_P10(MW)"],
        prediction_df["置信上界_P90(MW)"],
        alpha=0.15, color="#1f77b4", label="80% 置信区间 (P10-P90)",
    )

    # 初始预测曲线
    ax1.plot(
        times, prediction_df["预测负荷(MW)"],
        color="#1f77b4", linewidth=2.0, linestyle="--",
        label="LightGBM 初始预测", zorder=5,
    )

    # 滚动修正曲线
    if corrected_df is not None:
        ax1.plot(
            times, corrected_df["预测负荷(MW)"],
            color="#9467bd", linewidth=2.0, linestyle="-.",
            label="滚动修正后预测", zorder=6,
        )

    # 真实值曲线
    if actual_df is not None:
        common = times.intersection(actual_df.index)
        ax1.plot(
            common, actual_df.loc[common, "actual_load"],
            color="#d62728", linewidth=2.5,
            label="实际负荷（真实值）", zorder=7,
        )

    # 峰谷区间背景色
    for _, row in prediction_df.iterrows():
        color = PERIOD_COLORS.get(row["峰谷标签"], "#aec7e8")
        ax1.axvspan(row.name, row.name + pd.Timedelta(minutes=15),
                    alpha=0.05, color=color)

    ax1.set_title("2026年3月1日全省96点负荷预测（LightGBM + 残差滚动修正）",
                  fontsize=14, fontweight="bold", pad=12)
    ax1.set_ylabel("负荷（MW）", fontsize=12)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax1.tick_params(axis="x", rotation=45)
    ax1.legend(loc="upper left", fontsize=10, framealpha=0.85)
    ax1.grid(True, alpha=0.3)

    # 峰谷图例色块
    from matplotlib.patches import Patch
    period_patches = [Patch(color=c, alpha=0.4, label=label)
                      for label, c in PERIOD_COLORS.items()]
    ax1.legend(handles=ax1.get_legend_handles_labels()[0] + period_patches,
               labels=ax1.get_legend_handles_labels()[1] + list(PERIOD_COLORS.keys()),
               loc="upper left", fontsize=9, framealpha=0.85, ncol=2)

    # 子图2：预测误差（如有真实值）
    ax2 = axes[1]
    if actual_df is not None:
        common = times.intersection(actual_df.index)
        error = (
            prediction_df.loc[common, "预测负荷(MW)"]
            - actual_df.loc[common, "actual_load"]
        )
        ax2.bar(common, error, width=0.008, color=np.where(error >= 0, "#ff7f0e", "#1f77b4"),
                alpha=0.7, label="预测误差 = 预测 - 实际")
        ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax2.set_ylabel("误差（MW）", fontsize=11)
        ax2.set_title("逐时段预测误差", fontsize=12)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax2.tick_params(axis="x", rotation=45)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[图表] 预测曲线图已保存至: {save_path}")


def plot_feature_importance(
    model: lgb.LGBMRegressor,
    feature_cols: List[str],
    top_n: int = 20,
    save_path: str = "output/feature_importance.png",
) -> None:
    """绘制特征重要性图。"""
    imp = model.feature_importances_
    feat_imp = pd.Series(imp, index=feature_cols).sort_values(ascending=False)[:top_n]

    fig, ax = plt.subplots(figsize=(10, 6))
    feat_imp[::-1].plot(kind="barh", ax=ax, color="#1f77b4", alpha=0.8)
    ax.set_title(f"特征重要性 Top-{top_n}", fontsize=13)
    ax.set_xlabel("重要性得分", fontsize=11)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[图表] 特征重要性图已保存至: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """完整预测流程：数据加载 → 特征工程 → 训练 → 预测 → 修正 → 可视化。"""

    print("=" * 65)
    print("  2026年3月1日全省96点负荷预测（LightGBM + 残差滚动修正）")
    print("=" * 65)

    # ── 步骤1：读取数据 ────────────────────────────────────────────────────
    print("\n【步骤1】读取数据文件...")
    hist_df = load_historical_data()
    forecast_df = load_forecast_data()
    real_load_df = load_real_load()
    weather_forecast_df = load_weather_forecast()

    # 提取3月1日官方预测（作为预测特征）
    official_forecast_march1 = forecast_df[
        forecast_df.index.date == TARGET_DATE.date()
    ]["official_forecast"]
    print(f"  3月1日官方预测槽位数: {len(official_forecast_march1)}")

    # ── 步骤2：特征工程 ────────────────────────────────────────────────────
    print("\n【步骤2】特征工程...")
    feat_df = build_feature_matrix(hist_df, forecast_df)
    # 仅保留负荷有效的行（训练集，排除3月1日占位行）
    train_feat = feat_df[feat_df["load_mw"].notna()].copy()

    # 排除春节及恢复期的异常数据（此段负荷严重偏低，影响模型对正常运行日的预测）
    # 仅保留春节前的正常运行数据（含恢复期后的数据在训练中不可靠）
    normal_mask = train_feat.index < HOLIDAY_EXCLUDE_START
    train_feat_normal = train_feat[normal_mask].copy()
    # 移除 load_residual 为 NaN 的行（官方预测缺失时无法计算残差）
    train_feat_normal = train_feat_normal[train_feat_normal[TARGET_COL].notna()].copy()
    print(f"  全量训练样本: {len(train_feat)} 行 ({train_feat.index.min().date()} → {train_feat.index.max().date()})")
    print(f"  正常期训练样本（春节前数据）: {len(train_feat_normal)} 行")
    print(f"  排除时段: {HOLIDAY_EXCLUDE_START.date()} 及之后（春节+恢复期）")

    feature_cols = get_available_features(train_feat_normal)
    print(f"  特征列数: {len(feature_cols)}")

    # ── 步骤3：交叉验证评估 ───────────────────────────────────────────────
    print("\n【步骤3】滑动窗口交叉验证（5折，每折7天，仅在正常运行期）...")
    cv_results, mean_mape, mean_mae = cross_validate(train_feat_normal, n_splits=5, test_days=7)
    print(f"\n  交叉验证均值 MAPE: {mean_mape:.3f}%")
    print(f"  交叉验证均值 MAE: {mean_mae:.1f} MW")

    # ── 步骤4：在全量历史数据上训练最终模型 ──────────────────────────────
    print("\n【步骤4】训练最终主模型（正常运行期历史数据，排除春节及恢复期）...")
    main_model = train_lgbm_model(train_feat_normal, LGBM_PARAMS)
    print("  训练分位数模型（P10/P90 置信区间）...")
    q10_model = train_lgbm_model(train_feat_normal, LGBM_Q10_PARAMS)
    q90_model = train_lgbm_model(train_feat_normal, LGBM_Q90_PARAMS)
    print("  三个模型训练完成。")

    # 历史数据上的整体指标（在正常期上评估，残差 + 官方预测 = 最终负荷）
    hist_pred_residuals = main_model.predict(train_feat_normal[feature_cols])
    hist_official = train_feat_normal["official_forecast"].values
    hist_pred_loads = hist_pred_residuals + hist_official
    hist_metrics = compute_metrics(train_feat_normal["load_mw"].values, hist_pred_loads)
    print(f"\n  历史数据拟合指标（正常运行期训练集）:")
    for k, v in hist_metrics.items():
        print(f"    {k}: {v}")

    # ── 步骤5：构建3月1日特征并预测 ──────────────────────────────────────
    print("\n【步骤5】构建2026-03-01预测特征矩阵...")
    march1_feat = build_prediction_row(
        hist_df, forecast_df, weather_forecast_df, official_forecast_march1
    )
    print(f"  3月1日特征矩阵: {march1_feat.shape}")

    print("\n【步骤6】预测2026-03-01全省96点负荷...")
    prediction_df = predict_march1(main_model, q10_model, q90_model, march1_feat)
    print(f"  预测完成，共 {len(prediction_df)} 个时段")

    # ── 步骤6：与真实值对比（验证精度）──────────────────────────────────
    print("\n【步骤7】与真实负荷数据对比（2026-03-01）...")
    common_idx = prediction_df.index.intersection(real_load_df.index)
    if len(common_idx) > 0:
        y_true = real_load_df.loc[common_idx, "actual_load"].values
        y_pred = prediction_df.loc[common_idx, "预测负荷(MW)"].values
        real_metrics = compute_metrics(y_true, y_pred)
        print(f"  3月1日预测误差指标:")
        for k, v in real_metrics.items():
            print(f"    {k}: {v}")
    else:
        print("  无法与真实值对比（索引不匹配）")
        real_metrics = {}

    # ── 步骤7：不确定性分析 ───────────────────────────────────────────────
    print("\n【步骤8】不确定性分析...")
    interval_width = (
        prediction_df["置信上界_P90(MW)"] - prediction_df["置信下界_P10(MW)"]
    )
    print(f"  80% 置信区间平均宽度: {interval_width.mean():.1f} MW")
    print(f"  最宽区间（时段）: {interval_width.idxmax().strftime('%H:%M')} "
          f"({interval_width.max():.1f} MW)")
    print(f"  最窄区间（时段）: {interval_width.idxmin().strftime('%H:%M')} "
          f"({interval_width.min():.1f} MW)")

    # 按峰谷分析
    print("\n  按峰谷时段的预测统计:")
    for period in ["尖峰", "峰", "平", "谷"]:
        mask = prediction_df["峰谷标签"] == period
        if mask.any():
            mean_pred = prediction_df.loc[mask, "预测负荷(MW)"].mean()
            mean_width = (
                prediction_df.loc[mask, "置信上界_P90(MW)"]
                - prediction_df.loc[mask, "置信下界_P10(MW)"]
            ).mean()
            print(f"    {period}: 均值={mean_pred:.0f} MW, 区间宽度={mean_width:.0f} MW")

    # ── 步骤8：滚动修正演示 ───────────────────────────────────────────────
    print("\n【步骤9】滚动修正演示（使用3月1日前24个时段真实数据修正后续预测）...")
    corrected_metrics: Dict = {}   # 修正后指标（若有真实数据则填充）
    # 模拟：取前24个时段（6小时）的真实数据进行修正
    known_slots = 24
    partial_actual = real_load_df.iloc[:known_slots].copy()
    corrected_df = rolling_correction(
        prediction_df=prediction_df,
        new_actual_df=partial_actual,
        hist_df=hist_df,
        forecast_df=forecast_df,
        weather_forecast_df=weather_forecast_df,
        official_forecast_march1=official_forecast_march1,
        correction_method="residual_smoothing",
    )

    # 修正后误差
    if len(common_idx) > 0:
        y_pred_corrected = corrected_df.loc[common_idx, "预测负荷(MW)"].values
        corrected_metrics = compute_metrics(y_true, y_pred_corrected)
        print(f"\n  修正后预测误差指标（全96点）:")
        for k, v in corrected_metrics.items():
            print(f"    {k}: {v}")

    # ── 步骤9：输出结果表格 ───────────────────────────────────────────────
    print("\n【步骤10】输出预测结果...")

    # 完整结果表（含真实值对比）
    output_table = prediction_df.copy()
    if len(common_idx) > 0:
        output_table["实际负荷(MW)"] = real_load_df.loc[
            output_table.index.intersection(real_load_df.index), "actual_load"
        ]
        output_table["预测误差(MW)"] = (
            output_table["预测负荷(MW)"] - output_table["实际负荷(MW)"]
        )
        output_table["MAPE(%)"] = np.abs(
            output_table["预测误差(MW)"] / output_table["实际负荷(MW)"] * 100
        ).round(3)

    # 添加修正后预测
    output_table["修正后预测(MW)"] = corrected_df["预测负荷(MW)"].round(2)

    # 保存 Excel
    excel_path = os.path.join(OUTPUT_DIR, "march1_load_prediction.xlsx")
    output_table.to_excel(excel_path, sheet_name="3月1日负荷预测")
    print(f"  预测结果表已保存至: {excel_path}")

    # 打印前几行
    print("\n  预测结果（前10行）:")
    preview_cols = ["时段序号", "时间", "峰谷标签", "预测负荷(MW)",
                    "置信下界_P10(MW)", "置信上界_P90(MW)"]
    if "实际负荷(MW)" in output_table.columns:
        preview_cols += ["实际负荷(MW)", "MAPE(%)"]
    print(output_table[preview_cols].head(10).to_string())

    # ── 步骤10：可视化 ────────────────────────────────────────────────────
    print("\n【步骤11】生成可视化图表...")
    plot_prediction(
        prediction_df=prediction_df,
        actual_df=real_load_df if len(common_idx) > 0 else None,
        corrected_df=corrected_df,
        save_path=os.path.join(OUTPUT_DIR, "march1_load_prediction.png"),
    )
    plot_feature_importance(
        model=main_model,
        feature_cols=feature_cols,
        save_path=os.path.join(OUTPUT_DIR, "feature_importance.png"),
    )

    # ── 汇总指标 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  模型综合指标汇总")
    print("=" * 65)
    print(f"  历史数据CV均值 MAPE    : {mean_mape:.3f}%")
    print(f"  历史数据CV均值 MAE     : {mean_mae:.1f} MW")
    if real_metrics:
        print(f"  3月1日预测 MAPE        : {real_metrics.get('MAPE(%)', 'N/A')}%")
        print(f"  3月1日预测 MAE         : {real_metrics.get('MAE(MW)', 'N/A')} MW")
        print(f"  3月1日预测 RMSE        : {real_metrics.get('RMSE(MW)', 'N/A')} MW")
    if len(common_idx) > 0 and corrected_metrics:
        print(f"  3月1日修正后 MAPE      : {corrected_metrics.get('MAPE(%)', 'N/A')}%")
        print(f"  3月1日修正后 MAE       : {corrected_metrics.get('MAE(MW)', 'N/A')} MW")
    print(f"\n  80% 置信区间平均宽度   : {interval_width.mean():.1f} MW")
    print("\n  输出文件:")
    print(f"    预测结果表: {excel_path}")
    print(f"    预测曲线图: output/march1_load_prediction.png")
    print(f"    特征重要性: output/feature_importance.png")
    print("\n" + "=" * 65)


if __name__ == "__main__":
    # 使用Agg后端时需配置中文字体
    try:
        plt.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei",
                                            "DejaVu Sans", "Arial Unicode MS"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    main()
