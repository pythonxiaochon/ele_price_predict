# 模型改进建议 — `models/load_forecasting.py`
# Model Improvement Recommendations

> 目标：提升江西省电力现货市场日前96点负荷预测精度，同时为发电侧价格竞标策略提供更可靠的输入。

---

## 一、特征工程增强

### 1.1 集成时段标签（低谷/平段/高峰）

当前模型仅使用时钟特征（hour, minute, dayofweek）。引入从 `analysis/time_period_labeling.py` 生成的时段标签，可以让模型直接感知电价结构差异。

```python
# 在 data/preprocess.py 的 add_time_features() 中新增
from analysis.time_period_labeling import add_period_label

def add_time_features(df, holiday_dates=None):
    out = ... # 现有逻辑
    # 新增：时段标签（低谷=0, 平段=1, 高峰=2）
    out = add_period_label(out, slot_col="time_slot_1based")
    period_map = {"低谷": 0, "平段": 1, "高峰": 2}
    out["time_period_enc"] = out["time_period"].map(period_map)
    # 新增：交叉特征
    out["period_x_hour"] = out["time_period_enc"] * out["hour"]
    return out
```

**所需数据列**：`time_slot`（0-95），通过 `slot + 1` 映射到1-96时段标签。  
**预期提升**：峰时段负荷预测 MAPE 降低 0.5–1.5%。

---

### 1.2 竞价空间特征

竞价空间（MW）反映当天可调度容量，与系统紧张程度高度相关。

```python
# 数据列：竞价空间（MW）→ 归一化后作为特征
# 推荐 lag：96步（前一天同时刻）、192步（前两天）
feat = add_lag_features(feat, ["竞价空间（MW）"], lags=[96, 192])
feat = add_rolling_features(feat, ["竞价空间（MW）"], windows=[96], stats=["mean"])
```

**所需数据列**：`整合型 Excel.xlsx` → `竞价空间（MW）`  
**预期提升**：有助于区分紧缺日与宽松日，高峰段 RMSE 降低 50–100 MW。

---

### 1.3 前日价格反馈特征

引入日前价格作为反馈特征，帮助模型感知前一天的市场紧张程度。

```python
# 推荐 lag 步数：96（前一天同时刻）
feat = add_lag_features(
    feat,
    columns=["日前出清价格（元 / 兆瓦时）", "实时出清价格（元/兆瓦时）"],
    lags=[96, 192],
)
# 价差特征
feat["da_rt_spread_lag96"] = (
    feat["日前出清价格（元 / 兆瓦时）_lag96"]
    - feat["实时出清价格（元/兆瓦时）_lag96"]
)
```

**所需数据列**：`日前出清价格（元 / 兆瓦时）`、`实时出清价格（元/兆瓦时）`  
**预期提升**：价格反馈特征对高峰时段尤为重要，R² 有望提升 0.01–0.03。

---

### 1.4 新能源出力滞后特征

```python
# 风速→风电出力 lag
feat = add_lag_features(feat, ["风速", "光伏出清电量"], lags=[4, 96])
# 新能源占比特征
feat["renewables_ratio"] = (
    feat["光伏出清电量"].fillna(0) / (feat["全省实时负荷"] + 1e-6)
)
```

**所需数据列**：`整合型 Excel.xlsx` → `风速`、`光伏出清电量`、`全省实时负荷`  
**预期提升**：改善白天平段新能源高渗透时段的预测偏差。

---

## 二、分时段建模

### 2.1 为每个时段独立训练模型

不同时段的负荷驱动因素差异显著（低谷受工业排班影响、高峰受温度和商业负荷影响）。

```python
class SegmentedLoadForecaster:
    """为低谷/平段/高峰分别训练独立 LightGBM 模型。"""

    PERIODS = ["低谷", "平段", "高峰"]

    def __init__(self, base_params=None):
        self._models = {}
        self._base_params = base_params or _DEFAULT_LGBM_PARAMS

    def fit(self, df):
        from analysis.time_period_labeling import add_period_label
        df = add_period_label(df.copy(), slot_col="time_slot_1based")
        for period in self.PERIODS:
            mask = df["time_period"] == period
            sub = df[mask]
            # 针对时段调整超参数
            params = dict(self._base_params)
            if period == "高峰":
                params["num_leaves"] = 127      # 更复杂的模型
                params["learning_rate"] = 0.03
            elif period == "低谷":
                params["num_leaves"] = 31       # 低谷规律性强，简单模型即可
            self._models[period] = lgb.LGBMRegressor(**params)
            self._models[period].fit(sub[feature_cols], sub["load_mw"])
        return self
```

**预期提升**：各时段独立模型 MAPE 可较全局模型降低 0.5–2%，高峰段效果最明显。

---

## 三、超参数优化 — 时段专用参数

| 时段 | num_leaves | learning_rate | min_data_in_leaf | 说明 |
|------|-----------|---------------|------------------|------|
| 低谷 | 31        | 0.05          | 30               | 规律平稳，浅模型防过拟合 |
| 平段 | 63        | 0.05          | 20               | 默认参数表现良好 |
| 高峰 | 127       | 0.03          | 10               | 高峰波动大，深模型+小学习率 |

```python
# 使用 Optuna 进行自动调优（按时段）
import optuna

def objective(trial, X_train, y_train):
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 50),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
    }
    model = lgb.LGBMRegressor(n_estimators=300, **params, verbose=-1)
    # 时序交叉验证评估
    ...
    return cv_mape
```

**所需依赖**：`pip install optuna`  
**预期提升**：超参数优化后全时段平均 MAPE 可降低 0.3–0.8%。

---

## 四、完整特征列清单

以下是建议加入 `LoadForecaster._build_features()` 的全部特征：

### 4.1 原有特征（保留）
| 特征名 | 来源 | 说明 |
|--------|------|------|
| `hour`, `minute`, `dayofweek`, `month` | DatetimeIndex | 时钟特征 |
| `time_slot` (0-95) | DatetimeIndex | 15分钟粒度标识 |
| `is_weekend`, `is_holiday`, `is_workday` | DatetimeIndex + 假期表 | 日类型 |
| `load_mw_lag4/8/96/192/288` | 历史负荷 | 短/中/长期负荷滞后 |
| `load_mw_roll4/96_mean/std` | 历史负荷 | 滚动统计 |
| `temperature_lag96` | 气象 | 前一天同时刻温度 |

### 4.2 新增特征（建议）
| 特征名 | 来源数据列 | 优先级 |
|--------|-----------|--------|
| `time_period_enc` (0/1/2) | `时段序号` + 分时标签模块 | ⭐⭐⭐ 高 |
| `period_x_hour` | 交叉特征 | ⭐⭐ 中 |
| `竞价空间_lag96` | `竞价空间（MW）` | ⭐⭐⭐ 高 |
| `竞价空间_roll96_mean` | `竞价空间（MW）` | ⭐⭐ 中 |
| `da_price_lag96` | `日前出清价格（元 / 兆瓦时）` | ⭐⭐⭐ 高 |
| `rt_price_lag96` | `实时出清价格（元/兆瓦时）` | ⭐⭐⭐ 高 |
| `da_rt_spread_lag96` | 两者之差 | ⭐⭐ 中 |
| `风速_lag96` | `风速` | ⭐⭐ 中 |
| `光伏出清电量_lag96` | `光伏出清电量` | ⭐⭐ 中 |
| `renewables_ratio` | `光伏出清电量` / `全省实时负荷` | ⭐⭐ 中 |
| `跨省联络线功率_lag96` | `跨省联络线功率（MW）` | ⭐ 低 |
| `正负荷备用_lag96` | `正负荷备用` | ⭐ 低 |

---

## 五、预期性能提升汇总

| 改进项 | 预期 MAPE 降低 | 难度 |
|--------|--------------|------|
| 时段标签特征 | 0.5–1.5% | 低 |
| 竞价空间特征 | 0.3–0.8% | 低 |
| 价格反馈特征 | 0.5–1.0% | 低 |
| 新能源特征 | 0.3–0.6% | 低 |
| 分时段建模 | 0.5–2.0% | 中 |
| 超参数优化（Optuna） | 0.3–0.8% | 中 |
| **合计（保守估计）** | **2.0–5.0%** | — |

> 注：提升幅度取决于历史数据的质量和时间跨度。建议优先实施高优先级特征（⭐⭐⭐），在验证改进后再引入分时段建模。

---

## 六、代码集成示例

```python
# 在 models/load_forecasting.py 中修改 _build_features()

def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
    from analysis.time_period_labeling import add_period_label

    feat = add_time_features(df)

    # -- 新增：时段编码 --
    # 确保存在 time_slot 列（0-95），转为1-96作为标签输入
    feat["time_slot_1based"] = feat["time_slot"] + 1
    feat = add_period_label(feat, slot_col="time_slot_1based")
    period_map = {"低谷": 0, "平段": 1, "高峰": 2}
    feat["time_period_enc"] = feat["time_period"].map(period_map)
    feat["period_x_hour"] = feat["time_period_enc"] * feat["hour"]
    feat.drop(columns=["time_period", "time_slot_1based"], inplace=True, errors="ignore")

    # -- 原有滞后/滚动特征 --
    feat = add_lag_features(feat, [self.TARGET], self.lag_steps)
    feat = add_lag_features(feat, self.weather_cols, [96])

    # -- 新增：价格和竞价空间滞后 --
    price_cols = [c for c in ["日前出清价格（元 / 兆瓦时）",
                               "实时出清价格（元/兆瓦时）",
                               "竞价空间（MW）"] if c in feat.columns]
    if price_cols:
        feat = add_lag_features(feat, price_cols, [96])

    feat = add_rolling_features(feat, [self.TARGET], self.roll_windows, stats=["mean", "std"])
    feat.dropna(inplace=True)
    return feat
```

---

*文件生成于 2026-03-25 | 基于江西省电力现货市场 2025-10 至 2026-02 历史数据*
