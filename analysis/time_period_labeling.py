"""
时段标签生成模块 — 江西省3月份时段划分规则
Time-Period Labelling Module — Jiangxi Province March Rules

March 2026 four-tier segmentation (no ultra-peak / 尖峰 in March):
    低谷 (Valley)  : 01:00–05:00 | 11:30–12:00 | 14:00–14:30
    平段 (Flat)    : 00:00–01:00 | 05:00–11:30 | 12:00–12:30* | 12:30–14:00
                     | 14:30–16:00* | 22:00–24:00
    高峰 (Peak)    : 16:00–22:00
    尖峰 (Ultra)   : 无 (not applicable for March)

    *Intervals not explicitly enumerated in the official schedule are assigned
     to 平段 by elimination so that every minute of the day is covered.

Public API:
    label_slot(slot)         → str label for a 1-based sequence slot (1–96).
    label_timestamp(ts)      → str label for a pd.Timestamp.
    add_period_label(df, ...) → DataFrame with a new 'time_period' column.
    PERIOD_SLOTS             → dict mapping period name → frozenset of slots.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Union

import pandas as pd

# ---------------------------------------------------------------------------
# March 2026 period definitions
# Slots are 1-based (slot 1 = 00:00–00:15, slot 96 = 23:45–24:00).
# ---------------------------------------------------------------------------

# Each tuple (start_minute_inclusive, end_minute_exclusive) uses minutes-of-day.
_MARCH_VALLEY_RANGES = [
    (60, 300),     # 01:00–05:00
    (690, 720),    # 11:30–12:00
    (840, 870),    # 14:00–14:30
]

_MARCH_PEAK_RANGES = [
    (960, 1320),   # 16:00–22:00
]

# Everything not covered by valley or peak is flat (平段).
# Explicit flat ranges for documentation:
#   00:00–01:00 | 05:00–11:30 | 12:00–14:00* | 14:30–16:00* | 22:00–24:00
# (*includes unlisted gaps assigned to flat by elimination)

PERIOD_LABELS = {
    "低谷": "低谷",
    "平段": "平段",
    "高峰": "高峰",
    "尖峰": "尖峰",
}


def _minute_to_label(minute_of_day: int) -> str:
    """Return the period label for a given minute-of-day (0–1439)."""
    for start, end in _MARCH_VALLEY_RANGES:
        if start <= minute_of_day < end:
            return "低谷"
    for start, end in _MARCH_PEAK_RANGES:
        if start <= minute_of_day < end:
            return "高峰"
    return "平段"


def _build_slot_map() -> Dict[int, str]:
    """Pre-compute label for every 1-based slot (1–96)."""
    mapping: Dict[int, str] = {}
    for slot in range(1, 97):
        # Slot i covers minutes [(i-1)*15, i*15)
        minute = (slot - 1) * 15
        mapping[slot] = _minute_to_label(minute)
    return mapping


# Cached slot → label mapping (immutable after module load)
_SLOT_LABEL_MAP: Dict[int, str] = _build_slot_map()

# Reverse mapping: period name → frozenset of 1-based slots
PERIOD_SLOTS: Dict[str, FrozenSet[int]] = {
    label: frozenset(s for s, lbl in _SLOT_LABEL_MAP.items() if lbl == label)
    for label in ("低谷", "平段", "高峰")
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def label_slot(slot: int) -> str:
    """Return the March time-period label for a 1-based 15-min sequence slot.

    Args:
        slot: Integer in range [1, 96].

    Returns:
        One of '低谷', '平段', '高峰'.

    Raises:
        ValueError: If *slot* is outside [1, 96].
    """
    if slot not in _SLOT_LABEL_MAP:
        raise ValueError(f"slot must be in [1, 96], got {slot!r}")
    return _SLOT_LABEL_MAP[slot]


def label_timestamp(ts: Union[pd.Timestamp, str]) -> str:
    """Return the March time-period label for a pd.Timestamp (or date string).

    Only the time-of-day component is used; the date is ignored.

    Args:
        ts: Timestamp or ISO date-time string.

    Returns:
        One of '低谷', '平段', '高峰'.
    """
    ts = pd.Timestamp(ts)
    minute_of_day = ts.hour * 60 + ts.minute
    return _minute_to_label(minute_of_day)


def add_period_label(
    df: pd.DataFrame,
    slot_col: str = "时段序号",
    label_col: str = "time_period",
) -> pd.DataFrame:
    """Add a time-period label column to a DataFrame.

    The function accepts either a slot-number column or uses the DataFrame's
    DateTimeIndex when *slot_col* is not present.

    Args:
        df: Input DataFrame.  Must contain *slot_col* **or** have a
            pd.DatetimeIndex.
        slot_col: Name of the 1-based slot column (1–96).  If absent, the
                  function falls back to the DatetimeIndex.
        label_col: Name of the new label column.

    Returns:
        Copy of *df* with an additional *label_col* column containing
        '低谷', '平段', or '高峰'.

    Raises:
        ValueError: If neither *slot_col* nor a DatetimeIndex is available.
    """
    out = df.copy()

    if slot_col in out.columns:
        out[label_col] = out[slot_col].map(_SLOT_LABEL_MAP)
    elif isinstance(out.index, pd.DatetimeIndex):
        out[label_col] = out.index.map(label_timestamp)
    else:
        raise ValueError(
            f"DataFrame must contain column '{slot_col}' or have a "
            "pd.DatetimeIndex for time-period labelling."
        )

    return out


def period_summary() -> pd.DataFrame:
    """Return a human-readable summary table of March period definitions.

    Returns:
        DataFrame with columns ['时段', '时间范围', '时段数量', '时间占比(%)'].
    """
    rows = []
    for period_name, ranges in [
        ("低谷", _MARCH_VALLEY_RANGES),
        ("高峰", _MARCH_PEAK_RANGES),
    ]:
        time_ranges = []
        for start, end in ranges:
            sh, sm = divmod(start, 60)
            eh, em = divmod(end, 60)
            time_ranges.append(f"{sh:02d}:{sm:02d}–{eh:02d}:{em:02d}")
        slot_count = len(PERIOD_SLOTS[period_name])
        rows.append({
            "时段": period_name,
            "时间范围": "、".join(time_ranges),
            "时段数量": slot_count,
            "时间占比(%)": round(slot_count / 96 * 100, 1),
        })

    # Flat is everything else
    flat_count = len(PERIOD_SLOTS["平段"])
    # Derive flat ranges programmatically from unoccupied minutes
    flat_intervals = []
    covered = set()
    for start, end in _MARCH_VALLEY_RANGES + _MARCH_PEAK_RANGES:
        covered.update(range(start, end))
    i = 0
    while i < 1440:
        if i not in covered:
            j = i
            while j < 1440 and j not in covered:
                j += 1
            sh, sm = divmod(i, 60)
            eh, em = divmod(j, 60)
            flat_intervals.append(f"{sh:02d}:{sm:02d}–{eh:02d}:{em:02d}")
            i = j
        else:
            i += 1
    flat_ranges = "、".join(flat_intervals)
    rows.insert(
        1,
        {
            "时段": "平段",
            "时间范围": flat_ranges,
            "时段数量": flat_count,
            "时间占比(%)": round(flat_count / 96 * 100, 1),
        },
    )

    return pd.DataFrame(rows)
