"""
Feature engineering utilities for organoid-spike-visual-analytics.

Design boundary
---------------
This module receives `cleaned_df` from `validation.py` and converts it into
analysis-ready intermediate tables for downstream plotting, connectivity, and
metrics modules.

It deliberately does not plot anything and does not import visualization
libraries. The intended workflow is:

    raw_df -> validation.validate_spike_bin_dataframe() -> cleaned_df
           -> SpikeFeatureContext(cleaned_df)
           -> prepare_xxx_data(ctx)
           -> heatmap / raster / population trace / spatial nodes / dashboard inputs

`SpikeFeatureContext` is intentionally lightweight. It centralizes:
- cleaned_df ownership
- electrode_id natural sorting
- time-window sorting
- optional condition comparison gates
- optional spatial analysis gates
- slicing and small table cache

Core calculations stay in stateless compute_xxx functions. Plot-facing and
summary-facing assembly stays in prepare_xxx_data functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Hashable, Iterable, Literal, Optional

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

REQUIRED_STANDARD_COLUMNS = {
    "electrode_id",
    "bin_start_ms",
    "bin_end_ms",
    "bin_width_ms",
    "spike_count",
    "firing_rate_hz",
}

SPATIAL_COLUMNS = ("x_um", "y_um")
VALUE_COLUMNS = ("spike_count", "firing_rate_hz")

BurstMethod = Literal["mean_std", "quantile"]


# -----------------------------------------------------------------------------
# Small sorting / validation helpers
# -----------------------------------------------------------------------------

_NUMBER_PATTERN = re.compile(r"[-+]?\d*\.?\d+")


def _copy_if_requested(df: pd.DataFrame, *, copy: bool = True) -> pd.DataFrame:
    """Return a defensive copy unless the caller explicitly opts out."""
    return df.copy() if copy else df


def _ensure_standard_cleaned_df(columns: Iterable[Hashable]) -> None:
    """
    Lightweight contract check for validation.py output.

    Only column names are needed here. `validation.py` already performs the
    dataframe-level validation and standardization, so features.py should not
    pass the full dataframe into this schema check.
    """
    column_set = set(columns)
    missing = REQUIRED_STANDARD_COLUMNS.difference(column_set)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(
            "features.py expects validation-standardized cleaned_df. "
            f"Missing required columns: {missing_text}"
        )


def _validate_value_col(value_col: str) -> None:
    """Keep public matrix-building functions restricted to canonical values."""
    if value_col not in VALUE_COLUMNS:
        allowed = ", ".join(VALUE_COLUMNS)
        raise ValueError(f"value_col must be one of: {allowed}")


def _safe_divide(numerator: float, denominator: float, default: float = np.nan) -> float:
    """Return numerator / denominator, or default when denominator is not usable."""
    if pd.isna(denominator) or float(denominator) <= 0:
        return default
    return float(numerator) / float(denominator)


def _first_number(value: object) -> Optional[float]:
    """Extract the first numeric token from an electrode id, if present."""
    match = _NUMBER_PATTERN.search(str(value))
    if match is None:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def _electrode_sort_key(value: object) -> tuple[int, float, str]:
    """
    Sort electrode_id by embedded number when available, otherwise by text.

    Examples
    --------
    E1, E2, E10 are sorted as E1 < E2 < E10 instead of lexicographic
    E1 < E10 < E2.
    """
    text = str(value)
    number = _first_number(text)
    if number is None:
        return (1, np.inf, text)
    return (0, number, text)


def sorted_electrode_ids(values: Iterable[Hashable]) -> list[Hashable]:
    """Return unique electrode ids in the project-wide natural order."""
    unique = pd.Series(list(values)).dropna().drop_duplicates().tolist()
    return sorted(unique, key=_electrode_sort_key)


def sorted_time_starts(values: Iterable[float]) -> list[float]:
    """Return unique bin_start_ms values in ascending numeric order."""
    unique = pd.Series(list(values)).astype(float).dropna().drop_duplicates().tolist()
    return sorted(unique)


def _aggregation_for_value(value_col: str) -> str:
    """
    Choose a sensible default aggregation for duplicate electrode-time rows.

    Duplicate rows can appear when users have multiple trials or recordings.
    For spike_count, summing preserves event counts. For firing_rate_hz, mean is
    a better default because rates are already normalized by bin width.
    """
    return "sum" if value_col == "spike_count" else "mean"


def _condition_cache_key(condition: Optional[Hashable]) -> str:
    """Make a stable cache key fragment for optional condition filters."""
    return "__all__" if condition is None else f"condition={condition!r}"


def _recording_duration_s(df: pd.DataFrame) -> float:
    """Return slice duration in seconds from standardized time-window columns."""
    if df.empty:
        return 0.0
    duration_ms = float(df["bin_end_ms"].max() - df["bin_start_ms"].min())
    return max(duration_ms / 1000.0, 0.0)


def _empty_condition_summary() -> pd.DataFrame:
    """Return the stable condition-summary schema with no rows."""
    return pd.DataFrame(
        columns=[
            "condition",
            "mean_total_spike_count_per_bin",
            "mean_firing_rate_hz",
            "active_electrode_ratio",
            "burst_count",
            "burst_rate",
        ]
    )


# -----------------------------------------------------------------------------
# Public stateless computation functions
# -----------------------------------------------------------------------------


def build_electrode_time_matrix(
    df: pd.DataFrame,
    value_col: str = "spike_count",
    *,
    electrode_order: Optional[list[Hashable]] = None,
    time_order: Optional[list[float]] = None,
    fill_value: float = 0.0,
    aggfunc: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build an electrode x time matrix.

    Parameters
    ----------
    df:
        Validation-standardized cleaned dataframe, optionally pre-filtered by
        condition/trial/recording.
    value_col:
        Either "spike_count" or "firing_rate_hz".
    electrode_order:
        Optional explicit row order. When omitted, natural electrode sorting is
        used, so E1/E2/E10 remain numerically ordered.
    time_order:
        Optional explicit column order. When omitted, bin_start_ms is sorted
        ascending.
    fill_value:
        Value used for missing electrode-time combinations.
    aggfunc:
        Optional pivot aggregation function. Defaults to sum for spike_count and
        mean for firing_rate_hz.

    Returns
    -------
    pandas.DataFrame
        Rows are electrode_id, columns are bin_start_ms, values are value_col.
    """
    _ensure_standard_cleaned_df(df.columns)
    _validate_value_col(value_col)

    if df.empty:
        return pd.DataFrame(dtype=float)

    electrode_order = electrode_order or sorted_electrode_ids(df["electrode_id"])
    time_order = time_order or sorted_time_starts(df["bin_start_ms"])
    aggfunc = aggfunc or _aggregation_for_value(value_col)

    matrix = df.pivot_table(
        index="electrode_id",
        columns="bin_start_ms",
        values=value_col,
        aggfunc=aggfunc,
        fill_value=fill_value,
    )

    matrix = matrix.reindex(index=electrode_order, columns=time_order, fill_value=fill_value)
    matrix.index.name = "electrode_id"
    matrix.columns.name = "bin_start_ms"

    return matrix.astype(float)


def compute_active_electrode_ratio_by_bin(
    df: pd.DataFrame,
    *,
    electrode_order: Optional[list[Hashable]] = None,
) -> pd.DataFrame:
    """
    Compute active-electrode count and ratio for each time bin.

    A bin treats one electrode as active when the summed spike_count for that
    electrode and bin is greater than 0. This avoids over-counting when the same
    electrode/bin appears in multiple trials or recordings.
    """
    _ensure_standard_cleaned_df(df.columns)
    columns = ["bin_start_ms", "active_electrode_count", "active_electrode_ratio"]
    if df.empty:
        return pd.DataFrame(columns=columns)

    electrode_order = electrode_order or sorted_electrode_ids(df["electrode_id"])
    n_electrodes = len(electrode_order)

    electrode_bin = (
        df.groupby(["bin_start_ms", "electrode_id"], as_index=False, sort=True)["spike_count"]
        .sum()
        .assign(_is_active=lambda x: x["spike_count"] > 0)
    )
    out = (
        electrode_bin.groupby("bin_start_ms", as_index=False, sort=True)["_is_active"]
        .sum()
        .rename(columns={"_is_active": "active_electrode_count"})
    )
    out["active_electrode_ratio"] = [
        _safe_divide(count, n_electrodes) for count in out["active_electrode_count"]
    ]
    return out[columns]


def compute_population_firing_rate(
    df: pd.DataFrame,
    *,
    electrode_order: Optional[list[Hashable]] = None,
) -> pd.DataFrame:
    """
    Compute population activity by time bin.

    Returns one row per bin with total spike count, mean firing rate, and active
    electrode count. The table is suitable for population trace plotting and
    burst detection.
    """
    _ensure_standard_cleaned_df(df.columns)
    columns = [
        "bin_start_ms",
        "bin_end_ms",
        "total_spike_count",
        "mean_firing_rate_hz",
        "active_electrode_count",
        "active_electrode_ratio",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    base = (
        df.groupby("bin_start_ms", as_index=False, sort=True)
        .agg(
            bin_end_ms=("bin_end_ms", "max"),
            total_spike_count=("spike_count", "sum"),
            mean_firing_rate_hz=("firing_rate_hz", "mean"),
        )
        .sort_values("bin_start_ms")
    )
    active = compute_active_electrode_ratio_by_bin(df, electrode_order=electrode_order)
    out = base.merge(active, on="bin_start_ms", how="left")
    out["active_electrode_count"] = out["active_electrode_count"].fillna(0).astype(int)
    return out[columns].reset_index(drop=True)


def detect_population_bursts(
    population_activity: pd.DataFrame,
    *,
    activity_col: str = "mean_firing_rate_hz",
    bin_start_col: str = "bin_start_ms",
    bin_end_col: str = "bin_end_ms",
    method: BurstMethod = "mean_std",
    n_std: float = 2.0,
    quantile: float = 0.90,
    min_duration_bins: int = 1,
) -> pd.DataFrame:
    """
    Detect simple population bursts from a population activity table.

    First-stage rule, intentionally simple:
    - method="mean_std": activity > mean + n_std * std
    - method="quantile": activity > activity.quantile(quantile)

    Consecutive high-activity bins are merged into one burst event.
    """
    required = {bin_start_col, activity_col}
    missing = required.difference(population_activity.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"population_activity missing columns: {missing_text}")

    columns = [
        "burst_start_ms",
        "burst_end_ms",
        "burst_peak_ms",
        "burst_strength",
        "burst_duration_bins",
        "threshold",
        "detection_method",
    ]

    if population_activity.empty:
        return pd.DataFrame(columns=columns)

    work = population_activity.sort_values(bin_start_col).reset_index(drop=True).copy()
    values = pd.to_numeric(work[activity_col], errors="coerce").fillna(0.0)

    if len(values) == 0 or float(values.max()) <= 0.0:
        return pd.DataFrame(columns=columns)

    if method == "mean_std":
        threshold = float(values.mean() + n_std * values.std(ddof=0))
    elif method == "quantile":
        threshold = float(values.quantile(quantile))
    else:
        raise ValueError("method must be either 'mean_std' or 'quantile'")

    high = values > threshold
    if not bool(high.any()):
        return pd.DataFrame(columns=columns)

    events: list[dict[str, float | int | str]] = []
    start_idx: Optional[int] = None

    for idx, is_high in enumerate(high.tolist()):
        if is_high and start_idx is None:
            start_idx = idx
        is_last = idx == len(high) - 1
        if start_idx is not None and ((not is_high) or is_last):
            end_idx = idx if (is_high and is_last) else idx - 1
            duration_bins = end_idx - start_idx + 1
            if duration_bins >= min_duration_bins:
                segment = work.iloc[start_idx : end_idx + 1]
                peak_row_idx = segment[activity_col].astype(float).idxmax()
                peak_row = work.loc[peak_row_idx]

                if bin_end_col in work.columns:
                    burst_end = float(work.loc[end_idx, bin_end_col])
                else:
                    burst_end = float(work.loc[end_idx, bin_start_col])

                events.append(
                    {
                        "burst_start_ms": float(work.loc[start_idx, bin_start_col]),
                        "burst_end_ms": burst_end,
                        "burst_peak_ms": float(peak_row[bin_start_col]),
                        "burst_strength": float(peak_row[activity_col]),
                        "burst_duration_bins": int(duration_bins),
                        "threshold": threshold,
                        "detection_method": method,
                    }
                )
            start_idx = None

    return pd.DataFrame(events, columns=columns)


def compute_mean_firing_rate_hz(df: pd.DataFrame) -> float:
    """Return dataset-level mean firing_rate_hz."""
    _ensure_standard_cleaned_df(df.columns)
    if df.empty:
        return np.nan
    return float(df["firing_rate_hz"].mean())


def compute_electrode_mean_firing_rate_hz(
    df: pd.DataFrame,
    *,
    electrode_order: Optional[list[Hashable]] = None,
) -> pd.DataFrame:
    """Return one node-level mean_firing_rate_hz value per electrode."""
    _ensure_standard_cleaned_df(df.columns)
    columns = ["electrode_id", "mean_firing_rate_hz"]
    if df.empty:
        return pd.DataFrame(columns=columns)

    electrode_order = electrode_order or sorted_electrode_ids(df["electrode_id"])
    out = (
        df.groupby("electrode_id", as_index=False, sort=False)["firing_rate_hz"]
        .mean()
        .rename(columns={"firing_rate_hz": "mean_firing_rate_hz"})
    )
    order_map = {electrode_id: i for i, electrode_id in enumerate(electrode_order)}
    out["_electrode_order"] = out["electrode_id"].map(order_map)
    out = out.sort_values("_electrode_order", kind="mergesort").drop(columns="_electrode_order")
    return out[columns].reset_index(drop=True)


def compute_active_electrode_ratio(
    df: pd.DataFrame,
    *,
    electrode_order: Optional[list[Hashable]] = None,
) -> float:
    """Return the mean active-electrode ratio across all time bins."""
    _ensure_standard_cleaned_df(df.columns)
    if df.empty:
        return np.nan

    by_bin = compute_active_electrode_ratio_by_bin(df, electrode_order=electrode_order)
    if by_bin.empty:
        return np.nan
    return float(by_bin["active_electrode_ratio"].mean())



def compute_condition_summary_row(
    *,
    condition: Hashable,
    df: pd.DataFrame,
    population_activity: pd.DataFrame,
    bursts: pd.DataFrame,
    mean_firing_rate_hz: float,
    active_electrode_ratio: float,
) -> dict[str, float | int | Hashable]:
    """Assemble one condition-level feature summary row from computed inputs."""
    _ensure_standard_cleaned_df(df.columns)
    duration_s = _recording_duration_s(df)
    mean_total = (
        float(population_activity["total_spike_count"].mean())
        if not population_activity.empty
        else np.nan
    )
    return {
        "condition": condition,
        "mean_total_spike_count_per_bin": mean_total,
        "mean_firing_rate_hz": mean_firing_rate_hz,
        "active_electrode_ratio": active_electrode_ratio,
        "burst_count": int(len(bursts)),
        "burst_rate": _safe_divide(len(bursts), duration_s),
    }


# Backward-friendly aliases with explicit metric names.
mean_firing_rate_hz = compute_mean_firing_rate_hz
active_electrode_ratio = compute_active_electrode_ratio


# -----------------------------------------------------------------------------
# Lightweight context class
# -----------------------------------------------------------------------------


@dataclass
class SpikeFeatureContext:
    """
    Lightweight manager for feature-ready cleaned_df.

    The class keeps state and gates close to the data. It does not compute chart
    tables, dashboard tables, network tables, or export bundles. Those are
    handled by prepare_xxx_data functions below.
    """

    cleaned_df: pd.DataFrame
    default_burst_method: BurstMethod = "mean_std"
    default_burst_n_std: float = 2.0
    default_burst_quantile: float = 0.90
    copy_input: bool = True
    _df: pd.DataFrame = field(init=False, repr=False)
    _cache: dict[str, pd.DataFrame] = field(init=False, default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Validate the input columns and keep one internally sorted dataframe."""
        _ensure_standard_cleaned_df(self.cleaned_df.columns)
        work = _copy_if_requested(self.cleaned_df, copy=self.copy_input)
        order_map = {electrode_id: i for i, electrode_id in enumerate(sorted_electrode_ids(work["electrode_id"]))}
        work = work.copy()
        work["_electrode_order"] = work["electrode_id"].map(order_map)
        self._df = (
            work.sort_values(
                by=["bin_start_ms", "bin_end_ms", "_electrode_order"],
                kind="mergesort",
            )
            .drop(columns="_electrode_order")
            .reset_index(drop=True)
        )

    @property
    def df(self) -> pd.DataFrame:
        """Return a defensive copy of the standardized dataframe."""
        return self._df.copy()

    @property
    def spatial_available(self) -> bool:
        """Whether spatial outputs should be generated."""
        return all(col in self._df.columns for col in SPATIAL_COLUMNS) and not self._df[list(SPATIAL_COLUMNS)].isna().any().any()

    @property
    def condition_available(self) -> bool:
        """Whether a condition column exists at all."""
        return "condition" in self._df.columns

    @property
    def condition_compare_available(self) -> bool:
        """Whether condition comparison should be enabled."""
        return self.condition_available and self._df["condition"].dropna().nunique() > 1

    @property
    def electrode_order(self) -> list[Hashable]:
        """Global electrode order shared by all feature outputs."""
        key = "electrode_order"
        cached = self.get_cached_table(key)
        if cached is None:
            cached = pd.DataFrame({"electrode_id": sorted_electrode_ids(self._df["electrode_id"])})
            self.set_cached_table(key, cached)
        return cached["electrode_id"].tolist()

    @property
    def time_order(self) -> list[float]:
        """Global bin_start_ms order shared by matrix-like outputs."""
        key = "time_order"
        cached = self.get_cached_table(key)
        if cached is None:
            cached = pd.DataFrame({"bin_start_ms": sorted_time_starts(self._df["bin_start_ms"])})
            self.set_cached_table(key, cached)
        return cached["bin_start_ms"].tolist()

    @property
    def conditions(self) -> list[Hashable]:
        """Return condition labels in first-appearance order when comparison exists."""
        if not self.condition_compare_available:
            return []
        return self._df["condition"].dropna().drop_duplicates().tolist()

    def analysis_gates(self) -> dict[str, bool]:
        """Small status dictionary for app/dashboard modules."""
        return {
            "spatial_available": self.spatial_available,
            "condition_available": self.condition_available,
            "condition_compare_available": self.condition_compare_available,
        }

    def sorted_dataframe(self) -> pd.DataFrame:
        """Return cleaned_df sorted by time and natural electrode order."""
        return self._df.copy()

    def slice(self, condition: Optional[Hashable] = None) -> pd.DataFrame:
        """Return rows for one condition, or all rows when condition is None."""
        if condition is None:
            return self._df.copy()
        if "condition" not in self._df.columns:
            return self._df.iloc[0:0].copy()
        return self._df[self._df["condition"] == condition].copy()

    def condition_slices(self) -> list[tuple[Hashable, pd.DataFrame]]:
        """Return condition slices, or one synthetic all-slice when no comparison exists."""
        if self.condition_compare_available:
            return [(condition, self.slice(condition)) for condition in self.conditions]
        return [("all", self.slice(None))]

    def get_cached_table(self, key: str) -> Optional[pd.DataFrame]:
        """Return a cached dataframe copy, or None when the key is absent."""
        value = self._cache.get(key)
        if value is None:
            return None
        return value.copy()

    def set_cached_table(self, key: str, value: pd.DataFrame) -> None:
        """Store a dataframe copy in the small context cache."""
        self._cache[key] = value.copy()


# -----------------------------------------------------------------------------
# Prepare-layer functions
# -----------------------------------------------------------------------------


def prepare_electrode_time_matrix_data(
    ctx: SpikeFeatureContext,
    value_col: str = "spike_count",
    *,
    condition: Optional[Hashable] = None,
    use_global_time_order: bool = False,
) -> pd.DataFrame:
    """Prepare electrode x time matrix for heatmap, raster, and connectivity input."""
    _validate_value_col(value_col)
    cache_key = f"matrix|{value_col}|{_condition_cache_key(condition)}|global_time={use_global_time_order}"
    cached = ctx.get_cached_table(cache_key)
    if cached is not None:
        return cached

    df = ctx.slice(condition)
    time_order = ctx.time_order if use_global_time_order else sorted_time_starts(df["bin_start_ms"])
    out = build_electrode_time_matrix(
        df,
        value_col=value_col,
        electrode_order=ctx.electrode_order,
        time_order=time_order,
        fill_value=0.0,
    )
    ctx.set_cached_table(cache_key, out)
    return out.copy()


def prepare_population_activity_data(
    ctx: SpikeFeatureContext,
    *,
    condition: Optional[Hashable] = None,
) -> pd.DataFrame:
    """Prepare rich per-bin population activity for trace, burst, and summary use."""
    cache_key = f"population_activity|{_condition_cache_key(condition)}"
    cached = ctx.get_cached_table(cache_key)
    if cached is not None:
        return cached

    df = ctx.slice(condition)
    out = compute_population_firing_rate(df, electrode_order=ctx.electrode_order)
    ctx.set_cached_table(cache_key, out)
    return out.copy()


def prepare_population_trace_data(
    ctx: SpikeFeatureContext,
    *,
    condition: Optional[Hashable] = None,
) -> pd.DataFrame:
    """Prepare the minimal table for Network / Population Firing Rate Trace."""
    activity = prepare_population_activity_data(ctx, condition=condition)
    return activity[["bin_start_ms", "mean_firing_rate_hz"]].copy()


def prepare_burst_table(
    ctx: SpikeFeatureContext,
    *,
    condition: Optional[Hashable] = None,
    method: Optional[BurstMethod] = None,
    n_std: Optional[float] = None,
    quantile: Optional[float] = None,
    min_duration_bins: int = 1,
) -> pd.DataFrame:
    """Prepare simple population burst events for all data or one condition."""
    method = method or ctx.default_burst_method
    n_std = ctx.default_burst_n_std if n_std is None else n_std
    quantile = ctx.default_burst_quantile if quantile is None else quantile
    cache_key = (
        f"bursts|{_condition_cache_key(condition)}|method={method}|"
        f"n_std={n_std}|quantile={quantile}|min_bins={min_duration_bins}"
    )
    cached = ctx.get_cached_table(cache_key)
    if cached is not None:
        return cached

    activity = prepare_population_activity_data(ctx, condition=condition)
    bursts = detect_population_bursts(
        activity,
        method=method,
        n_std=n_std,
        quantile=quantile,
        min_duration_bins=min_duration_bins,
    )
    if condition is not None and not bursts.empty:
        bursts.insert(0, "condition", condition)
    ctx.set_cached_table(cache_key, bursts)
    return bursts.copy()



def prepare_spatial_node_table(
    ctx: SpikeFeatureContext,
    *,
    condition: Optional[Hashable] = None,
) -> pd.DataFrame:
    """Prepare node-level activity and coordinate data for spatial plots.

    This function no longer depends on a general electrode-summary scheduler. It
    assembles only the fields needed by spatial plots: coordinates, node-level
    mean firing rate, total spike count, and active-bin ratio.
    """
    cols = [
        "electrode_id",
        "x_um",
        "y_um",
        "mean_firing_rate_hz",
        "total_spike_count",
        "active_bin_ratio",
    ]
    output_cols = (["condition"] + cols) if condition is not None else cols
    if not ctx.spatial_available:
        return pd.DataFrame(columns=output_cols)

    cache_key = f"spatial_node_table|{_condition_cache_key(condition)}"
    cached = ctx.get_cached_table(cache_key)
    if cached is not None:
        return cached

    df = ctx.slice(condition)
    if df.empty:
        out = pd.DataFrame(columns=output_cols)
        ctx.set_cached_table(cache_key, out)
        return out.copy()

    time_order = sorted_time_starts(df["bin_start_ms"])
    n_bins = len(time_order)

    coords = (
        ctx.df[["electrode_id", "x_um", "y_um"]]
        .groupby("electrode_id", as_index=False, sort=False)
        .agg(x_um=("x_um", "median"), y_um=("y_um", "median"))
    )
    mean_fr = compute_electrode_mean_firing_rate_hz(df, electrode_order=ctx.electrode_order)
    total_spikes = (
        df.groupby("electrode_id", as_index=False, sort=False)["spike_count"]
        .sum()
        .rename(columns={"spike_count": "total_spike_count"})
    )
    electrode_bin = (
        df.groupby(["electrode_id", "bin_start_ms"], as_index=False, sort=False)["spike_count"]
        .sum()
        .assign(_is_active=lambda x: x["spike_count"] > 0)
    )
    active_bins = (
        electrode_bin.groupby("electrode_id", as_index=False, sort=False)["_is_active"]
        .sum()
        .rename(columns={"_is_active": "active_bin_count"})
    )

    out = coords.merge(mean_fr, on="electrode_id", how="inner")
    out = out.merge(total_spikes, on="electrode_id", how="left")
    out = out.merge(active_bins, on="electrode_id", how="left")
    out["total_spike_count"] = out["total_spike_count"].fillna(0)
    out["active_bin_count"] = out["active_bin_count"].fillna(0)
    out["active_bin_ratio"] = [
        _safe_divide(count, n_bins) for count in out["active_bin_count"]
    ]
    out = out.drop(columns="active_bin_count")

    if condition is not None:
        out.insert(0, "condition", condition)

    order_map = {electrode_id: i for i, electrode_id in enumerate(ctx.electrode_order)}
    out["_electrode_order"] = out["electrode_id"].map(order_map)
    out = out.sort_values("_electrode_order", kind="mergesort").drop(columns="_electrode_order")
    out = out[output_cols].reset_index(drop=True)
    ctx.set_cached_table(cache_key, out)
    return out.copy()

def prepare_raster_data(
    ctx: SpikeFeatureContext,
    *,
    condition: Optional[Hashable] = None,
    active_only: bool = True,
) -> pd.DataFrame:
    """
    Prepare long-form binned raster data.

    A real spike raster would need raw spike timestamps. This project uses
    binned spike CSV, so the raster interface exposes active bins with their
    bin_start_ms/bin_end_ms and spike_count.
    """
    cache_key = f"raster|{_condition_cache_key(condition)}|active_only={active_only}"
    cached = ctx.get_cached_table(cache_key)
    if cached is not None:
        return cached

    df = ctx.slice(condition)
    cols = ["electrode_id", "bin_start_ms", "bin_end_ms", "spike_count", "firing_rate_hz"]
    if ctx.condition_available and "condition" in df.columns:
        cols.insert(0, "condition")
    out = df[cols].copy()
    if active_only:
        out = out[out["spike_count"] > 0]

    order_map = {electrode_id: i for i, electrode_id in enumerate(ctx.electrode_order)}
    out["electrode_order"] = out["electrode_id"].map(order_map)
    out = out.sort_values(["bin_start_ms", "electrode_order"], kind="mergesort").reset_index(drop=True)
    ctx.set_cached_table(cache_key, out)
    return out.copy()


def prepare_heatmap_data(
    ctx: SpikeFeatureContext,
    *,
    value_col: str = "spike_count",
    condition: Optional[Hashable] = None,
    include_bursts: bool = True,
) -> dict[str, pd.DataFrame]:
    """Prepare exactly the tables needed by a binned spike heatmap."""
    data = {
        "matrix": prepare_electrode_time_matrix_data(ctx, value_col=value_col, condition=condition),
    }
    if include_bursts:
        data["bursts"] = prepare_burst_table(ctx, condition=condition)
    return data


def prepare_condition_summary(ctx: SpikeFeatureContext) -> pd.DataFrame:
    """
    Prepare condition-level activity metrics for dashboard_data.py.

    When condition comparison is unavailable, this returns one row with
    condition="all". The function dispatches already separated computations:
    population activity, mean_firing_rate_hz, active_electrode_ratio, and bursts.
    """
    cache_key = "condition_summary"
    cached = ctx.get_cached_table(cache_key)
    if cached is not None:
        return cached

    rows: list[dict[str, float | int | Hashable]] = []
    for label, df_slice in ctx.condition_slices():
        condition = label if ctx.condition_compare_available else None
        if df_slice.empty:
            rows.append(
                {
                    "condition": label,
                    "mean_total_spike_count_per_bin": np.nan,
                    "mean_firing_rate_hz": np.nan,
                    "active_electrode_ratio": np.nan,
                    "burst_count": 0,
                    "burst_rate": np.nan,
                }
            )
            continue

        population_activity = prepare_population_activity_data(ctx, condition=condition)
        bursts = prepare_burst_table(ctx, condition=condition)
        mean_fr = compute_mean_firing_rate_hz(df_slice)
        active_ratio = compute_active_electrode_ratio(df_slice, electrode_order=ctx.electrode_order)
        rows.append(
            compute_condition_summary_row(
                condition=label,
                df=df_slice,
                population_activity=population_activity,
                bursts=bursts,
                mean_firing_rate_hz=mean_fr,
                active_electrode_ratio=active_ratio,
            )
        )

    out = pd.DataFrame(rows, columns=_empty_condition_summary().columns)
    ctx.set_cached_table(cache_key, out)
    return out.copy()


def prepare_metrics_dashboard_table(ctx: SpikeFeatureContext) -> pd.DataFrame:
    """
    Prepare feature-level metrics for Metrics Change Dashboard.

    This table intentionally contains only activity/burst metrics produced in
    features.py. dashboard_data.py should merge connectivity.py and
    network_metrics.py outputs into the final dashboard table.
    """
    out = prepare_condition_summary(ctx).copy()
    out.insert(1, "scope", "condition" if ctx.condition_compare_available else "all")
    return out


def prepare_feature_bundle(ctx: SpikeFeatureContext) -> dict[str, pd.DataFrame | dict[str, bool]]:
    """
    Convenience bundle for notebooks/export evidence packs.

    Correlation matrix is intentionally absent from this bundle. It belongs in
    connectivity.py and can use prepare_electrode_time_matrix_data(ctx) as input.
    General electrode-summary output is also intentionally absent.
    """
    return {
        "analysis_gates": ctx.analysis_gates(),
        "electrode_time_matrix": prepare_electrode_time_matrix_data(ctx),
        "population_activity": prepare_population_activity_data(ctx),
        "bursts": prepare_burst_table(ctx),
        "condition_summary": prepare_condition_summary(ctx),
        "spatial_node_table": prepare_spatial_node_table(ctx),
    }
