"""
Data upload and validation utilities for organoid-spike-visual-analytics.

Design boundary
---------------
This module intentionally keeps validation "necessary but not over-smart":
1. It does not rename ambiguous columns, infer time windows, or auto-complete arbitrary fields.
2. It only standardizes fields explicitly allowed by the project spec:
   - seconds / milliseconds time windows -> internal millisecond columns
   - spike_count / firing_rate_hz mutual completion
   - small coordinate floating-point jitter correction
3. It returns a structured report so Streamlit can show Error / Warning / Info clearly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd


# -----------------------------
# Project-level field definition
# -----------------------------
TEXT_COLUMNS = {"recording_id", "trial_id", "condition", "electrode_id"}

TIME_PAIR_MS = ("bin_start_ms", "bin_end_ms")
TIME_PAIR_S = ("bin_start_s", "bin_end_s")
METRIC_COLUMNS = ("spike_count", "firing_rate_hz")
COORD_COLUMNS = ("x_um", "y_um")

STATUS_ERROR = "Error (Not usable)"
STATUS_WARNING = "Warning (Ready with Warning)"
STATUS_INFO = "Info (Ready)"


@dataclass(frozen=True)
class ValidationIssue:
    """One validation message displayed in Data Health Report."""

    level: str  # "Error", "Warning", or "Info"
    code: str
    message: str
    suggestion: str = ""
    affected_rows: Optional[int] = None


@dataclass
class ValidationResult:
    """Structured output from validate_spike_bin_dataframe()."""

    status: str
    usable: bool
    cleaned_df: Optional[pd.DataFrame]
    issues: list[ValidationIssue] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    spatial_available: bool = False
    condition_compare_available: bool = False
    time_unit_used: Optional[str] = None
    metric_source: Optional[str] = None

    @property
    def errors(self) -> list[ValidationIssue]:
        return [x for x in self.issues if x.level == "Error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [x for x in self.issues if x.level == "Warning"]

    @property
    def infos(self) -> list[ValidationIssue]:
        return [x for x in self.issues if x.level == "Info"]


# -----------------------------
# Small internal helper functions
# -----------------------------
def _issue(
    issues: list[ValidationIssue],
    level: str,
    code: str,
    message: str,
    suggestion: str = "",
    affected_rows: Optional[int] = None,
) -> None:
    """Append one issue with consistent field order."""
    issues.append(
        ValidationIssue(
            level=level,
            code=code,
            message=message,
            suggestion=suggestion,
            affected_rows=affected_rows,
        )
    )


def _series_to_numeric(series: pd.Series) -> pd.Series:
    """
    Strict enough numeric conversion for CSV input.

    Allowed examples: 1, 1.5, -2, "3.0".
    Rejected examples: "1 ms", "high", "E001" in numeric columns.
    Empty cells are converted to NaN and handled by required-field checks.
    """
    return pd.to_numeric(series, errors="coerce")


def _bad_numeric_mask(original: pd.Series, numeric: pd.Series) -> pd.Series:
    """
    Identify cells that contain something but cannot be parsed as a number.

    True missing values are not marked here; missingness is checked separately.
    """
    has_value = original.notna() & (original.astype(str).str.strip() != "")
    return has_value & numeric.isna()


def _choose_time_pair(columns: set[str], issues: list[ValidationIssue]) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Choose the time-window columns.

    Rule:
    - If both millisecond columns exist, use millisecond columns first.
    - Else if both second columns exist, use second columns.
    - Start and end must appear as a pair; only start or only end is not usable.
    """
    has_ms_start, has_ms_end = TIME_PAIR_MS[0] in columns, TIME_PAIR_MS[1] in columns
    has_s_start, has_s_end = TIME_PAIR_S[0] in columns, TIME_PAIR_S[1] in columns

    if has_ms_start and has_ms_end:
        if has_s_start and has_s_end:
            _issue(
                issues,
                "Info",
                "TIME_MS_PREFERRED",
                "Both second- and millisecond-level time windows detected; bin_start_ms / bin_end_ms will be preferred.",
            )
        return TIME_PAIR_MS[0], TIME_PAIR_MS[1], "ms"

    if has_s_start and has_s_end:
        return TIME_PAIR_S[0], TIME_PAIR_S[1], "s"

    time_cols_present = [c for c in (*TIME_PAIR_MS, *TIME_PAIR_S) if c in columns]
    if time_cols_present:
        _issue(
            issues,
            "Error",
            "TIME_PAIR_INCOMPLETE",
            "Incomplete time-window fields: both start and end are required; cannot provide only start or only end.",
            "Please provide bin_start_ms + bin_end_ms, or bin_start_s + bin_end_s.",
        )
    else:
        _issue(
            issues,
            "Error",
            "TIME_PAIR_MISSING",
            "Missing time-window fields: bin start/end are required.",
            "Please provide bin_start_ms + bin_end_ms, or bin_start_s + bin_end_s.",
        )
    return None, None, None


def _build_empty_result(issues: list[ValidationIssue]) -> ValidationResult:
    """Return a consistent non-usable result when schema-level errors block normalization."""
    return ValidationResult(
        status=STATUS_ERROR,
        usable=False,
        cleaned_df=None,
        issues=issues,
        summary={},
    )


# -----------------------------
# Public validation function
# -----------------------------
def validate_spike_bin_dataframe(
    df: pd.DataFrame,
    *,
    coordinate_tolerance_um: float = 1e-3,
) -> ValidationResult:
    """
    Validate and normalize one spike-bin dataframe.

    Parameters
    ----------
    df:
        Raw dataframe loaded from the uploaded CSV.
    coordinate_tolerance_um:
        Small tolerance for repeated electrode coordinates. If one electrode has tiny
        floating-point coordinate jitter within this tolerance, coordinates are corrected
        to that electrode's median coordinate. Larger conflicts are reported as Error.

    Returns
    -------
    ValidationResult
        Includes status, cleaned_df, issues, downstream flags, and summary statistics.
    """
    issues: list[ValidationIssue] = []

    if df is None or df.empty:
        _issue(
            issues,
            "Error",
            "EMPTY_FILE",
            "Uploaded file is empty; nothing to analyze.",
            "Please upload a spike-bin CSV containing at least electrode_id, a time window, and a spike metric (spike_count or firing_rate_hz).",
        )
        return _build_empty_result(issues)

    # Keep original input unchanged. All standardization happens on a copy.
    work = df.copy()
    columns = set(work.columns)

    # ---- 1. Minimal schema checks ----
    if "electrode_id" not in columns:
        _issue(
            issues,
            "Error",
            "ELECTRODE_ID_MISSING",
            "Missing required field: electrode_id.",
            "Please add electrode_id to uniquely identify each electrode/channel.",
        )

    start_col, end_col, time_unit = _choose_time_pair(columns, issues)

    has_spike = "spike_count" in columns
    has_fr = "firing_rate_hz" in columns
    if not has_spike and not has_fr:
        _issue(
            issues,
            "Error",
            "METRIC_MISSING",
            "Missing spike metric: at least one of spike_count or firing_rate_hz is required.",
            "Please provide spike_count or firing_rate_hz (both are also accepted).",
        )

    # If required schema is not present, stop before accessing missing columns.
    if any(x.level == "Error" for x in issues):
        return _build_empty_result(issues)

    assert start_col is not None and end_col is not None and time_unit is not None

    # ---- 2. Numeric strictness for all non-text columns ----
    # Project rule: except recording_id / trial_id / condition / electrode_id,
    # all columns must be numeric. This includes extra user-provided columns.
    numeric_columns = [c for c in work.columns if c not in TEXT_COLUMNS]
    parsed_numeric: dict[str, pd.Series] = {}

    for col in numeric_columns:
        numeric = _series_to_numeric(work[col])
        parsed_numeric[col] = numeric
        bad_mask = _bad_numeric_mask(work[col], numeric)
        if bad_mask.any():
            _issue(
                issues,
                "Error",
                "NON_NUMERIC_VALUE",
                f"Column {col} contains non-numeric values.",
                "All columns except recording_id, trial_id, condition, and electrode_id must be numeric.",
                affected_rows=int(bad_mask.sum()),
            )

    # Required numeric columns cannot contain empty cells.
    required_numeric_cols = [start_col, end_col]
    if has_spike:
        required_numeric_cols.append("spike_count")
    if has_fr:
        required_numeric_cols.append("firing_rate_hz")

    for col in required_numeric_cols:
        missing_mask = parsed_numeric[col].isna()
        if missing_mask.any():
            _issue(
                issues,
                "Error",
                "REQUIRED_NUMERIC_MISSING",
                f"Required numeric field {col} contains missing or non-parsable values.",
                "Please fill this column, or remove rows with unparseable values and re-upload.",
                affected_rows=int(missing_mask.sum()),
            )

    if any(x.level == "Error" for x in issues):
        return _build_empty_result(issues)

    # From this point on, all core numeric fields are safe to use.
    for col, numeric in parsed_numeric.items():
        work[col] = numeric

    # ---- 3. Build canonical millisecond time-window columns ----
    # Internal downstream modules should use bin_start_ms / bin_end_ms / bin_width_ms.
    # This is not arbitrary field completion; it is the explicitly allowed time-unit normalization.
    if time_unit == "ms":
        work["bin_start_ms"] = work[start_col]
        work["bin_end_ms"] = work[end_col]
    else:
        work["bin_start_ms"] = work[start_col] * 1000.0
        work["bin_end_ms"] = work[end_col] * 1000.0
        _issue(
            issues,
            "Info",
            "TIME_SECONDS_CONVERTED",
            "Second-level time windows detected and internally converted to bin_start_ms / bin_end_ms.",
        )

    work["bin_width_ms"] = work["bin_end_ms"] - work["bin_start_ms"]

    bad_time = work["bin_width_ms"] <= 0
    if bad_time.any():
        _issue(
            issues,
            "Error",
            "INVALID_TIME_WINDOW",
            "There are rows with bin_end <= bin_start.",
            "Please ensure every row has bin_end > bin_start.",
            affected_rows=int(bad_time.sum()),
        )

    # ---- 4. Metric normalization: spike_count <-> firing_rate_hz ----
    # spike_count can be numeric decimals in the uploaded CSV; internally round to integer.
    if has_spike:
        fractional_spike = (work["spike_count"] % 1).abs() > 1e-12
        if fractional_spike.any():
            _issue(
                issues,
                "Warning",
                "SPIKE_COUNT_ROUNDED",
                "spike_count contains decimals; values have been rounded to integers.",
                "If those decimals were intentional, consider using firing_rate_hz instead.",
                affected_rows=int(fractional_spike.sum()),
            )
        work["spike_count"] = np.rint(work["spike_count"]).astype(int)

    # Only one metric exists: complete the other one using the bin width.
    # Formula: firing_rate_hz = spike_count / bin_width_s
    #          spike_count = firing_rate_hz * bin_width_s, rounded to nearest integer.
    if has_spike and not has_fr:
        work["firing_rate_hz"] = work["spike_count"] / (work["bin_width_ms"] / 1000.0)
        metric_source = "spike_count"
        _issue(
            issues,
            "Info",
            "FIRING_RATE_COMPLETED",
            "Only spike_count detected; firing_rate_hz has been completed using bin width.",
        )
    elif has_fr and not has_spike:
        work["spike_count"] = np.rint(work["firing_rate_hz"] * (work["bin_width_ms"] / 1000.0)).astype(int)
        metric_source = "firing_rate_hz"
        _issue(
            issues,
            "Info",
            "SPIKE_COUNT_COMPLETED",
            "Only firing_rate_hz detected; spike_count has been completed using bin_width_ms.",
        )
    else:
        metric_source = "spike_count + firing_rate_hz"
        _issue(
            issues,
            "Info",
            "BOTH_METRICS_FOUND",
            "Both spike_count and firing_rate_hz detected; both are retained and spike_count normalized to integers.",
        )

    if any(x.level == "Error" for x in issues):
        return _build_empty_result(issues)

    # ---- 5. Optional spatial analysis gate ----
    spatial_available = False
    has_x, has_y = "x_um" in columns, "y_um" in columns

    if has_x and has_y:
        coord_missing = work[["x_um", "y_um"]].isna().any(axis=1)
        if coord_missing.any():
            _issue(
                issues,
                "Warning",
                "COORDINATE_MISSING",
                "x_um / y_um contain missing values; spatial analysis will be disabled for this upload.",
                "If you need a Spatial Activity Map, ensure every row contains both x_um and y_um.",
                affected_rows=int(coord_missing.sum()),
            )
        else:
            # For each electrode_id, coordinates must be unique up to a small tolerance.
            # If within tolerance, correct all rows for that electrode to the median coordinate.
            corrected_electrodes = 0
            conflict_electrodes: list[str] = []

            for electrode_id, group in work.groupby("electrode_id", dropna=False):
                x_range = group["x_um"].max() - group["x_um"].min()
                y_range = group["y_um"].max() - group["y_um"].min()

                if x_range <= coordinate_tolerance_um and y_range <= coordinate_tolerance_um:
                    median_x = float(group["x_um"].median())
                    median_y = float(group["y_um"].median())
                    idx = group.index
                    if x_range > 0 or y_range > 0:
                        corrected_electrodes += 1
                    work.loc[idx, "x_um"] = median_x
                    work.loc[idx, "y_um"] = median_y
                else:
                    conflict_electrodes.append(str(electrode_id))

            if conflict_electrodes:
                _issue(
                    issues,
                    "Error",
                    "ELECTRODE_COORDINATE_CONFLICT",
                    "The same electrode_id corresponds to multiple distinct x_um / y_um coordinates.",
                    "Ensure each electrode_id has a unique spatial coordinate; tiny floating-point jitter is automatically corrected.",
                    affected_rows=len(conflict_electrodes),
                )
            else:
                spatial_available = True
                if corrected_electrodes:
                    _issue(
                        issues,
                        "Warning",
                        "COORDINATE_JITTER_FIXED",
                        "Detected tiny floating-point jitter in electrode coordinates; coordinates were unified per electrode_id.",
                        affected_rows=corrected_electrodes,
                    )
                _issue(
                    issues,
                    "Info",
                    "SPATIAL_ANALYSIS_READY",
                    "Complete and usable x_um / y_um detected; spatial analysis plots will be available.",
                )
    elif has_x != has_y:
        _issue(
            issues,
            "Warning",
            "COORDINATE_PAIR_INCOMPLETE",
            "Only one of x_um or y_um was detected; spatial analysis will be disabled.",
            "If you need spatial analysis, provide both x_um and y_um.",
        )
    else:
        _issue(
            issues,
            "Info",
            "SPATIAL_ANALYSIS_SKIPPED",
            "x_um / y_um not provided; data remains analyzable but spatial plots will not be generated.",
        )

    if any(x.level == "Error" for x in issues):
        return _build_empty_result(issues)

    # ---- 6. Optional condition-comparison gate ----
    condition_compare_available = False
    if "condition" in columns:
        n_conditions = int(work["condition"].dropna().nunique())
        if n_conditions > 1:
            condition_compare_available = True
            _issue(
                issues,
                "Info",
                "CONDITION_COMPARISON_READY",
                f"Detected {n_conditions} conditions; condition comparison analysis will be enabled.",
            )
        else:
            _issue(
                issues,
                "Info",
                "CONDITION_COMPARISON_SKIPPED_SINGLE",
                "condition does not have multiple values; condition comparison analysis will be skipped.",
            )
    else:
        n_conditions = 0
        _issue(
            issues,
            "Info",
            "CONDITION_COMPARISON_SKIPPED_MISSING",
            "No condition field uploaded; condition comparison analysis will be skipped.",
        )

    # ---- 7. Compact summary for UI display ----
    summary = {
        "n_rows": int(len(work)),
        "n_electrodes": int(work["electrode_id"].nunique(dropna=True)),
        "n_recordings": int(work["recording_id"].nunique(dropna=True)) if "recording_id" in work.columns else None,
        "n_trials": int(work["trial_id"].nunique(dropna=True)) if "trial_id" in work.columns else None,
        "n_conditions": n_conditions,
        "time_unit_used": time_unit,
        "min_time_ms": float(work["bin_start_ms"].min()),
        "max_time_ms": float(work["bin_end_ms"].max()),
        "median_bin_width_ms": float(work["bin_width_ms"].median()),
        "spatial_available": spatial_available,
        "condition_compare_available": condition_compare_available,
    }

    has_warning = any(x.level == "Warning" for x in issues)
    status = STATUS_WARNING if has_warning else STATUS_INFO

    return ValidationResult(
        status=status,
        usable=True,
        cleaned_df=work,
        issues=issues,
        summary=summary,
        spatial_available=spatial_available,
        condition_compare_available=condition_compare_available,
        time_unit_used=time_unit,
        metric_source=metric_source,
    )


def issues_to_dataframe(issues: list[ValidationIssue]) -> pd.DataFrame:
    """Convert validation issues to a Streamlit-friendly dataframe."""
    return pd.DataFrame(
        [
            {
                "level": x.level,
                "code": x.code,
                "message": x.message,
                "suggestion": x.suggestion,
                "affected_rows": x.affected_rows,
            }
            for x in issues
        ]
    )
