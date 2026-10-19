"""
Functional connectivity utilities for organoid-spike-visual-analytics.

Design boundary
---------------
This module receives `SpikeFeatureContext` from features.py and builds
connectivity-ready intermediate tables for downstream plotting, dashboard, and
network-metrics modules.

It deliberately does not plot anything and does not import visualization
libraries. The intended workflow is:

    cleaned_df -> SpikeFeatureContext(cleaned_df)
               -> features.prepare_electrode_time_matrix_data(ctx)
               -> compute_correlation_matrix()
               -> compute_connectivity_edge_table()
               -> correlation matrix / spatial network / dashboard inputs

Core calculations stay in stateless compute_xxx functions. Plot-facing and
summary-facing assembly stays in prepare_xxx functions.
"""

from __future__ import annotations

from typing import Hashable, Literal, Optional

import numpy as np
import pandas as pd

try:  # Package import when used as src.connectivity
    from .features import (
        SpikeFeatureContext,
        prepare_electrode_time_matrix_data,
        prepare_spatial_node_table,
    )
except ImportError:  # Direct-file import during simple local checks
    from features import (  # type: ignore
        SpikeFeatureContext,
        prepare_electrode_time_matrix_data,
        prepare_spatial_node_table,
    )


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

CorrelationMethod = Literal["pearson", "spearman"]
ThresholdMode = Literal["absolute", "positive", "negative"]

VALUE_COLUMNS = ("spike_count", "firing_rate_hz")

CORRELATION_MATRIX_NAME = "correlation_weight"

EDGE_COLUMNS = [
    "source_electrode_id",
    "target_electrode_id",
    "weight",
    "abs_weight",
    "positive_weight",
    "sign",
    "threshold",
    "threshold_mode",
    "is_strong",
]

NODE_CONNECTIVITY_COLUMNS = [
    "electrode_id",
    "degree",
    "strength",
    "mean_abs_weight",
]

CONNECTIVITY_SUMMARY_COLUMNS = [
    "mean_connectivity_weight",
    "strong_connection_ratio",
    "network_density",
    "mean_degree",
    "n_nodes",
    "n_possible_edges",
    "n_valid_edges",
    "n_strong_edges",
    "threshold",
    "threshold_mode",
]


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def _condition_cache_key(condition: Optional[Hashable]) -> str:
    """Make a stable cache-key fragment for optional condition filters."""
    return "__all__" if condition is None else f"condition={condition!r}"



def _safe_divide(numerator: float, denominator: float, default: float = np.nan) -> float:
    """Return numerator / denominator, or default when denominator is not usable."""
    if pd.isna(denominator) or float(denominator) <= 0:
        return default
    return float(numerator) / float(denominator)



def _validate_value_col(value_col: str) -> None:
    """Restrict public prepare functions to project-standard value columns."""
    if value_col not in VALUE_COLUMNS:
        allowed = ", ".join(VALUE_COLUMNS)
        raise ValueError(f"value_col must be one of: {allowed}")



def _validate_correlation_method(method: str) -> None:
    """Restrict correlation method to pandas-supported methods used by this project."""
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be either 'pearson' or 'spearman'")



def _validate_threshold(threshold: float) -> None:
    """Keep threshold in the interpretable correlation-weight range."""
    if pd.isna(threshold) or float(threshold) < 0.0 or float(threshold) > 1.0:
        raise ValueError("threshold must be a number between 0 and 1")



def _validate_threshold_mode(threshold_mode: str) -> None:
    """Restrict threshold semantics to explicit edge-selection modes."""
    if threshold_mode not in {"absolute", "positive", "negative"}:
        raise ValueError("threshold_mode must be 'absolute', 'positive', or 'negative'")



def _validate_electrode_time_matrix(matrix: pd.DataFrame) -> None:
    """
    Check the minimal contract for an electrode x time matrix.

    Rows must be electrode ids. Columns are time bins. Values should be numeric
    activity values produced by features.prepare_electrode_time_matrix_data().
    """
    if not isinstance(matrix, pd.DataFrame):
        raise TypeError("electrode_time_matrix must be a pandas DataFrame")
    if matrix.index.hasnans:
        raise ValueError("electrode_time_matrix index contains missing electrode ids")



def _empty_correlation_matrix(electrode_order: Optional[list[Hashable]] = None) -> pd.DataFrame:
    """Return a stable empty or zero-sized square correlation matrix."""
    if electrode_order is None:
        out = pd.DataFrame(dtype=float)
    else:
        out = pd.DataFrame(0.0, index=electrode_order, columns=electrode_order)
    out.index.name = "electrode_id"
    out.columns.name = "electrode_id"
    return out



def _empty_edge_table(include_condition: bool = False) -> pd.DataFrame:
    """Return an empty edge-table schema."""
    columns = (["condition"] + EDGE_COLUMNS) if include_condition else EDGE_COLUMNS
    return pd.DataFrame(columns=columns)



def _empty_node_connectivity_table() -> pd.DataFrame:
    """Return an empty node-connectivity schema."""
    return pd.DataFrame(columns=NODE_CONNECTIVITY_COLUMNS)



def _empty_connectivity_summary() -> pd.DataFrame:
    """Return an empty connectivity-summary schema."""
    return pd.DataFrame(columns=CONNECTIVITY_SUMMARY_COLUMNS)



def _strong_mask(weights: pd.Series, *, threshold: float, threshold_mode: ThresholdMode) -> pd.Series:
    """Return the boolean mask defining strong edges."""
    if threshold_mode == "absolute":
        return weights.abs() >= threshold
    if threshold_mode == "positive":
        return weights >= threshold
    if threshold_mode == "negative":
        return weights <= -threshold
    raise ValueError("threshold_mode must be 'absolute', 'positive', or 'negative'")



def _edge_sign(weight: float) -> str:
    """Return positive / negative / zero label for one edge weight."""
    if weight > 0:
        return "positive"
    if weight < 0:
        return "negative"
    return "zero"


# -----------------------------------------------------------------------------
# Public stateless computation functions
# -----------------------------------------------------------------------------


def compute_correlation_matrix(
    electrode_time_matrix: pd.DataFrame,
    *,
    method: CorrelationMethod = "pearson",
    min_periods: int = 2,
    fill_diagonal: bool = True,
) -> pd.DataFrame:
    """
    Compute electrode-by-electrode functional connectivity.

    Parameters
    ----------
    electrode_time_matrix:
        Rows are electrode_id, columns are bin_start_ms, values are activity
        values such as spike_count or firing_rate_hz.
    method:
        Correlation method. The first-stage platform uses pearson by default.
    min_periods:
        Minimum valid time points required for one pairwise correlation.
    fill_diagonal:
        Whether to set self-correlation to 1.0.

    Returns
    -------
    pandas.DataFrame
        Square correlation matrix indexed and columned by electrode_id.
    """
    _validate_electrode_time_matrix(electrode_time_matrix)
    _validate_correlation_method(method)

    if electrode_time_matrix.empty:
        return _empty_correlation_matrix()

    matrix = electrode_time_matrix.apply(pd.to_numeric, errors="coerce").astype(float)
    electrode_order = matrix.index.tolist()

    if len(electrode_order) == 0:
        return _empty_correlation_matrix()

    if matrix.shape[1] < min_periods:
        corr = _empty_correlation_matrix(electrode_order)
    else:
        # pandas computes column-column correlations, so transpose first.
        corr = matrix.T.corr(method=method, min_periods=min_periods)
        corr = corr.reindex(index=electrode_order, columns=electrode_order)
        corr = corr.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

    if fill_diagonal and not corr.empty:
        for index in range(len(corr)):
            corr.iloc[index, index] = 1.0

    corr.index.name = "electrode_id"
    corr.columns.name = "electrode_id"
    return corr



def compute_connectivity_edge_table(
    correlation_matrix: pd.DataFrame,
    *,
    threshold: float = 0.5,
    threshold_mode: ThresholdMode = "positive",
) -> pd.DataFrame:
    """
    Convert a correlation matrix into an undirected edge table.

    The returned table keeps all upper-triangle electrode pairs and marks strong
    connections with `is_strong`. Plotting functions can filter strong edges,
    while dashboard functions can still compute ratios against all valid pairs.
    """
    _validate_threshold(threshold)
    _validate_threshold_mode(threshold_mode)

    if correlation_matrix is None or correlation_matrix.empty:
        return _empty_edge_table()

    corr = correlation_matrix.copy()
    electrodes = corr.index.tolist()
    if len(electrodes) < 2:
        return _empty_edge_table()

    missing_columns = [electrode for electrode in electrodes if electrode not in corr.columns]
    if missing_columns:
        missing_text = ", ".join(map(str, missing_columns))
        raise ValueError(f"correlation_matrix must be square; missing columns: {missing_text}")

    corr = corr.reindex(index=electrodes, columns=electrodes)
    rows: list[dict[str, object]] = []

    for i, source in enumerate(electrodes):
        for j in range(i + 1, len(electrodes)):
            target = electrodes[j]
            weight = pd.to_numeric(pd.Series([corr.loc[source, target]]), errors="coerce").iloc[0]
            if pd.isna(weight):
                continue
            weight = float(weight)
            rows.append(
                {
                    "source_electrode_id": source,
                    "target_electrode_id": target,
                    "weight": weight,
                    "abs_weight": abs(weight),
                    "positive_weight": max(weight, 0.0),
                    "sign": _edge_sign(weight),
                    "threshold": float(threshold),
                    "threshold_mode": threshold_mode,
                    "is_strong": bool(
                        _strong_mask(
                            pd.Series([weight]),
                            threshold=threshold,
                            threshold_mode=threshold_mode,
                        ).iloc[0]
                    ),
                }
            )

    return pd.DataFrame(rows, columns=EDGE_COLUMNS)



def compute_node_connectivity_table(
    edge_table: pd.DataFrame,
    *,
    electrode_order: Optional[list[Hashable]] = None,
    strong_only: bool = True,
) -> pd.DataFrame:
    """
    Compute node-level degree and strength from an edge table.

    Degree is the number of incident edges. Strength is the sum of incident
    positive weights. By default only strong edges are used because spatial
    network node size should reflect thresholded hub tendency.
    """
    if edge_table is None or edge_table.empty:
        if electrode_order is None:
            return _empty_node_connectivity_table()
        return pd.DataFrame(
            {
                "electrode_id": electrode_order,
                "degree": 0,
                "strength": 0.0,
                "mean_abs_weight": 0.0,
            },
            columns=NODE_CONNECTIVITY_COLUMNS,
        )

    required = {"source_electrode_id", "target_electrode_id", "abs_weight", "is_strong"}
    missing = required.difference(edge_table.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"edge_table missing columns: {missing_text}")

    work = edge_table.copy()
    if "positive_weight" not in work.columns:
        if "weight" not in work.columns:
            raise ValueError("edge_table missing columns: positive_weight")
        work["positive_weight"] = pd.to_numeric(work["weight"], errors="coerce").clip(lower=0)
    work["positive_weight"] = pd.to_numeric(work["positive_weight"], errors="coerce").fillna(0.0)

    if strong_only:
        work = work[work["is_strong"].astype(bool)]

    if electrode_order is None:
        electrode_order = pd.concat(
            [edge_table["source_electrode_id"], edge_table["target_electrode_id"]],
            ignore_index=True,
        ).dropna().drop_duplicates().tolist()

    if work.empty:
        return pd.DataFrame(
            {
                "electrode_id": electrode_order,
                "degree": 0,
                "strength": 0.0,
                "mean_abs_weight": 0.0,
            },
            columns=NODE_CONNECTIVITY_COLUMNS,
        )

    source_side = work[["source_electrode_id", "abs_weight", "positive_weight"]].rename(
        columns={"source_electrode_id": "electrode_id"}
    )
    target_side = work[["target_electrode_id", "abs_weight", "positive_weight"]].rename(
        columns={"target_electrode_id": "electrode_id"}
    )
    incident = pd.concat([source_side, target_side], ignore_index=True)

    out = (
        incident.groupby("electrode_id", as_index=False, sort=False)
        .agg(
            degree=("abs_weight", "size"),
            strength=("positive_weight", "sum"),
            mean_abs_weight=("abs_weight", "mean"),
        )
    )

    order = pd.DataFrame({"electrode_id": electrode_order})
    out = order.merge(out, on="electrode_id", how="left")
    out["degree"] = out["degree"].fillna(0).astype(int)
    out["strength"] = out["strength"].fillna(0.0).astype(float)
    out["mean_abs_weight"] = out["mean_abs_weight"].fillna(0.0).astype(float)
    return out[NODE_CONNECTIVITY_COLUMNS]



def compute_connectivity_summary(
    edge_table: pd.DataFrame,
    *,
    n_nodes: int,
    threshold: float = 0.5,
    threshold_mode: ThresholdMode = "positive",
) -> dict[str, float | int | str]:
    """
    Compute connection-level metrics for Metrics Change Dashboard.

    These are not advanced graph-theory metrics. global_efficiency,
    clustering_coefficient, and modularity should be computed in
    network_metrics.py using the same edge table or adjacency matrix.
    """
    _validate_threshold(threshold)
    _validate_threshold_mode(threshold_mode)

    n_nodes = int(n_nodes)
    n_possible_edges = int(n_nodes * (n_nodes - 1) / 2) if n_nodes >= 2 else 0

    if edge_table is None or edge_table.empty:
        return {
            "mean_connectivity_weight": np.nan,
            "strong_connection_ratio": np.nan,
            "network_density": np.nan,
            "mean_degree": np.nan,
            "n_nodes": n_nodes,
            "n_possible_edges": n_possible_edges,
            "n_valid_edges": 0,
            "n_strong_edges": 0,
            "threshold": float(threshold),
            "threshold_mode": threshold_mode,
        }

    required = {"abs_weight", "is_strong"}
    missing = required.difference(edge_table.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"edge_table missing columns: {missing_text}")

    valid_edges = edge_table[pd.to_numeric(edge_table["abs_weight"], errors="coerce").notna()].copy()
    n_valid_edges = int(len(valid_edges))
    n_strong_edges = int(valid_edges["is_strong"].astype(bool).sum())

    return {
        "mean_connectivity_weight": float(valid_edges["abs_weight"].mean()) if n_valid_edges else np.nan,
        "strong_connection_ratio": _safe_divide(n_strong_edges, n_valid_edges),
        "network_density": _safe_divide(n_strong_edges, n_possible_edges),
        "mean_degree": _safe_divide(2 * n_strong_edges, n_nodes),
        "n_nodes": n_nodes,
        "n_possible_edges": n_possible_edges,
        "n_valid_edges": n_valid_edges,
        "n_strong_edges": n_strong_edges,
        "threshold": float(threshold),
        "threshold_mode": threshold_mode,
    }


# -----------------------------------------------------------------------------
# Prepare-layer functions
# -----------------------------------------------------------------------------


def prepare_spatial_network_connectivity_data(
    edge_table: pd.DataFrame,
    *,
    electrode_order: Optional[list[Hashable]] = None,
    strong_only: bool = True,
) -> dict[str, pd.DataFrame]:
    """Prepare connectivity-only inputs for a spatial network graph."""
    include_condition = edge_table is not None and "condition" in edge_table.columns

    if edge_table is None or edge_table.empty:
        return {
            "edges": _empty_edge_table(include_condition=include_condition),
            "node_connectivity": compute_node_connectivity_table(
                edge_table,
                electrode_order=electrode_order,
                strong_only=strong_only,
            ),
        }

    edges = edge_table.copy()
    if strong_only and "is_strong" in edges.columns:
        edges = edges[edges["is_strong"].astype(bool)].copy()

    node_connectivity = compute_node_connectivity_table(
        edge_table,
        electrode_order=electrode_order,
        strong_only=strong_only,
    )
    return {
        "edges": edges.reset_index(drop=True),
        "node_connectivity": node_connectivity,
    }


def prepare_correlation_matrix_data(
    ctx: SpikeFeatureContext,
    *,
    value_col: str = "spike_count",
    condition: Optional[Hashable] = None,
    method: CorrelationMethod = "pearson",
    min_periods: int = 2,
) -> pd.DataFrame:
    """Prepare the correlation matrix for Correlation Matrix plotting."""
    _validate_value_col(value_col)
    _validate_correlation_method(method)

    cache_key = (
        f"connectivity|corr|value={value_col}|{_condition_cache_key(condition)}|"
        f"method={method}|min_periods={min_periods}"
    )
    cached = ctx.get_cached_table(cache_key)
    if cached is not None:
        return cached

    matrix = prepare_electrode_time_matrix_data(
        ctx,
        value_col=value_col,
        condition=condition,
        use_global_time_order=False,
    )
    corr = compute_correlation_matrix(
        matrix,
        method=method,
        min_periods=min_periods,
        fill_diagonal=True,
    )
    ctx.set_cached_table(cache_key, corr)
    return corr.copy()



def prepare_connectivity_edge_table(
    ctx: SpikeFeatureContext,
    *,
    value_col: str = "spike_count",
    condition: Optional[Hashable] = None,
    method: CorrelationMethod = "pearson",
    min_periods: int = 2,
    threshold: float = 0.5,
    threshold_mode: ThresholdMode = "positive",
) -> pd.DataFrame:
    """
    Prepare the shared connectivity edge table.

    This is the common intermediate interface for spatial network graphs,
    connectivity dashboard summaries, exports, notebooks, and network_metrics.py.
    """
    _validate_value_col(value_col)
    _validate_correlation_method(method)
    _validate_threshold(threshold)
    _validate_threshold_mode(threshold_mode)

    cache_key = (
        f"connectivity|edges|value={value_col}|{_condition_cache_key(condition)}|"
        f"method={method}|min_periods={min_periods}|threshold={threshold}|"
        f"mode={threshold_mode}"
    )
    cached = ctx.get_cached_table(cache_key)
    if cached is not None:
        return cached

    corr = prepare_correlation_matrix_data(
        ctx,
        value_col=value_col,
        condition=condition,
        method=method,
        min_periods=min_periods,
    )
    edges = compute_connectivity_edge_table(
        corr,
        threshold=threshold,
        threshold_mode=threshold_mode,
    )
    if condition is not None:
        edges.insert(0, "condition", condition)

    ctx.set_cached_table(cache_key, edges)
    return edges.copy()



def prepare_spatial_network_data(
    ctx: SpikeFeatureContext,
    *,
    value_col: str = "spike_count",
    condition: Optional[Hashable] = None,
    method: CorrelationMethod = "pearson",
    min_periods: int = 2,
    threshold: float = 0.5,
    threshold_mode: ThresholdMode = "positive",
    strong_only: bool = True,
) -> dict[str, pd.DataFrame | dict[str, bool]]:
    """
    Prepare nodes and edges for Spatial Network Reorganization Graph.

    Node coordinates and activity-color fields come from features.py. Connection
    strength and degree come from the edge table in this module.
    """
    _validate_value_col(value_col)
    _validate_correlation_method(method)
    _validate_threshold(threshold)
    _validate_threshold_mode(threshold_mode)

    gates = ctx.analysis_gates()
    include_condition = condition is not None

    if not ctx.spatial_available:
        return {
            "analysis_gates": gates,
            "nodes": pd.DataFrame(
                columns=[
                    "condition",
                    "electrode_id",
                    "x_um",
                    "y_um",
                    "mean_firing_rate_hz",
                    "total_spike_count",
                    "active_bin_ratio",
                    "degree",
                    "strength",
                    "mean_abs_weight",
                ]
            ),
            "edges": _empty_edge_table(include_condition=include_condition),
        }

    cache_key = (
        f"connectivity|spatial_network|value={value_col}|{_condition_cache_key(condition)}|"
        f"method={method}|min_periods={min_periods}|threshold={threshold}|"
        f"mode={threshold_mode}|strong_only={strong_only}"
    )
    cached_nodes = ctx.get_cached_table(f"{cache_key}|nodes")
    cached_edges = ctx.get_cached_table(f"{cache_key}|edges")
    if cached_nodes is not None and cached_edges is not None:
        return {
            "analysis_gates": gates,
            "nodes": cached_nodes,
            "edges": cached_edges,
        }

    edge_table = prepare_connectivity_edge_table(
        ctx,
        value_col=value_col,
        condition=condition,
        method=method,
        min_periods=min_periods,
        threshold=threshold,
        threshold_mode=threshold_mode,
    )
    connectivity_data = prepare_spatial_network_connectivity_data(
        edge_table,
        electrode_order=ctx.electrode_order,
        strong_only=strong_only,
    )

    spatial_nodes = prepare_spatial_node_table(ctx, condition=condition)
    node_connectivity = connectivity_data["node_connectivity"]
    nodes = spatial_nodes.merge(node_connectivity, on="electrode_id", how="left")
    for col in ["degree", "strength", "mean_abs_weight"]:
        nodes[col] = nodes[col].fillna(0)
    nodes["degree"] = nodes["degree"].astype(int)
    nodes["strength"] = nodes["strength"].astype(float)
    nodes["mean_abs_weight"] = nodes["mean_abs_weight"].astype(float)

    edges = connectivity_data["edges"].copy()
    if not edges.empty and not nodes.empty:
        coord_cols = ["electrode_id", "x_um", "y_um"]
        coords = nodes[coord_cols].drop_duplicates("electrode_id")
        source_coords = coords.rename(
            columns={
                "electrode_id": "source_electrode_id",
                "x_um": "source_x_um",
                "y_um": "source_y_um",
            }
        )
        target_coords = coords.rename(
            columns={
                "electrode_id": "target_electrode_id",
                "x_um": "target_x_um",
                "y_um": "target_y_um",
            }
        )
        edges = edges.merge(source_coords, on="source_electrode_id", how="left")
        edges = edges.merge(target_coords, on="target_electrode_id", how="left")

    nodes = nodes.reset_index(drop=True)
    edges = edges.reset_index(drop=True)
    ctx.set_cached_table(f"{cache_key}|nodes", nodes)
    ctx.set_cached_table(f"{cache_key}|edges", edges)

    return {
        "analysis_gates": gates,
        "nodes": nodes.copy(),
        "edges": edges.copy(),
    }



def prepare_connectivity_dashboard_table(
    ctx: SpikeFeatureContext,
    *,
    value_col: str = "spike_count",
    method: CorrelationMethod = "pearson",
    min_periods: int = 2,
    threshold: float = 0.5,
    threshold_mode: ThresholdMode = "positive",
) -> pd.DataFrame:
    """
    Prepare connection-level metrics for Metrics Change Dashboard.

    dashboard_data.py should merge this table with feature-level metrics from
    features.py and graph-theory metrics from network_metrics.py.
    """
    _validate_value_col(value_col)
    _validate_correlation_method(method)
    _validate_threshold(threshold)
    _validate_threshold_mode(threshold_mode)

    cache_key = (
        f"connectivity|dashboard|value={value_col}|method={method}|"
        f"min_periods={min_periods}|threshold={threshold}|mode={threshold_mode}"
    )
    cached = ctx.get_cached_table(cache_key)
    if cached is not None:
        return cached

    rows: list[dict[str, object]] = []
    scope = "condition" if ctx.condition_compare_available else "all"

    for label, _df_slice in ctx.condition_slices():
        condition = label if ctx.condition_compare_available else None
        edges = prepare_connectivity_edge_table(
            ctx,
            value_col=value_col,
            condition=condition,
            method=method,
            min_periods=min_periods,
            threshold=threshold,
            threshold_mode=threshold_mode,
        )
        summary = compute_connectivity_summary(
            edges,
            n_nodes=len(ctx.electrode_order),
            threshold=threshold,
            threshold_mode=threshold_mode,
        )
        row = {"condition": label, "scope": scope}
        row.update(summary)
        rows.append(row)

    columns = ["condition", "scope"] + CONNECTIVITY_SUMMARY_COLUMNS
    out = pd.DataFrame(rows, columns=columns)
    ctx.set_cached_table(cache_key, out)
    return out.copy()



def prepare_connectivity_bundle(
    ctx: SpikeFeatureContext,
    *,
    value_col: str = "spike_count",
    condition: Optional[Hashable] = None,
    method: CorrelationMethod = "pearson",
    min_periods: int = 2,
    threshold: float = 0.5,
    threshold_mode: ThresholdMode = "positive",
) -> dict[str, pd.DataFrame | dict[str, pd.DataFrame | dict[str, bool]] | dict[str, bool]]:
    """
    Convenience bundle for notebooks, exports, and evidence packs.

    The bundle keeps connectivity outputs together without becoming the main
    execution path for plotting functions.
    """
    spatial_network = prepare_spatial_network_data(
        ctx,
        value_col=value_col,
        condition=condition,
        method=method,
        min_periods=min_periods,
        threshold=threshold,
        threshold_mode=threshold_mode,
        strong_only=True,
    )
    return {
        "analysis_gates": ctx.analysis_gates(),
        "correlation_matrix": prepare_correlation_matrix_data(
            ctx,
            value_col=value_col,
            condition=condition,
            method=method,
            min_periods=min_periods,
        ),
        "edge_table": prepare_connectivity_edge_table(
            ctx,
            value_col=value_col,
            condition=condition,
            method=method,
            min_periods=min_periods,
            threshold=threshold,
            threshold_mode=threshold_mode,
        ),
        "spatial_network_nodes": spatial_network["nodes"],
        "spatial_network_edges": spatial_network["edges"],
        "connectivity_dashboard": prepare_connectivity_dashboard_table(
            ctx,
            value_col=value_col,
            method=method,
            min_periods=min_periods,
            threshold=threshold,
            threshold_mode=threshold_mode,
        ),
    }
