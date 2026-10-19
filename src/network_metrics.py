"""
Network-structure metric utilities for organoid-spike-visual-analytics.

Design boundary
---------------
This module receives `SpikeFeatureContext` from features.py and uses the shared
connectivity edge table from connectivity.py to compute graph-level metrics for
Metrics Change Dashboard.

It deliberately does not validate uploaded CSV files, build spike features,
compute correlations, or plot anything. The intended workflow is:

    cleaned_df -> SpikeFeatureContext(cleaned_df)
               -> connectivity.prepare_connectivity_edge_table(ctx)
               -> build_adjacency_matrix()
               -> compute_global_efficiency()
               -> compute_clustering_coefficient()
               -> compute_modularity()
               -> prepare_network_metrics_dashboard_table()

The first-stage implementation uses deterministic, dependency-light graph
calculation with numpy / pandas only. Community labels for modularity are derived
from connected components, so results remain stable and reproducible without
introducing an external community-detection package.
"""

from __future__ import annotations

from typing import Hashable, Optional

import numpy as np
import pandas as pd

try:  # Package import when used as src.network_metrics
    from .features import SpikeFeatureContext
    from .connectivity import (
        CorrelationMethod,
        ThresholdMode,
        prepare_connectivity_edge_table,
    )
except ImportError:  # Direct-file import during simple local checks
    from features import SpikeFeatureContext  # type: ignore
    from connectivity import (  # type: ignore
        CorrelationMethod,
        ThresholdMode,
        prepare_connectivity_edge_table,
    )


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

ADJACENCY_WEIGHT_COLUMNS = ("weight", "abs_weight", "positive_weight")

NETWORK_METRIC_COLUMNS = [
    "global_efficiency",
    "clustering_coefficient",
    "modularity",
    "n_nodes",
    "n_edges_used",
    "strong_only",
    "weight_col",
    "threshold",
    "threshold_mode",
]


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def _condition_cache_key(condition: Optional[Hashable]) -> str:
    """Make a stable cache-key fragment for optional condition filters."""
    return "__all__" if condition is None else f"condition={condition!r}"



def _empty_adjacency_matrix(electrode_order: Optional[list[Hashable]] = None) -> pd.DataFrame:
    """Return a stable empty or zero-filled square adjacency matrix."""
    if electrode_order is None:
        out = pd.DataFrame(dtype=float)
    else:
        out = pd.DataFrame(0.0, index=electrode_order, columns=electrode_order)
    out.index.name = "electrode_id"
    out.columns.name = "electrode_id"
    return out



def _validate_weight_col(weight_col: str) -> None:
    """Restrict adjacency construction to known edge-weight columns."""
    if weight_col not in ADJACENCY_WEIGHT_COLUMNS:
        allowed = ", ".join(ADJACENCY_WEIGHT_COLUMNS)
        raise ValueError(f"weight_col must be one of: {allowed}")



def _validate_adjacency_matrix(adjacency: pd.DataFrame) -> None:
    """Check the minimal contract for a square node-by-node adjacency matrix."""
    if not isinstance(adjacency, pd.DataFrame):
        raise TypeError("adjacency must be a pandas DataFrame")
    if adjacency.index.hasnans:
        raise ValueError("adjacency index contains missing electrode ids")
    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square")
    missing_columns = [node for node in adjacency.index.tolist() if node not in adjacency.columns]
    if missing_columns:
        missing_text = ", ".join(map(str, missing_columns))
        raise ValueError(f"adjacency columns must match index; missing columns: {missing_text}")



def _adjacency_array(adjacency: pd.DataFrame) -> np.ndarray:
    """Return a clean, non-negative numeric adjacency array."""
    _validate_adjacency_matrix(adjacency)
    if adjacency.empty:
        return np.zeros((0, 0), dtype=float)

    ordered = adjacency.reindex(index=adjacency.index, columns=adjacency.index)
    values = ordered.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    values = np.where(np.isfinite(values), values, 0.0)
    values = np.clip(values, 0.0, None)
    np.fill_diagonal(values, 0.0)
    return values



def _edge_count_from_adjacency(adjacency: pd.DataFrame) -> int:
    """Count undirected positive-weight edges in the upper triangle."""
    values = _adjacency_array(adjacency)
    if values.size == 0:
        return 0
    return int(np.count_nonzero(np.triu(values > 0, k=1)))



def _connected_components_from_array(values: np.ndarray) -> list[np.ndarray]:
    """Return connected components from a non-negative adjacency array."""
    n_nodes = values.shape[0]
    if n_nodes == 0:
        return []

    binary = values > 0
    visited = np.zeros(n_nodes, dtype=bool)
    components: list[np.ndarray] = []

    for start in range(n_nodes):
        if visited[start]:
            continue

        stack = [start]
        visited[start] = True
        component: list[int] = []

        while stack:
            node = stack.pop()
            component.append(node)
            neighbors = np.flatnonzero(binary[node])
            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(int(neighbor))

        components.append(np.array(component, dtype=int))

    return components


# -----------------------------------------------------------------------------
# Public stateless computation functions
# -----------------------------------------------------------------------------


def build_adjacency_matrix(
    edge_table: pd.DataFrame,
    *,
    electrode_order: Optional[list[Hashable]] = None,
    strong_only: bool = True,
    weight_col: str = "positive_weight",
) -> pd.DataFrame:
    """
    Convert the shared connectivity edge table into an undirected adjacency matrix.

    Parameters
    ----------
    edge_table:
        Connectivity edge table produced by connectivity.prepare_connectivity_edge_table().
    electrode_order:
        Optional full node order. When provided, isolated electrodes are retained
        as zero-degree rows and columns.
    strong_only:
        Whether to use only thresholded strong edges.
    weight_col:
        Edge weight used in the graph. The default positive_weight keeps the
        project's first-stage network metrics focused on positive functional links.

    Returns
    -------
    pandas.DataFrame
        Square node-by-node adjacency matrix. Non-positive and missing weights
        are ignored.
    """
    _validate_weight_col(weight_col)

    if edge_table is None or edge_table.empty:
        return _empty_adjacency_matrix(electrode_order)

    required = {"source_electrode_id", "target_electrode_id"}
    if strong_only:
        required.add("is_strong")
    missing = required.difference(edge_table.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"edge_table missing columns: {missing_text}")

    work = edge_table.copy()
    if weight_col not in work.columns:
        if "weight" not in work.columns:
            raise ValueError(f"edge_table missing columns: {weight_col}")
        base_weight = pd.to_numeric(work["weight"], errors="coerce")
        if weight_col == "positive_weight":
            work[weight_col] = base_weight.clip(lower=0.0)
        elif weight_col == "abs_weight":
            work[weight_col] = base_weight.abs()
        else:
            work[weight_col] = base_weight

    if strong_only:
        work = work[work["is_strong"].astype(bool)].copy()

    work[weight_col] = pd.to_numeric(work[weight_col], errors="coerce")
    work = work.dropna(subset=["source_electrode_id", "target_electrode_id", weight_col])
    work = work[work[weight_col] > 0].copy()

    if electrode_order is None:
        electrode_order = pd.concat(
            [edge_table["source_electrode_id"], edge_table["target_electrode_id"]],
            ignore_index=True,
        ).dropna().drop_duplicates().tolist()

    adjacency = _empty_adjacency_matrix(electrode_order)
    if work.empty or adjacency.empty:
        return adjacency

    known_nodes = set(adjacency.index.tolist())
    for row in work.itertuples(index=False):
        source = getattr(row, "source_electrode_id")
        target = getattr(row, "target_electrode_id")
        if source not in known_nodes or target not in known_nodes or source == target:
            continue
        weight = float(getattr(row, weight_col))
        current = float(adjacency.loc[source, target])
        adjacency.loc[source, target] = max(current, weight)
        adjacency.loc[target, source] = max(current, weight)

    return adjacency.astype(float)



def compute_global_efficiency(adjacency: pd.DataFrame, *, weighted: bool = True) -> float:
    """
    Compute global efficiency from an adjacency matrix.

    Positive edge weights are treated as connection strengths. In weighted mode,
    shortest-path distance for one edge is 1 / weight. Disconnected node pairs
    contribute 0 to efficiency.
    """
    values = _adjacency_array(adjacency)
    n_nodes = values.shape[0]
    if n_nodes < 2:
        return np.nan

    positive = values > 0
    if not bool(positive.any()):
        return 0.0

    distance = np.full((n_nodes, n_nodes), np.inf, dtype=float)
    np.fill_diagonal(distance, 0.0)

    if weighted:
        distance[positive] = 1.0 / values[positive]
    else:
        distance[positive] = 1.0

    for k in range(n_nodes):
        distance = np.minimum(distance, distance[:, [k]] + distance[[k], :])

    with np.errstate(divide="ignore", invalid="ignore"):
        efficiency = 1.0 / distance
    efficiency[~np.isfinite(efficiency)] = 0.0
    np.fill_diagonal(efficiency, 0.0)

    return float(efficiency.sum() / (n_nodes * (n_nodes - 1)))



def compute_clustering_coefficient(adjacency: pd.DataFrame) -> float:
    """
    Compute the mean binary clustering coefficient across all nodes.

    The metric uses the thresholded graph topology rather than raw edge weights.
    Nodes with fewer than two neighbors contribute 0.
    """
    values = _adjacency_array(adjacency)
    n_nodes = values.shape[0]
    if n_nodes == 0:
        return np.nan
    if n_nodes < 3:
        return 0.0

    binary = values > 0
    coefficients: list[float] = []

    for node in range(n_nodes):
        neighbors = np.flatnonzero(binary[node])
        degree = len(neighbors)
        if degree < 2:
            coefficients.append(0.0)
            continue

        neighbor_graph = binary[np.ix_(neighbors, neighbors)]
        actual_edges = int(np.count_nonzero(np.triu(neighbor_graph, k=1)))
        possible_edges = degree * (degree - 1) / 2
        coefficients.append(float(actual_edges / possible_edges))

    return float(np.mean(coefficients))



def compute_modularity(adjacency: pd.DataFrame) -> float:
    """
    Compute weighted modularity using connected components as communities.

    This first-stage choice avoids non-deterministic community detection while
    still giving the dashboard a stable module-separation signal. If the graph
    has no positive-weight edges, modularity is 0.
    """
    values = _adjacency_array(adjacency)
    n_nodes = values.shape[0]
    if n_nodes == 0:
        return np.nan

    total_edge_weight = float(values.sum() / 2.0)
    if total_edge_weight <= 0:
        return 0.0

    strengths = values.sum(axis=1)
    components = _connected_components_from_array(values)
    q = 0.0
    two_m = 2.0 * total_edge_weight

    for component in components:
        subgraph_weight = float(values[np.ix_(component, component)].sum())
        component_strength = float(strengths[component].sum())
        q += (subgraph_weight / two_m) - (component_strength / two_m) ** 2

    q = float(q)
    return 0.0 if abs(q) < 1e-12 else q


# -----------------------------------------------------------------------------
# Prepare-layer function
# -----------------------------------------------------------------------------


def prepare_network_metrics_dashboard_table(
    ctx: SpikeFeatureContext,
    *,
    value_col: str = "spike_count",
    method: CorrelationMethod = "pearson",
    min_periods: int = 2,
    threshold: float = 0.5,
    threshold_mode: ThresholdMode = "positive",
    strong_only: bool = True,
    weight_col: str = "positive_weight",
) -> pd.DataFrame:
    """
    Prepare graph-theory metrics for Metrics Change Dashboard.

    The output is condition-level when condition comparison is available. When no
    comparison is available, it returns one row with condition="all". dashboard_data.py
    should merge this table with feature-level and connectivity-level dashboard
    tables.
    """
    _validate_weight_col(weight_col)

    cache_key = (
        f"network_metrics|dashboard|value={value_col}|method={method}|"
        f"min_periods={min_periods}|threshold={threshold}|mode={threshold_mode}|"
        f"strong_only={strong_only}|weight={weight_col}"
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
        adjacency = build_adjacency_matrix(
            edges,
            electrode_order=ctx.electrode_order,
            strong_only=strong_only,
            weight_col=weight_col,
        )

        row = {
            "condition": label,
            "scope": scope,
            "global_efficiency": compute_global_efficiency(adjacency),
            "clustering_coefficient": compute_clustering_coefficient(adjacency),
            "modularity": compute_modularity(adjacency),
            "n_nodes": int(len(ctx.electrode_order)),
            "n_edges_used": _edge_count_from_adjacency(adjacency),
            "strong_only": bool(strong_only),
            "weight_col": weight_col,
            "threshold": float(threshold),
            "threshold_mode": threshold_mode,
        }
        rows.append(row)

    columns = ["condition", "scope"] + NETWORK_METRIC_COLUMNS
    out = pd.DataFrame(rows, columns=columns)
    ctx.set_cached_table(cache_key, out)
    return out.copy()
