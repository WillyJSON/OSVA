import pandas as pd

from src.connectivity import prepare_connectivity_edge_table
from src.features import SpikeFeatureContext, prepare_electrode_time_matrix_data
from src.network_metrics import (
    build_adjacency_matrix,
    compute_clustering_coefficient,
    compute_global_efficiency,
    compute_modularity,
)
from src.validation import validate_spike_bin_dataframe


def test_feature_connectivity_network_pipeline_runs_end_to_end():
    raw = pd.DataFrame(
        {
            "electrode_id": ["E1", "E2", "E3"] * 4,
            "bin_start_ms": [0, 0, 0, 100, 100, 100, 200, 200, 200, 300, 300, 300],
            "bin_end_ms": [100, 100, 100, 200, 200, 200, 300, 300, 300, 400, 400, 400],
            "spike_count": [0, 0, 3, 1, 2, 2, 2, 4, 1, 3, 6, 0],
        }
    )
    result = validate_spike_bin_dataframe(raw)
    assert result.usable is True
    assert result.cleaned_df is not None

    context = SpikeFeatureContext(result.cleaned_df)
    matrix = prepare_electrode_time_matrix_data(context)
    edges = prepare_connectivity_edge_table(context, threshold=0.0)
    adjacency = build_adjacency_matrix(edges, electrode_order=context.electrode_order)

    assert matrix.shape == (3, 4)
    assert len(edges) == 3
    assert adjacency.shape == (3, 3)
    assert compute_global_efficiency(adjacency) > 0
    assert 0 <= compute_clustering_coefficient(adjacency) <= 1
    assert -1 <= compute_modularity(adjacency) <= 1
