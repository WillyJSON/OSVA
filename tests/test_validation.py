import pandas as pd

from src.validation import validate_spike_bin_dataframe


def test_validation_normalizes_second_windows_and_completes_rate():
    raw = pd.DataFrame(
        {
            "electrode_id": ["E1", "E2"],
            "bin_start_s": [0.0, 0.0],
            "bin_end_s": [0.5, 0.5],
            "spike_count": [2, 1],
        }
    )

    result = validate_spike_bin_dataframe(raw)

    assert result.usable is True
    assert result.time_unit_used == "s"
    assert result.metric_source == "spike_count"
    assert result.cleaned_df is not None
    assert result.cleaned_df["bin_width_ms"].tolist() == [500.0, 500.0]
    assert result.cleaned_df["firing_rate_hz"].tolist() == [4.0, 2.0]
