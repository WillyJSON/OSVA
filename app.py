"""
Streamlit entry point for organoid-spike-visual-analytics.

Current implemented page:
- Data Upload & Validation

Run:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from src.io import load_spike_bin_csv
from src.validation import issues_to_dataframe, validate_spike_bin_dataframe


st.set_page_config(
    page_title="OSVA",
    page_icon="🧠",
    layout="wide",
)

st.title("OSVA")
st.caption("Organoid Spike Visual Analytics — upload, validate, and prepare a spike-bin table")

st.header("Data Upload & Validation")
uploaded_file = st.file_uploader(
    "Upload spike-bin CSV",
    type=["csv"],
    help="Requires electrode_id, a time window (start/end), and at least one of spike_count or firing_rate_hz.",
)

if uploaded_file is None:
    st.info("Please upload a CSV, for example data/synthetic/mea_spike_bins_standard.csv.")
    st.stop()

try:
    raw_df = load_spike_bin_csv(uploaded_file)
except Exception as exc:  # noqa: BLE001 - Streamlit should show a friendly upload error.
    st.error(f"CSV read failed: {exc}")
    st.stop()

result = validate_spike_bin_dataframe(raw_df)

# -----------------------------
# Status banner
# -----------------------------
if result.status.startswith("Error"):
    st.error(result.status)
elif result.status.startswith("Warning"):
    st.warning(result.status)
else:
    st.success(result.status)

# -----------------------------
# Data Health Report
# -----------------------------
st.subheader("Data Health Report")
issues_df = issues_to_dataframe(result.issues)
# Dynamically choose a reasonable height so full messages are visible where possible.
n_issues = len(result.issues)
height = min(800, max(200, 40 * n_issues))
st.dataframe(issues_df, use_container_width=True, height=height)

if not result.usable:
    st.stop()

# -----------------------------
# Compact dataset summary
# -----------------------------
st.subheader("Dataset Summary")
summary = result.summary
cols = st.columns(5)
cols[0].metric("Rows", summary["n_rows"])
cols[1].metric("Electrodes", summary["n_electrodes"])
cols[2].metric("Conditions", summary["n_conditions"])
cols[3].metric("Spatial", "Ready" if result.spatial_available else "Skipped")
cols[4].metric("Condition Compare", "Ready" if result.condition_compare_available else "Skipped")

with st.expander("More validation summary", expanded=False):
    st.json(summary)

# -----------------------------
# Preview and download standardized data
# -----------------------------
tab_raw, tab_clean = st.tabs(["Raw Preview", "Standardized Preview"])
with tab_raw:
    st.dataframe(raw_df.head(100), use_container_width=True)
with tab_clean:
    st.dataframe(result.cleaned_df.head(100), use_container_width=True)

csv_bytes = result.cleaned_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download standardized CSV",
    data=csv_bytes,
    file_name="validated_spike_bins_standardized.csv",
    mime="text/csv",
)

st.divider()
st.caption(
    "Downstream modules should use standardized columns: "
    "bin_start_ms, bin_end_ms, bin_width_ms, spike_count, firing_rate_hz."
)
