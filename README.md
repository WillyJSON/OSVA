# OSVA

**Organoid Spike Visual Analytics** — a small, reproducible Python project for validating binned multi-electrode-array (MEA) spike tables and preparing analysis inputs for organoid experiments.

## Current scope

The Streamlit application currently implements **only** the upload and validation interface. It accepts a spike-bin CSV, reports schema/data-quality issues, standardizes approved columns, previews the result, and offers a CSV download. It is not yet a complete visual analytics dashboard.

The reusable modules in `src/` provide the analysis building blocks for later views:

- validation and canonicalization of spike-bin tables;
- electrode-time features and population activity summaries;
- correlation-based functional-connectivity edge tables; and
- deterministic, dependency-light network metrics.

## Included sample data

`data/synthetic/mea_spike_bins_standard.csv` is a **synthetic** binned-spike example (not a recording from a subject). Its documented schema is in `data/synthetic/README_mea_spike_bins.md`. The example contains 64 simulated electrodes, three conditions, and 0.5-second bins.

Real or restricted recordings belong in `data/raw/`, which is ignored by Git. Generated artifacts belong in `data/processed/`, `figure/`, or `report/`, which are also ignored.

## Quick start

Requires Python 3.10 or newer and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
uv run pytest
uv run streamlit run app.py
```

Then upload `data/synthetic/mea_spike_bins_standard.csv` in the browser. For a non-interactive Streamlit smoke check:

```bash
uv run streamlit run app.py --server.headless true --server.port 8501
```

## Input contract

A table must include:

- `electrode_id`;
- either `bin_start_ms` + `bin_end_ms` or `bin_start_s` + `bin_end_s`; and
- at least one of `spike_count` or `firing_rate_hz`.

When seconds are supplied, validation creates canonical millisecond columns. When exactly one spike metric is supplied, validation completes the other using the bin width. Optional `x_um`/`y_um` coordinates enable spatial analysis inputs, and a multi-valued `condition` enables condition-comparison inputs.

## Project layout

```text
app.py                   Streamlit upload/validation UI
src/                     Validation, feature, connectivity, and network modules
data/synthetic/          Versioned synthetic sample and schema
data/raw/                Ignored local real-data drop zone
data/processed/          Ignored derived-data output
figure/                  Ignored figure output
report/                  Ignored report output
tests/                   Validation and end-to-end module tests
```

## Reproducibility and limitations

The repository pins its resolved environment in `uv.lock` after `uv sync`; `pyproject.toml` defines direct dependencies and `requirements.txt` provides a concise pip-compatible list. Network metrics use deterministic NumPy/Pandas calculations and connected components as communities, rather than a community-detection package.

This repository is a software and synthetic-data example, not a clinical or experimental interpretation tool. Validate study-specific preprocessing, statistical assumptions, and quality-control requirements before using it for scientific conclusions.
