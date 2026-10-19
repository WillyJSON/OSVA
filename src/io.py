"""
Input / output helpers for organoid-spike-visual-analytics.

This module deliberately does not guess column names or reshape user files.
It only reads a CSV and lets src.validation decide whether the data is usable.
"""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Union

import pandas as pd

PathLikeOrBuffer = Union[str, Path, BinaryIO]


def load_spike_bin_csv(file: PathLikeOrBuffer) -> pd.DataFrame:
    """
    Load an uploaded spike-bin CSV.

    Parameters
    ----------
    file:
        Either a local path or a Streamlit UploadedFile-like buffer.

    Notes
    -----
    - We do not rename columns here.
    - We do not infer missing fields here.
    - Validation and allowed standardization are handled in validation.py.
    """
    return pd.read_csv(file)


def save_processed_csv(df: pd.DataFrame, output_path: PathLikeOrBuffer) -> None:
    """Save a validated / standardized dataframe for reproducible downstream analysis."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
