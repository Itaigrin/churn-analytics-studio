"""
Download helpers.
"""

from __future__ import annotations

import pandas as pd

from .pipeline_builder import is_text_column

_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    CSV for opening in Excel. The byte-order mark makes Excel read Hebrew correctly.
    Text starting like a formula is neutralised, because a spreadsheet would run it.
    """
    safe = df.copy()
    for col in safe.columns:
        if is_text_column(safe[col]):
            safe[col] = safe[col].map(_neutralise_formula)
    return safe.to_csv(index=False).encode("utf-8-sig")


def _neutralise_formula(value):
    if isinstance(value, str) and value.startswith(_FORMULA_PREFIXES):
        return "'" + value
    return value
