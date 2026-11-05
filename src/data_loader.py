"""
File loading with automatic delimiter detection and encoding detection.

Everything is decoded up front from the whole file, so a problem in the last
row cannot be missed by looking only at a sample of the first few kilobytes.
"""

import io

import pandas as pd

from .errors import DataFileError

_HEBREW_NEIGHBOUR_SHARE = 0.5


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    """
    Load a CSV or Excel file from a Streamlit UploadedFile object.
    Raises DataFileError with a readable message when the file cannot be parsed.
    """
    name = uploaded_file.name.lower()
    uploaded_file.seek(0)

    if name.endswith((".xlsx", ".xls")):
        return _load_excel(uploaded_file)

    raw = uploaded_file.read()
    if not raw.strip():
        raise DataFileError("The file is empty.")

    text = _decode(raw)
    delimiter = _detect_delimiter(text[:8192])
    try:
        df = pd.read_csv(io.StringIO(text), sep=delimiter)
    except pd.errors.EmptyDataError as exc:
        raise DataFileError("The file has no columns to read.") from exc
    except pd.errors.ParserError as exc:
        raise DataFileError(
            f"The file is not a well-formed table, so it was not loaded: {exc}. "
            "Check that row in the file, or export it again from the source system."
        ) from exc
    df.columns = _clean_column_names(df.columns)
    return df


def _load_excel(uploaded_file) -> pd.DataFrame:
    try:
        xl = pd.ExcelFile(uploaded_file)
        sheets = {sheet: xl.parse(sheet) for sheet in xl.sheet_names}
    except (ValueError, OSError, ImportError) as exc:
        raise DataFileError(f"The Excel file could not be read: {exc}") from exc
    # Several sheets: the one with the most rows is taken to be the data
    df = max(sheets.values(), key=len)
    df.columns = _clean_column_names(df.columns)
    return df


def _decode(raw: bytes) -> str:
    """UTF-8 and UTF-16 are recognised exactly; legacy single-byte files by their bytes."""
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    encoding = "cp1255" if _looks_like_hebrew(raw) else "cp1252"
    return raw.decode(encoding, errors="replace")


def _looks_like_hebrew(raw: bytes) -> bool:
    """
    Tell Windows-1255 Hebrew from Windows-1252 Western text.

    Both use bytes above 0x7F, so the difference is in how they sit together.
    A Hebrew word is a run of such bytes, while an accented Western letter
    (é, ü) stands alone between ASCII letters.
    """
    high = [i for i, b in enumerate(raw) if b >= 0x80]
    if not high:
        return False
    high_set = set(high)
    with_neighbour = sum(1 for i in high if i - 1 in high_set or i + 1 in high_set)
    return with_neighbour / len(high) >= _HEBREW_NEIGHBOUR_SHARE


def _detect_delimiter(sample: str) -> str:
    first_line = sample.split("\n")[0] if "\n" in sample else sample
    counts = {d: first_line.count(d) for d in [",", ";", "\t", "|"]}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def _clean_column_names(columns) -> list:
    return [str(c).strip().lstrip("﻿") for c in columns]
