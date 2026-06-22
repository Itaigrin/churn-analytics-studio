"""
File loading with automatic delimiter detection and encoding fallback.
"""

import io
import pandas as pd


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    """
    Load a CSV or Excel file from a Streamlit UploadedFile object.
    Automatically detects CSV delimiter and tries multiple encodings.
    """
    name = uploaded_file.name.lower()

    if name.endswith((".xlsx", ".xls")):
        uploaded_file.seek(0)
        xl = pd.ExcelFile(uploaded_file)
        if len(xl.sheet_names) == 1:
            return xl.parse(xl.sheet_names[0])
        # Multiple sheets — pick the one with the most rows
        best_sheet, best_rows = xl.sheet_names[0], -1
        for sheet in xl.sheet_names:
            try:
                n = xl.parse(sheet).shape[0]
                if n > best_rows:
                    best_rows, best_sheet = n, sheet
            except Exception:
                continue
        return xl.parse(best_sheet)

    # Read a sample to detect delimiter
    uploaded_file.seek(0)
    raw_bytes = uploaded_file.read(8192)
    uploaded_file.seek(0)

    # Try encodings in order
    sample_text = None
    detected_enc = "utf-8"
    for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            sample_text = raw_bytes.decode(enc)
            detected_enc = enc
            break
        except UnicodeDecodeError:
            continue
    if sample_text is None:
        sample_text = raw_bytes.decode("utf-8", errors="replace")

    delimiter = _detect_delimiter(sample_text)

    for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=enc, sep=delimiter)
            df.columns = _clean_column_names(df.columns)
            return df
        except (UnicodeDecodeError, Exception):
            continue

    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file, encoding="utf-8", errors="replace", sep=delimiter)
    df.columns = _clean_column_names(df.columns)
    return df


def _detect_delimiter(sample: str) -> str:
    first_line = sample.split("\n")[0] if "\n" in sample else sample
    counts = {d: first_line.count(d) for d in [",", ";", "\t", "|"]}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def _clean_column_names(columns) -> list:
    return [str(c).strip().lstrip("﻿").lstrip("ï»¿") for c in columns]
