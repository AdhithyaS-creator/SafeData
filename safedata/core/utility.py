# safedata/core/utility.py

from typing import Dict
import pandas as pd
import numpy as np


def suppression_rate(df_raw: pd.DataFrame, df_anon: pd.DataFrame) -> float:
    """
    Fraction of records removed by anonymisation/suppression.

    0.0  -> no rows removed
    0.5  -> half the rows suppressed
    """
    n_raw = len(df_raw)
    n_anon = len(df_anon)
    if n_raw == 0:
        return 0.0
    return 1.0 - (n_anon / n_raw)


def categorical_tv_distance(
    df_raw: pd.DataFrame,
    df_anon: pd.DataFrame,
    col: str,
) -> float:
    """
    Total variation distance between categorical distributions of a column.

    TV = 0      -> identical distributions
    TV -> 1     -> completely different

    We treat missing values as a separate category, if present.
    """
    if col not in df_raw.columns or col not in df_anon.columns:
        return 0.0

    raw_counts = df_raw[col].value_counts(dropna=False)
    anon_counts = df_anon[col].value_counts(dropna=False)

    n_raw = raw_counts.sum()
    n_anon = anon_counts.sum()
    if n_raw == 0 or n_anon == 0:
        return 0.0

    # Align indices
    all_values = raw_counts.index.union(anon_counts.index)

    p_raw = (raw_counts.reindex(all_values, fill_value=0) / n_raw).to_numpy()
    p_anon = (anon_counts.reindex(all_values, fill_value=0) / n_anon).to_numpy()

    tv = 0.5 * np.abs(p_raw - p_anon).sum()
    return float(tv)


def numeric_mean_std_error(
    df_raw: pd.DataFrame,
    df_anon: pd.DataFrame,
    col: str,
) -> Dict[str, float]:
    """
    Relative error in mean and standard deviation of a numeric column.

    Returns a dict with:
      - mean_rel_error
      - std_rel_error

    Values are in [0, +inf), but in practice should be small if utility is good.
    """
    if col not in df_raw.columns or col not in df_anon.columns:
        return {"mean_rel_error": 0.0, "std_rel_error": 0.0}

    raw = df_raw[col].dropna()
    anon = df_anon[col].dropna()

    if raw.empty or anon.empty:
        return {"mean_rel_error": 0.0, "std_rel_error": 0.0}

    mu_raw = float(raw.mean())
    mu_anon = float(anon.mean())
    std_raw = float(raw.std(ddof=1))
    std_anon = float(anon.std(ddof=1))

    eps = 1e-9

    mean_rel_error = abs(mu_raw - mu_anon) / max(abs(mu_raw), eps)
    std_rel_error = abs(std_raw - std_anon) / max(abs(std_raw), eps)

    return {
        "mean_rel_error": float(mean_rel_error),
        "std_rel_error": float(std_rel_error),
    }
