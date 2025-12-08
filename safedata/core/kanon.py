# safedata/core/kanon.py

import pandas as pd
from typing import List
from pandas.api.types import is_numeric_dtype

from .risk import RiskAssessor


def generalise_age(df: pd.DataFrame, col: str = "age", bin_width: int = 10) -> pd.DataFrame:
    """
    Generalise numeric age into bins of given width (e.g. 10-year ranges).
    If age is already categorical (e.g. '30-39'), it is left unchanged.
    """
    if col not in df.columns:
        return df

    # If age is already generalised (string categories), don't touch it
    if not is_numeric_dtype(df[col]):
        return df

    df = df.copy()
    max_age = int(df[col].max())
    bins = list(range(0, max_age + bin_width, bin_width))
    labels = [f"{b}-{b + bin_width - 1}" for b in bins[:-1]]

    df[col] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
    return df


def generalise_native_country(df: pd.DataFrame, col: str = "native-country") -> pd.DataFrame:
    """
    Generalise native-country into coarse regions:
    - 'United-States'
    - 'Non-US'
    """
    if col not in df.columns:
        return df

    df = df.copy()
    df[col] = df[col].apply(
        lambda x: "United-States" if x == "United-States" else "Non-US"
    )
    return df


def enforce_k_anonymity(
    df: pd.DataFrame,
    qids: List[str],
    k: int,
    max_iters: int = 1,
) -> pd.DataFrame:
    """
    Simple heuristic K-anonymity:
    - generalise age into bins
    - generalise native-country into US / Non-US
    - re-evaluate equivalence class sizes
    - stop when:
        * all equivalence classes >= k, or
        * max_iters reached
    """
    df_k = df.copy()

    for _ in range(max_iters):
        assessor = RiskAssessor(df_k, qids)
        sizes = assessor.equivalence_class_sizes()

        # If all groups have size >= k, we're done
        if (sizes < k).sum() == 0:
            break

        # Apply generalisation
        df_k = generalise_age(df_k, "age", bin_width=10)
        df_k = generalise_native_country(df_k, "native-country")

    return df_k
