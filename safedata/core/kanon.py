# safedata/core/kanon.py

import pandas as pd
from typing import List
from pandas.api.types import is_numeric_dtype

from .risk import RiskAssessor


def generalise_age(df: pd.DataFrame, col: str = "age", bin_width: int = 10) -> pd.DataFrame:
    if col not in df.columns:
        return df

    if not is_numeric_dtype(df[col]):
        return df

    df = df.copy()
    max_age = int(df[col].max())
    bins = list(range(0, max_age + bin_width, bin_width))
    labels = [f"{b}-{b + bin_width - 1}" for b in bins[:-1]]

    df[col] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
    return df


def generalise_native_country(df: pd.DataFrame, col: str = "native-country") -> pd.DataFrame:
    if col not in df.columns:
        return df

    df = df.copy()
    df[col] = df[col].apply(
        lambda x: "United-States" if x == "United-States" else "Non-US"
    )
    return df


def _suppress_small_classes(df: pd.DataFrame, qids: List[str], k: int) -> pd.DataFrame:
    if not qids:
        return df

    df_work = df.dropna(subset=qids).copy()

    if df_work.empty:
        return df_work

    df_work["_class_size"] = (
        df_work.groupby(qids, observed=True)[qids[0]].transform("size")
    )

    df_suppressed = df_work[df_work["_class_size"] >= k].drop(columns="_class_size")

    return df_suppressed


def enforce_k_anonymity(
    df: pd.DataFrame,
    qids: List[str],
    k: int,
    max_iters: int = 1,
    suppress: bool = True,
) -> pd.DataFrame:
    """
    K-anonymity pipeline:
    1) Generalisation (age + native-country)
    2) Optional suppression of leftover small equivalence classes
    """
    df_k = df.copy()

    # --------- GENERALISATION PHASE ---------
    for _ in range(max_iters):
        assessor = RiskAssessor(df_k, qids)
        sizes = assessor.equivalence_class_sizes()

        if (sizes < k).sum() == 0:
            break

        df_k = generalise_age(df_k, "age", bin_width=10)
        df_k = generalise_native_country(df_k, "native-country")

    # --------- SUPPRESSION PHASE ---------
    if suppress:
        df_k = _suppress_small_classes(df_k, qids=qids, k=k)

    return df_k
