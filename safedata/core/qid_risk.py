import pandas as pd
import numpy as np


SENSITIVE_KEYWORDS = [
    "income", "salary", "disease", "health",
    "bank", "account", "religion"
]


def is_sensitive_column(col: str) -> bool:
    col_l = col.lower()
    return any(k in col_l for k in SENSITIVE_KEYWORDS)


def column_entropy(series: pd.Series) -> float:
    """
    Shannon entropy — measures value spread / unpredictability.
    """
    counts = series.value_counts(normalize=True, dropna=True)
    return -(counts * np.log2(counts)).sum()


def qid_risk_score(df: pd.DataFrame, col: str) -> dict:
    """
    Compute privacy-risk score for a single attribute.
    Higher = stronger QID candidate.
    """

    s = df[col]

    nunique = s.nunique(dropna=True)
    missing_ratio = s.isna().mean()

    # Distinct density wrt dataset size
    distinct_density = nunique / len(df)

    # Entropy (scaled 0-1)
    ent = column_entropy(s)
    ent_norm = ent / np.log2(max(nunique, 2))

    # Cardinality factor (categorical vs numeric)
    cardinality_factor = min(nunique / 50, 1.0)

    # Sensitivity penalty
    sensitive = is_sensitive_column(col)
    sensitive_penalty = 0.6 if sensitive else 0.0

    # Final score (tunable weights)
    score = (
        0.40 * distinct_density +
        0.30 * ent_norm +
        0.20 * cardinality_factor -
        0.20 * missing_ratio -
        sensitive_penalty
    )

    return {
        "column": col,
        "nunique": nunique,
        "missing_ratio": round(missing_ratio, 4),
        "distinct_density": round(distinct_density, 4),
        "entropy": round(ent, 4),
        "entropy_norm": round(ent_norm, 4),
        "score": round(score, 4),
        "is_sensitive": sensitive,
    }


def rank_qid_risks(df: pd.DataFrame, candidates: list[str]) -> dict:
    """
    Returns ranked QID risk groups based on score thresholds.
    """

    results = [qid_risk_score(df, c) for c in candidates]

    strong = []
    moderate = []
    weak = []
    avoid = []

    for r in results:
        if r["is_sensitive"]:
            avoid.append(r)
        elif r["score"] >= 0.55:
            strong.append(r)
        elif r["score"] >= 0.35:
            moderate.append(r)
        else:
            weak.append(r)

    # Sort high → low score in each bucket
    for group in (strong, moderate, weak, avoid):
        group.sort(key=lambda x: x["score"], reverse=True)

    return {
        "strong": strong,
        "moderate": moderate,
        "weak": weak,
        "avoid": avoid,
        "all_scored": results,
    }
