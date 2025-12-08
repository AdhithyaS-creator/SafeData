# safedata/core/qid_selector.py

from typing import Dict, List
import pandas as pd


def analyse_qid_candidates(
    df: pd.DataFrame,
    suggested_qids: List[str],
) -> Dict[str, List[Dict]]:
    """
    Analyse suggested QID columns and group them into:
      - strong_candidates
      - weak_candidates
      - avoid_as_qid

    Heuristics (dataset-driven):
      - nunique: how many distinct values
      - missing_ratio: fraction of missing values

    Strong QID candidate:
      3 <= nunique <= 200 and missing_ratio <= 0.3

    Weak QID candidate:
      nunique <= 2
      OR (nunique > 200 and nunique <= 0.5 * n)
      OR (0.3 < missing_ratio <= 0.6)

    Avoid as QID:
      nunique > 0.5 * n   (too close to identifier)
      OR missing_ratio > 0.6
    """
    n = len(df)
    if n == 0 or not suggested_qids:
        return {
            "strong_candidates": [],
            "weak_candidates": [],
            "avoid_as_qid": [],
        }

    strong: List[Dict] = []
    weak: List[Dict] = []
    avoid: List[Dict] = []

    for col in suggested_qids:
        if col not in df.columns:
            continue

        col_series = df[col]
        nunique = col_series.nunique(dropna=True)
        missing = col_series.isna().sum()

        missing_ratio = missing / n if n > 0 else 0.0

        info = {
            "column": col,
            "nunique": int(nunique),
            "missing_ratio": float(missing_ratio),
        }

        # Decide category
        if nunique > 0.5 * n or missing_ratio > 0.6:
            avoid.append(info)
        elif 3 <= nunique <= 200 and missing_ratio <= 0.3:
            strong.append(info)
        else:
            weak.append(info)

    return {
        "strong_candidates": strong,
        "weak_candidates": weak,
        "avoid_as_qid": avoid,
    }
