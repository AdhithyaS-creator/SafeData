import pandas as pd


def qid_uniqueness_score(df: pd.DataFrame, column: str) -> float:
    """
    Compute uniqueness score for a single column.

    Score ~ fraction of values that appear only once.
    Higher score → higher re-identification risk.
    """

    if column not in df.columns:
        return 0.0

    col = df[column].dropna()

    if len(col) == 0:
            return 0.0

    freq = col.value_counts(dropna=True)
    unique_count = (freq == 1).sum()

    return unique_count / len(col)


def rank_qids_by_uniqueness(df: pd.DataFrame, qid_candidates) -> list:
    """
    Rank QID candidates from highest to lowest re-identification risk.

    Returns list of:
      { column, uniqueness_score, nunique, missing_ratio }
    """

    results = []

    for col in qid_candidates:

        if col not in df.columns:
            continue

        nunique = df[col].nunique(dropna=True)
        missing_ratio = df[col].isna().mean()

        score = qid_uniqueness_score(df, col)

        results.append({
            "column": col,
            "uniqueness_score": round(score, 4),
            "nunique": int(nunique),
            "missing_ratio": round(float(missing_ratio), 4),
        })

    # Highest risk first
    results = sorted(results, key=lambda x: x["uniqueness_score"], reverse=True)

    return results
