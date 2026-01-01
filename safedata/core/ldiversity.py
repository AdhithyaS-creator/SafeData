import pandas as pd


def enforce_l_diversity(
    df: pd.DataFrame,
    qids,
    sensitive_attr: str,
    L: int = 2,
    suppress: bool = True,
):
    """
    DISTINCT L-Diversity enforcement.

    A QID group is valid only if it contains at least L DISTINCT values
    for the chosen sensitive attribute.

    If suppress=True → violating groups are removed.
    """

    if sensitive_attr not in df.columns:
        raise ValueError(f"Sensitive attribute '{sensitive_attr}' not found in dataset")

    # Count distinct sensitive values per QID-group
    diversity = (
        df.groupby(qids, observed=False)[sensitive_attr]
        .nunique(dropna=True)
        .reset_index(name="distinct_sensitive_values")
    )

    total_groups = len(diversity)

    # Identify violating groups
    violating = diversity[diversity["distinct_sensitive_values"] < L]

    violating_groups = len(violating)

    violation_ratio = (
        violating_groups / total_groups if total_groups > 0 else 0.0
    )

    if not suppress:
        # return report only
        return df.copy(), {
            "total_groups": total_groups,
            "violating_groups": violating_groups,
            "violation_ratio": violation_ratio,
            "L_value": L,
            "suppression_applied": False,
            "records_before": len(df),
            "records_after": len(df),
            "records_removed": 0,
            "suppression_rate": 0.0,
        }

    # ---- Suppress violating records ----
    merge_cols = qids

    valid_groups = diversity[diversity["distinct_sensitive_values"] >= L][merge_cols]

    df_before = len(df)

    df_after = df.merge(valid_groups, on=qids, how="inner")

    removed = df_before - len(df_after)

    report = {
        "total_groups": total_groups,
        "violating_groups": violating_groups,
        "violation_ratio": violation_ratio,
        "L_value": L,
        "suppression_applied": True,
        "records_before": df_before,
        "records_after": len(df_after),
        "records_removed": removed,
        "suppression_rate": removed / df_before if df_before > 0 else 0.0,
    }

    return df_after, report
