import pandas as pd


def enforce_l_diversity(
    df: pd.DataFrame,
    qids,
    sensitive_attr: str,
    L: int = 2,
    suppress: bool = True,
):
    """
    Enforces DISTINCT L-DIVERSITY on already K-anonymised data.

    A group is valid only if:
        number of DISTINCT values in sensitive attribute >= L

    If suppress=True  -> remove violating groups
    If suppress=False -> keep them (just report violations)
    """

    if sensitive_attr not in df.columns:
        raise ValueError(f"Sensitive attribute '{sensitive_attr}' not found in dataset")

    # Compute distinct value counts per equivalence class
    diversity = (
        df.groupby(qids)[sensitive_attr]
        .nunique(dropna=True)
        .reset_index(name="distinct_count")
    )

    # Mark violating groups
    violating_groups = diversity[diversity["distinct_count"] < L]

    num_groups = len(diversity)
    num_violating = len(violating_groups)

    violation_ratio = num_violating / max(num_groups, 1)

    report = {
        "total_groups": num_groups,
        "violating_groups": num_violating,
        "violation_ratio": violation_ratio,
        "L_value": L,
        "suppression_applied": suppress,
    }

    # If we are only analysing — stop here
    if not suppress:
            return df.copy(), report

    # -----------------------------
    # SUPPRESS violating groups
    # -----------------------------

    if num_violating == 0:
        return df.copy(), report

    # Join flags back to dataframe
    df_flagged = df.merge(diversity, on=qids, how="left")

    df_valid = df_flagged[df_flagged["distinct_count"] >= L].drop(columns=["distinct_count"])

    report["records_before"] = len(df)
    report["records_after"] = len(df_valid)
    report["records_removed"] = len(df) - len(df_valid)
    report["suppression_rate"] = report["records_removed"] / len(df)

    return df_valid, report
