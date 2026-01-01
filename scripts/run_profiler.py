# scripts/run_profiler.py

import pandas as pd

from safedata.core.profiler import DataProfiler
from safedata.core.risk import RiskAssessor
from safedata.core.kanon import enforce_k_anonymity
from safedata.core.utility import (
    suppression_rate,
    categorical_tv_distance,
    numeric_mean_std_error,
)
from safedata.core.qid_selector import analyse_qid_candidates
from safedata.core.ldiversity import enforce_l_diversity


def print_dict_block(title, d):
    print(f"\n{title}")
    for k, v in d.items():
        print(f"  {k}: {v}")


def run_for_qids(df: pd.DataFrame, qids, mode_label: str) -> None:
    """
    Full pipeline for selected QIDs:

      1) Raw risk
      2) K-anonymity (generalisation only)
      3) K-anonymity + suppression
      4) L-diversity on k-anonymous data
      5) Utility reports at each stage
    """

    print(f"\n=== {mode_label} ===")
    print("Using QIDs:")
    print("  ", qids)

    if not qids:
        print("[ERROR] No QIDs provided.")
        return

    # ------------------------------------------------
    # Ask user for K
    # ------------------------------------------------
    k_value = int(input("\nEnter k value for K-anonymity (recommended = 5): ") or 5)
    print(f"\n[ Applying K-ANONYMITY | k = {k_value} ]")

    # ------------------------------------------------
    # 1) RAW RISK
    # ------------------------------------------------
    assessor_raw = RiskAssessor(df, qids=qids)
    risk_raw = assessor_raw.risk_report()

    print_dict_block("--- RE-IDENTIFICATION RISK (RAW DATA) ---", risk_raw)

    # ------------------------------------------------
    # 2) GENERALISATION ONLY (NO SUPPRESSION)
    # ------------------------------------------------
    df_k_gen = enforce_k_anonymity(df, qids=qids, k=k_value, suppress=False)

    assessor_gen = RiskAssessor(df_k_gen, qids=qids)
    risk_gen = assessor_gen.risk_report()

    print_dict_block(
        f"--- RISK AFTER K-ANONYMITY (GENERALISATION ONLY, k={k_value}) ---",
        risk_gen,
    )

    # Utility vs raw
    suppr_gen = suppression_rate(df, df_k_gen)

    print(f"\n[ Utility vs Raw after Generalisation ]")
    print(f"  Suppression rate: {suppr_gen:.4f} (expected ~0)")

    for col in ["education", "native-country", "income"]:
        tv = categorical_tv_distance(df, df_k_gen, col)
        print(f"  TV distance for {col}: {tv:.4f}")

    num_err = numeric_mean_std_error(df, df_k_gen, "hours-per-week")
    print(
        f"  hours-per-week mean error={num_err['mean_rel_error']:.4f}, "
        f"std error={num_err['std_rel_error']:.4f}"
    )

    # ------------------------------------------------
    # 3) K-ANONYMITY + SUPPRESSION (FINAL K RELEASE)
    # ------------------------------------------------
    df_k_sup = enforce_k_anonymity(df, qids=qids, k=k_value, suppress=True)

    assessor_sup = RiskAssessor(df_k_sup, qids=qids)
    risk_sup = assessor_sup.risk_report()

    print_dict_block(
        f"--- RISK AFTER K-ANONYMITY + SUPPRESSION (k={k_value}) ---",
        risk_sup,
    )

    suppr_final_raw = suppression_rate(df, df_k_sup)
    suppr_final_gen = suppression_rate(df_k_gen, df_k_sup)

    print("\n[ Suppression impact ]")
    print(f"  vs raw: {suppr_final_raw:.4f}")
    print(f"  vs generalised: {suppr_final_gen:.4f}")

    for col in ["education", "native-country", "income"]:
        tv = categorical_tv_distance(df, df_k_sup, col)
        print(f"  TV distance for {col}: {tv:.4f}")

    num_err2 = numeric_mean_std_error(df, df_k_sup, "hours-per-week")
    print(
        f"  hours-per-week mean error={num_err2['mean_rel_error']:.4f}, "
        f"std error={num_err2['std_rel_error']:.4f}"
    )

    # =====================================================
    # 4) L-DIVERSITY (APPLIED AFTER K-ANONYMITY)
    # =====================================================
    print("\n===================================================")
    print("          APPLYING DISTINCT L-DIVERSITY")
    print("===================================================")

    sensitive_attr = input(
        "Enter sensitive attribute (default = income): "
    ).strip() or "income"

    L_value = int(input("Enter L value (recommended = 2): ") or 2)

    print(
        f"\n[ DISTINCT L-DIVERSITY | L = {L_value} | sensitive = {sensitive_attr} ]"
    )

    df_l, l_report = enforce_l_diversity(
        df_k_sup,
        qids=qids,
        sensitive_attr=sensitive_attr,
        L=L_value,
        suppress=True,
    )

    print_dict_block("\n--- L-DIVERSITY REPORT ---", l_report)

    print("\nRecords after K-Anon + L-Diversity:")
    print(f"  Before L-diversity: {len(df_k_sup)}")
    print(f"  After L-diversity:  {len(df_l)}")
    print(f"  Removed:            {l_report.get('records_removed', 0)}")

    print("\n=== END OF MODE RUN ===\n")


def main() -> None:
    # ------------------------------------------------
    # Load dataset
    # ------------------------------------------------
    df = pd.read_csv("data/adult.csv")

    profiler = DataProfiler(df)
    summary = profiler.summary_dict()

    print("=== DATA PROFILE ===")
    print(f"Rows: {summary['rows']}, Columns: {summary['cols']}")

    print("\nMissing values per column:")
    for col, cnt in summary["missing"].items():
        print(f"  {col}: {cnt}")

    print("\nUnique values per column:")
    for col, cnt in summary["unique"].items():
        print(f"  {col}: {cnt}")

    print("\nSuggested QIDs (from profiler):")
    print("  ", summary["suggested_qids"])

    # ------------------------------------------------
    # QID Candidate Analysis
    # ------------------------------------------------
    qid_analysis = analyse_qid_candidates(df, summary["suggested_qids"])

    print("\nQID candidate analysis:")

    print("  Strong candidates:")
    for info in qid_analysis["strong_candidates"]:
        print(
            f"    {info['column']} "
            f"(nunique={info['nunique']}, missing={info['missing_ratio']:.3f})"
        )

    print("  Weak candidates:")
    for info in qid_analysis["weak_candidates"]:
        print(
            f"    {info['column']} "
            f"(nunique={info['nunique']}, missing={info['missing_ratio']:.3f})"
        )

    print("  Avoid as QIDs:")
    for info in qid_analysis["avoid_as_qid"]:
        print(
            f"    {info['column']} "
            f"(nunique={info['nunique']}, missing={info['missing_ratio']:.3f})"
        )

    default_policy_qids = [
        q for q in ["age", "sex", "education", "native-country"] if q in df.columns
    ]

    # ------------------------------------------------
    # MODE SELECTION
    # ------------------------------------------------
    print("\nSelect mode:")
    print("  1) Mode A — Full QID set (stress test)")
    print(f"  2) Mode B — Policy QIDs {default_policy_qids}")
    print("  3) Mode C — Custom QIDs")

    choice = input("Enter 1, 2 or 3: ").strip()

    if choice == "1":
        run_for_qids(df, summary["suggested_qids"], "MODE A: FULL-QID VIEW")

    elif choice == "2":
        run_for_qids(df, default_policy_qids, "MODE B: POLICY-QID VIEW")

    elif choice == "3":
        all_cols = list(df.columns)
        print("\nAvailable columns:")
        print(all_cols)

        user_input = input("Enter QIDs (comma separated): ").strip()
        custom_qids = [c.strip() for c in user_input.split(",") if c.strip()]

        run_for_qids(df, custom_qids, "MODE C: CUSTOM-QID VIEW")

    else:
        print("Invalid option.")


if __name__ == "__main__":
    main()
