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


def run_for_qids(df: pd.DataFrame, qids, mode_label: str, k_value: int = 5) -> None:
    """
    Run full pipeline for a given set of QIDs:
    - raw risk
    - k-anonymity via generalisation (no suppression)
    - k-anonymity + suppression
    - risk after each stage
    - utility analysis on final (suppressed) dataset
    """
    print(f"\n=== {mode_label} ===")
    print("Using QIDs:")
    print("  ", qids)

    if not qids:
        print("[ERROR] No QIDs provided for this mode.")
        return

    # 1) Raw risk
    assessor_raw = RiskAssessor(df, qids=qids)
    risk_raw = assessor_raw.risk_report()

    print("\n--- RE-IDENTIFICATION RISK (RAW DATA) ---")
    for k, v in risk_raw.items():
        print(f"  {k}: {v}")

    # 2) K-anonymity via generalisation ONLY (no suppression)
    df_k_gen = enforce_k_anonymity(df, qids=qids, k=k_value, suppress=False)

    assessor_gen = RiskAssessor(df_k_gen, qids=qids)
    risk_gen = assessor_gen.risk_report()

    print(f"\n--- RISK AFTER K-ANONYMITY (GENERALISATION ONLY, k={k_value}) ---")
    for k, v in risk_gen.items():
        print(f"  {k}: {v}")

    # 3) K-anonymity + suppression (final released dataset)
    df_k_sup = enforce_k_anonymity(df, qids=qids, k=k_value, suppress=True)

    assessor_sup = RiskAssessor(df_k_sup, qids=qids)
    risk_sup = assessor_sup.risk_report()

    print(f"\n--- RISK AFTER K-ANONYMITY + SUPPRESSION (k={k_value}) ---")
    for k, v in risk_sup.items():
        print(f"  {k}: {v}")

    # 4) Utility analysis on final (suppressed) dataset
    print("\n--- UTILITY ANALYSIS (FINAL RELEASED DATASET) ---")

    # Overall suppression relative to original
    suppr_overall = suppression_rate(df, df_k_sup)
    print(f"Suppression rate (vs raw): {suppr_overall:.4f}  (fraction of records removed)")

    # Suppression relative to generalised-only version
    suppr_from_gen = suppression_rate(df_k_gen, df_k_sup)
    print(f"Suppression rate (vs generalised-only): {suppr_from_gen:.4f}")

    # Categorical utility on key columns (if they exist)
    for col in ["education", "native-country", "income"]:
        tv = categorical_tv_distance(df, df_k_sup, col)
        print(f"TV distance for {col}: {tv:.4f}")

    # Numeric utility on hours-per-week (if it exists)
    num_err = numeric_mean_std_error(df, df_k_sup, "hours-per-week")
    print(
        "hours-per-week mean rel. error: "
        f"{num_err['mean_rel_error']:.4f}, "
        "std rel. error: "
        f"{num_err['std_rel_error']:.4f}"
    )

    print("\n=== END OF MODE RUN ===\n")


def main() -> None:
    # Load dataset
    df = pd.read_csv("data/adult.csv")

    # ---------- 1. DATA PROFILING (COMMON) ----------
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

    # ---------- 2. QID CANDIDATE ANALYSIS (DATA-DRIVEN) ----------
    qid_analysis = analyse_qid_candidates(df, summary["suggested_qids"])

    print("\nQID candidate analysis:")

    print("  Strong candidates (high re-identification potential, low missingness):")
    for info in qid_analysis["strong_candidates"]:
        print(
            f"    {info['column']}: "
            f"nunique={info['nunique']}, "
            f"missing_ratio={info['missing_ratio']:.3f}"
        )

    print("  Weak / optional candidates:")
    for info in qid_analysis["weak_candidates"]:
        print(
            f"    {info['column']}: "
            f"nunique={info['nunique']}, "
            f"missing_ratio={info['missing_ratio']:.3f}"
        )

    print("  Columns to avoid as QIDs (too sparse or too identifying):")
    for info in qid_analysis["avoid_as_qid"]:
        print(
            f"    {info['column']}: "
            f"nunique={info['nunique']}, "
            f"missing_ratio={info['missing_ratio']:.3f}"
        )

    # Recommended default policy QIDs (still hard-coded, but now visible)
    default_policy_qids = [q for q in ["age", "sex", "education", "native-country"] if q in df.columns]

    # ---------- 3. MODE / QID SELECTION ----------
    print("\nSelect mode:")
    print("  1) Mode A – Full-QID view (all suggested QIDs)")
    print(f"  2) Mode B – Policy-QID view (default: {default_policy_qids})")
    print("  3) Mode C – Custom-QID view (you choose the QIDs)")

    choice = input("Enter 1, 2 or 3: ").strip()

    if choice == "1":
        # Mode A: use all suggested QIDs (stress-test)
        analysis_qids = summary["suggested_qids"]
        if not analysis_qids:
            print("[ERROR] No suggested QIDs found for Mode A.")
            return

        run_for_qids(df, analysis_qids, mode_label="MODE A: FULL-QID VIEW")

    elif choice == "2":
        # Mode B: use policy-defined QID subset (practical)
        if not default_policy_qids:
            print("[ERROR] None of the default policy QIDs are present in the dataset.")
            return

        run_for_qids(df, default_policy_qids, mode_label="MODE B: POLICY-QID VIEW")

    elif choice == "3":
        # Mode C: custom QID set (user-chosen)
        print("\n=== CUSTOM-QID SELECTION ===")
        print("You can choose from these candidates:")

        strong_names = [info["column"] for info in qid_analysis["strong_candidates"]]
        weak_names = [info["column"] for info in qid_analysis["weak_candidates"]]

        print("  Strong candidates:", strong_names)
        print("  Weak / optional candidates:", weak_names)
        print("  (You can also use any other column name from the dataset if needed.)")

        user_input = input(
            "Enter QID column names separated by commas "
            "(e.g., age, sex, education, native-country): "
        ).strip()

        if not user_input:
            print("[ERROR] No QIDs entered for custom mode.")
            return

        custom_qids = [c.strip() for c in user_input.split(",") if c.strip()]
        custom_qids = [q for q in custom_qids if q in df.columns]

        if not custom_qids:
            print("[ERROR] None of the entered QIDs exist in the dataset.")
            return

        run_for_qids(df, custom_qids, mode_label="MODE C: CUSTOM-QID VIEW")

    else:
        print("Invalid choice. Please run again and enter 1, 2 or 3.")


if __name__ == "__main__":
    main()
