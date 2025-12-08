# scripts/run_profiler.py

import pandas as pd

from safedata.core.profiler import DataProfiler
from safedata.core.risk import RiskAssessor
from safedata.core.kanon import enforce_k_anonymity


def main() -> None:
    # Load dataset
    df = pd.read_csv("C:\\Users\\USER\\Desktop\\SafeData\\data\\adult.csv")

    # ---------- DATA PROFILING ----------
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

    print("\nSuggested QIDs:")
    print("  ", summary["suggested_qids"])

    # ---------- QID SELECTION ----------
    qids = ["age", "sex", "education", "native-country"]
    qids = [q for q in qids if q in df.columns]

    print("\nUsing QIDs for risk assessment:", qids)

    # ---------- RAW RE-IDENTIFICATION RISK ----------
    assessor = RiskAssessor(df, qids=qids)
    risk_raw = assessor.risk_report()

    print("\n=== RE-IDENTIFICATION RISK (RAW DATA) ===")
    for k, v in risk_raw.items():
        print(f"  {k}: {v}")

    # ---------- APPLY K-ANONYMITY ----------
    k_value = 5
    df_k = enforce_k_anonymity(df, qids=qids, k=k_value)

    assessor_k = RiskAssessor(df_k, qids=qids)
    risk_k = assessor_k.risk_report()

    print(f"\n=== RE-IDENTIFICATION RISK AFTER K-ANONYMITY (k={k_value}) ===")
    for k, v in risk_k.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
