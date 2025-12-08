from pathlib import Path
import sys

import pandas as pd
import streamlit as st

# Ensure project root is on sys.path so "safedata" imports work
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from safedata.core.profiler import DataProfiler
from safedata.core.risk import RiskAssessor
from safedata.core.kanon import enforce_k_anonymity
from safedata.core.utility import (
    suppression_rate,
    categorical_tv_distance,
    numeric_mean_std_error,
)
from safedata.core.qid_selector import analyse_qid_candidates


def console_print(title: str, data):
    print("\n" + title)
    if isinstance(data, dict):
        for k, v in data.items():
            print(f"  {k}: {v}")
    else:
        print(" ", data)


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


@st.cache_data
def load_data() -> pd.DataFrame:
    data_path = ROOT / "data" / "adult.csv"
    df = pd.read_csv(data_path)
    return df


def show_profile(df: pd.DataFrame) -> dict:
    profiler = DataProfiler(df)
    summary = profiler.summary_dict()

    st.subheader("Dataset profile")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", summary["rows"])
    c2.metric("Columns", summary["cols"])
    c3.metric("File", "adult.csv")

    with st.expander("Missing values per column", expanded=False):
        st.dataframe(
            pd.DataFrame.from_dict(
                summary["missing"], orient="index", columns=["missing"]
            )
        )

    with st.expander("Unique values per column", expanded=False):
        st.dataframe(
            pd.DataFrame.from_dict(
                summary["unique"], orient="index", columns=["nunique"]
            )
        )

    st.markdown("Suggested QIDs (statistical candidates inferred from uniqueness):")
    st.code(summary["suggested_qids"])

    return summary


def show_qid_analysis(df: pd.DataFrame, suggested_qids):
    st.subheader("QID candidate analysis")

    st.markdown(
        "Columns are grouped by their re-identification potential and data quality. "
        "This is a data-driven view, not the final policy decision."
    )

    qid_analysis = analyse_qid_candidates(df, suggested_qids)

    strong = qid_analysis["strong_candidates"]
    weak = qid_analysis["weak_candidates"]
    avoid = qid_analysis["avoid_as_qid"]

    st.markdown("Strong candidates (high re-identification potential, low missingness):")
    if strong:
        df_strong = pd.DataFrame(strong)
        st.dataframe(df_strong[["column", "nunique", "missing_ratio"]])
    else:
        st.info("No strong candidates for this dataset.")

    st.markdown("Weak / optional candidates:")
    if weak:
        df_weak = pd.DataFrame(weak)
        st.dataframe(df_weak[["column", "nunique", "missing_ratio"]])
    else:
        st.info("No weak candidates for this dataset.")

    st.markdown("Columns to avoid as QIDs (too sparse or too identifying):")
    if avoid:
        df_avoid = pd.DataFrame(avoid)
        st.dataframe(df_avoid[["column", "nunique", "missing_ratio"]])
    else:
        st.info("No avoid-as-QID columns detected for this dataset.")

    return qid_analysis


def run_analysis(df: pd.DataFrame, qids, k_value: int):
    """
    Run full pipeline for given QIDs, log results to console,
    show them in the UI, and return the final anonymised dataframe (df_k_sup).
    """
    st.subheader("Re-identification risk and utility")

    if not qids:
        st.error("No QIDs selected.")
        return None

    cfg = {"QIDs": qids, "k": k_value, "records_raw": len(df)}
    console_print("=== ANALYSIS CONFIGURATION ===", cfg)

    st.markdown(f"Active QIDs: `{qids}`")
    st.markdown(f"k-anonymity parameter: `k = {k_value}`")

    # ========== 1) RAW RISK (BASELINE) ==========
    assessor_raw = RiskAssessor(df, qids=qids)
    risk_raw = assessor_raw.risk_report()
    console_print("--- RAW DATA RISK ---", risk_raw)

    st.markdown("#### 1. Raw data (no anonymisation)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Records", risk_raw["records"])
    c2.metric("Equiv. classes", risk_raw["num_equivalence_classes"])
    c3.metric("Uniqueness", pct(risk_raw["uniqueness_ratio"]))
    c4, c5, c6 = st.columns(3)
    c4.metric("Avg class size", f"{risk_raw['avg_equiv_class_size']:.2f}")
    c5.metric("Min class size", risk_raw["min_equiv_class_size"])
    c6.metric("Max class size", risk_raw["max_equiv_class_size"])

    st.caption(
        "This is the baseline risk before any anonymisation. "
        "Uniqueness above 0% indicates direct re-identification risk."
    )

    # ========== 2) GENERALISATION ONLY ==========
    df_k_gen = enforce_k_anonymity(df, qids=qids, k=k_value, suppress=False)
    assessor_gen = RiskAssessor(df_k_gen, qids=qids)
    risk_gen = assessor_gen.risk_report()
    console_print("--- AFTER K-ANONYMITY (GENERALISATION ONLY) ---", risk_gen)

    st.markdown("#### 2. After K-anonymity (generalisation only)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Records", risk_gen["records"])
    c2.metric("Equiv. classes", risk_gen["num_equivalence_classes"])
    c3.metric("Uniqueness", pct(risk_gen["uniqueness_ratio"]))
    c4, c5, c6 = st.columns(3)
    c4.metric("Avg class size", f"{risk_gen['avg_equiv_class_size']:.2f}")
    c5.metric("Min class size", risk_gen["min_equiv_class_size"])
    c6.metric("Max class size", risk_gen["max_equiv_class_size"])

    # Utility for generalisation vs RAW
    suppr_gen = suppression_rate(df, df_k_gen)  # should be 0.0
    console_print(
        "Utility after generalisation — suppression vs raw (%)",
        {"suppression": suppr_gen * 100},
    )

    tv_gen_rows = []
    tv_gen_dict = {}
    for col in ["education", "native-country", "income"]:
        tv = categorical_tv_distance(df, df_k_gen, col)
        tv_gen_rows.append({"column": col, "tv_distance (%)": tv * 100})
        tv_gen_dict[col] = tv * 100
    console_print("Utility after generalisation — TV distances (%)", tv_gen_dict)

    num_err_gen = numeric_mean_std_error(df, df_k_gen, "hours-per-week")
    num_err_gen_pct = {
        "mean_rel_error (%)": num_err_gen["mean_rel_error"] * 100,
        "std_rel_error (%)": num_err_gen["std_rel_error"] * 100,
    }
    console_print(
        "Utility after generalisation — numeric error (hours-per-week, %)",
        num_err_gen_pct,
    )

    st.markdown("Utility vs raw (after generalisation only):")
    c1, _ = st.columns(2)
    c1.metric("Suppression vs raw", pct(suppr_gen))

    with st.expander("Categorical TV distance (generalised vs raw)"):
        st.table(pd.DataFrame(tv_gen_rows))

    st.markdown("Numeric utility for `hours-per-week` (generalised vs raw):")
    c1, c2 = st.columns(2)
    c1.metric("Mean relative error", pct(num_err_gen["mean_rel_error"]))
    c2.metric("Std relative error", pct(num_err_gen["std_rel_error"]))

    # ========== 3) GENERALISATION + SUPPRESSION (FINAL) ==========
    df_k_sup = enforce_k_anonymity(df, qids=qids, k=k_value, suppress=True)
    assessor_sup = RiskAssessor(df_k_sup, qids=qids)
    risk_sup = assessor_sup.risk_report()
    console_print("--- AFTER K-ANONYMITY + SUPPRESSION (FINAL) ---", risk_sup)

    st.markdown("#### 3. After K-anonymity + suppression (final released dataset)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Records", risk_sup["records"])
    c2.metric("Equiv. classes", risk_sup["num_equivalence_classes"])
    c3.metric("Uniqueness", pct(risk_sup["uniqueness_ratio"]))
    c4, c5, c6 = st.columns(3)
    c4.metric("Avg class size", f"{risk_sup['avg_equiv_class_size']:.2f}")
    c5.metric("Min class size", risk_sup["min_equiv_class_size"])
    c6.metric("Max class size", risk_sup["max_equiv_class_size"])

    suppr_final_raw = suppression_rate(df, df_k_sup)
    suppr_final_gen = suppression_rate(df_k_gen, df_k_sup)
    console_print(
        "Final suppression (%)",
        {"vs_raw": suppr_final_raw * 100, "vs_generalised": suppr_final_gen * 100},
    )

    tv_sup_rows = []
    tv_sup_dict = {}
    for col in ["education", "native-country", "income"]:
        tv = categorical_tv_distance(df, df_k_sup, col)
        tv_sup_rows.append({"column": col, "tv_distance (%)": tv * 100})
        tv_sup_dict[col] = tv * 100
    console_print("Final TV distances (vs raw, %)", tv_sup_dict)

    num_err_sup = numeric_mean_std_error(df, df_k_sup, "hours-per-week")
    num_err_sup_pct = {
        "mean_rel_error (%)": num_err_sup["mean_rel_error"] * 100,
        "std_rel_error (%)": num_err_sup["std_rel_error"] * 100,
    }
    console_print("Final numeric utility (hours-per-week, %)", num_err_sup_pct)

    st.markdown("Utility vs raw / generalised (final dataset):")
    c1, c2 = st.columns(2)
    c1.metric("Suppression vs raw", pct(suppr_final_raw))
    c2.metric("Suppression vs generalised", pct(suppr_final_gen))

    with st.expander("Categorical TV distance (final vs raw)"):
        st.table(pd.DataFrame(tv_sup_rows))

    st.markdown("Numeric utility for `hours-per-week` (final vs raw):")
    c1, c2 = st.columns(2)
    c1.metric("Mean relative error", pct(num_err_sup["mean_rel_error"]))
    c2.metric("Std relative error", pct(num_err_sup["std_rel_error"]))

    st.markdown("Sample of final anonymised data:")
    st.dataframe(df_k_sup.head(50))

    # Return final anonymised dataframe so caller can offer download
    return df_k_sup


def main():
    st.set_page_config(
        page_title="SafeData – Privacy–Utility Dashboard",
        layout="wide",
    )

    st.title("SafeData – Privacy–Utility Preserving Framework")
    st.markdown(
        "This dashboard demonstrates how SafeData profiles a dataset, identifies quasi-identifiers, "
        "and evaluates the privacy–utility trade-off under different QID choices with k-anonymity."
    )

    df = load_data()
    summary = show_profile(df)

    tab1, tab2 = st.tabs(["Dataset & QID candidates", "Risk–Utility explorer"])

    with tab1:
        st.markdown(
            "Use this view to understand the structure of the dataset and the statistical QID candidates. "
            "These inform the policy-level selection of QIDs for anonymisation."
        )
        show_qid_analysis(df, summary["suggested_qids"])

    with tab2:
        qid_analysis = analyse_qid_candidates(df, summary["suggested_qids"])
        strong_names = [info["column"] for info in qid_analysis["strong_candidates"]]
        weak_names = [info["column"] for info in qid_analysis["weak_candidates"]]

        default_policy_qids = [
            q for q in ["age", "sex", "education", "native-country"] if q in df.columns
        ]

        st.sidebar.header("Configuration")

        mode = st.sidebar.radio(
            "QID selection mode",
            options=["Full suggested (Mode A)", "Policy QIDs (Mode B)", "Custom (Mode C)"],
            index=1,
        )

        # Fixed k
        k_value = 5
        st.sidebar.success("k-anonymity fixed at k = 5 (policy standard)")

        if mode == "Full suggested (Mode A)":
            qids = summary["suggested_qids"]
            st.info(
                "Mode A: Using all suggested QIDs as a worst-case stress test. "
                "This typically gives very strong privacy but may collapse utility."
            )
        elif mode == "Policy QIDs (Mode B)":
            qids = default_policy_qids
            st.info(
                f"Mode B: Using policy QIDs {default_policy_qids}. "
                "This reflects a realistic threat model and aims to balance privacy and utility."
            )
        else:
            st.sidebar.markdown("Strong QID candidates:")
            st.sidebar.write(strong_names)
            st.sidebar.markdown("Weak / optional candidates:")
            st.sidebar.write(weak_names)

            qids = st.sidebar.multiselect(
                "Select QIDs to use:",
                options=list(df.columns),
                default=default_policy_qids,
            )
            st.info(
                "Mode C: Custom QIDs. Use the candidate lists on the left as guidance, "
                "then explore how different QID sets affect risk and utility."
            )

        df_final = None
        if st.sidebar.button("Run analysis"):
            df_final = run_analysis(df, qids, k_value)

        # Download button appears below analysis, only if we have a final dataset
        if df_final is not None:
            st.markdown("### Download final anonymised dataset")
            csv_bytes = df_final.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download anonymised CSV",
                data=csv_bytes,
                file_name="adult_anonymised_k5.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
