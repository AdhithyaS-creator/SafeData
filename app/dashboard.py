from pathlib import Path
import sys
import pandas as pd
import streamlit as st

# Ensure project root is on sys.path
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
from safedata.core.ldiversity import enforce_l_diversity


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


@st.cache_data
def load_data() -> pd.DataFrame:
    data_path = ROOT / "data" / "adult.csv"
    return pd.read_csv(data_path)


# ------------------------------------------
# SECTION: DATASET PROFILE
# ------------------------------------------
def show_profile(df: pd.DataFrame) -> dict:
    profiler = DataProfiler(df)
    summary = profiler.summary_dict()

    st.subheader("Dataset Profile")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", summary["rows"])
    c2.metric("Columns", summary["cols"])
    c3.metric("File", "adult.csv")

    with st.expander("Missing values per column"):
        st.dataframe(
            pd.DataFrame.from_dict(
                summary["missing"], orient="index", columns=["missing"]
            )
        )

    with st.expander("Unique values per column"):
        st.dataframe(
            pd.DataFrame.from_dict(
                summary["unique"], orient="index", columns=["nunique"]
            )
        )

    st.markdown("Suggested QIDs (statistical candidates inferred from uniqueness):")
    st.code(summary["suggested_qids"])

    return summary


# ------------------------------------------
# SECTION: QID CANDIDATE ANALYSIS
# ------------------------------------------
def show_qid_analysis(df: pd.DataFrame, suggested_qids):
    st.subheader("QID Candidate Analysis")

    st.caption(
        "Columns are grouped based on re-identification potential and data quality. "
        "This is a data-driven recommendation — final QIDs are policy-driven."
    )

    qid_analysis = analyse_qid_candidates(df, suggested_qids)

    strong = qid_analysis["strong_candidates"]
    weak = qid_analysis["weak_candidates"]
    avoid = qid_analysis["avoid_as_qid"]

    st.markdown("Strong candidates (high uniqueness & low missingness):")
    if strong:
        st.dataframe(pd.DataFrame(strong))
    else:
        st.info("No strong candidates detected.")

    st.markdown("Weak / optional candidates:")
    if weak:
        st.dataframe(pd.DataFrame(weak))
    else:
        st.info("No weak candidates detected.")

    st.markdown("Columns to avoid as QIDs:")
    if avoid:
        st.dataframe(pd.DataFrame(avoid))
    else:
        st.info("No avoid-as-QID columns detected.")

    return qid_analysis


# ------------------------------------------
# CORE PIPELINE RUNNER
# ------------------------------------------
def run_pipeline(df, qids, k_value: int, enable_ldiv: bool):
    st.subheader("Risk–Utility Analysis")

    if not qids:
        st.error("No QIDs selected.")
        return None

    st.markdown(f"**Active QIDs:** `{qids}`")
    st.markdown(f"**k-anonymity parameter:** `{k_value}`")

    # ---------- 1) RAW RISK ----------
    assessor_raw = RiskAssessor(df, qids)
    risk_raw = assessor_raw.risk_report()

    with st.container(border=True):
        st.markdown("### Raw Data — Baseline Re-identification Risk")

        c1, c2, c3 = st.columns(3)
        c1.metric("Records", risk_raw["records"])
        c2.metric("Equivalence Classes", risk_raw["num_equivalence_classes"])
        c3.metric("Uniqueness", pct(risk_raw["uniqueness_ratio"]))

        c4, c5, c6 = st.columns(3)
        c4.metric("Avg Class Size", f"{risk_raw['avg_equiv_class_size']:.2f}")
        c5.metric("Min Class Size", risk_raw["min_equiv_class_size"])
        c6.metric("Max Class Size", risk_raw["max_equiv_class_size"])

        st.caption("Baseline risk prior to anonymisation.")

    # ---------- 2) K-ANON (GENERALISATION ONLY) ----------
    df_k_gen = enforce_k_anonymity(df, qids=qids, k=k_value, suppress=False)
    assessor_gen = RiskAssessor(df_k_gen, qids)
    risk_gen = assessor_gen.risk_report()

    suppr_gen = suppression_rate(df, df_k_gen)

    tv_gen_rows = [
        {
            "column": col,
            "tv_distance (%)": categorical_tv_distance(df, df_k_gen, col) * 100,
        }
        for col in ["education", "native-country", "income"]
        if col in df.columns
    ]

    num_err_gen = numeric_mean_std_error(df, df_k_gen, "hours-per-week")

    with st.container(border=True):
        st.markdown(f"### K-Anonymity — Generalisation Only (k = {k_value})")

        c1, c2, c3 = st.columns(3)
        c1.metric("Records", risk_gen["records"])
        c2.metric("Equivalence Classes", risk_gen["num_equivalence_classes"])
        c3.metric("Uniqueness", pct(risk_gen["uniqueness_ratio"]))

        c4, c5, c6 = st.columns(3)
        c4.metric("Avg Class Size", f"{risk_gen['avg_equiv_class_size']:.2f}")
        c5.metric("Min Class Size", risk_gen["min_equiv_class_size"])
        c6.metric("Max Class Size", risk_gen["max_equiv_class_size"])

        st.markdown("**Utility vs Raw (after generalisation)**")
        st.metric("Suppression", pct(suppr_gen))

        with st.expander("Categorical TV Distance (Generalised vs Raw)"):
            st.table(pd.DataFrame(tv_gen_rows))

        c1, c2 = st.columns(2)
        c1.metric("Mean Error (hours/week)", pct(num_err_gen["mean_rel_error"]))
        c2.metric("Std Error (hours/week)", pct(num_err_gen["std_rel_error"]))

    # ---------- 3) K-ANON + SUPPRESSION (FINAL) ----------
    df_k_sup = enforce_k_anonymity(df, qids=qids, k=k_value, suppress=True)
    assessor_sup = RiskAssessor(df_k_sup, qids)
    risk_sup = assessor_sup.risk_report()

    suppr_final_raw = suppression_rate(df, df_k_sup)
    suppr_final_gen = suppression_rate(df_k_gen, df_k_sup)

    tv_sup_rows = [
        {
            "column": col,
            "tv_distance (%)": categorical_tv_distance(df, df_k_sup, col) * 100,
        }
        for col in ["education", "native-country", "income"]
        if col in df.columns
    ]

    num_err_sup = numeric_mean_std_error(df, df_k_sup, "hours-per-week")

    with st.container(border=True):
        st.markdown(f"### K-Anonymity + Suppression (Final Released Dataset) — k = {k_value}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Records", risk_sup["records"])
        c2.metric("Equivalence Classes", risk_sup["num_equivalence_classes"])
        c3.metric("Uniqueness", pct(risk_sup["uniqueness_ratio"]))

        c4, c5, c6 = st.columns(3)
        c4.metric("Avg Class Size", f"{risk_sup['avg_equiv_class_size']:.2f}")
        c5.metric("Min Class Size", risk_sup["min_equiv_class_size"])
        c6.metric("Max Class Size", risk_sup["max_equiv_class_size"])

        st.markdown("**Suppression Impact**")
        c1, c2 = st.columns(2)
        c1.metric("Suppression vs Raw", pct(suppr_final_raw))
        c2.metric("Suppression vs Generalised", pct(suppr_final_gen))

        with st.expander("Categorical TV Distance (Final vs Raw)"):
            st.table(pd.DataFrame(tv_sup_rows))

        c1, c2 = st.columns(2)
        c1.metric("Mean Error (hours/week)", pct(num_err_sup["mean_rel_error"]))
        c2.metric("Std Error (hours/week)", pct(num_err_sup["std_rel_error"]))

        st.markdown("Preview of Final Anonymised Data")
        st.dataframe(df_k_sup.head(50))

    # ---------- 4) OPTIONAL L-DIVERSITY ----------
    if enable_ldiv:
        st.markdown("----")

        with st.container(border=True):
            st.markdown("### L-Diversity (Distinct) — Sensitive Attribute = income")

            df_ldiv, l_report = enforce_l_diversity(
                df_k_sup,
                qids=qids,
                sensitive_attr="income",
                L=2,
                suppress=True,
            )

            st.json(l_report)

            st.metric("Records After L-Diversity", len(df_ldiv))
            st.metric("Suppression Rate", pct(l_report["suppression_rate"]))

        return df_ldiv

    return df_k_sup


# ------------------------------------------
# STREAMLIT MAIN UI
# ------------------------------------------
def main():
    st.set_page_config(
        page_title="SafeData — Privacy Utility Dashboard",
        layout="wide",
    )

    st.title("SafeData — Privacy & Utility Preserving Framework")

    df = load_data()
    summary = show_profile(df)

    tab1, tab2 = st.tabs(["Dataset & QID Candidates", "Risk–Utility Explorer"])

    with tab1:
        show_qid_analysis(df, summary["suggested_qids"])

    with tab2:
        default_policy_qids = [
            q for q in ["age", "sex", "education", "native-country"]
            if q in df.columns
        ]

        st.sidebar.header("Configuration")

        mode = st.sidebar.radio(
            "QID Selection Mode",
            ["Full Suggested (Mode A)", "Policy QIDs (Mode B)", "Custom (Mode C)"],
            index=1,
        )

        k_value = st.sidebar.slider("k-Anonymity (k value)", 2, 20, 5, step=1)

        enable_ldiv = st.sidebar.checkbox("Apply L-Diversity (Distinct)", value=False)

        if mode == "Full Suggested (Mode A)":
            qids = summary["suggested_qids"]

        elif mode == "Policy QIDs (Mode B)":
            qids = default_policy_qids

        else:
            qids = st.sidebar.multiselect(
                "Select QIDs",
                options=list(df.columns),
                default=default_policy_qids,
            )

        if st.sidebar.button("Run Analysis"):
            df_final = run_pipeline(df, qids, k_value, enable_ldiv)

            if df_final is not None:
                st.markdown("### Download Final Anonymised Dataset")
                st.download_button(
                    "Download CSV",
                    df_final.to_csv(index=False).encode("utf-8"),
                    "anonymised_output.csv",
                    "text/csv",
                )


if __name__ == "__main__":
    main()
