# safedata/core/risk.py

import pandas as pd
from typing import List, Dict, Any


class RiskAssessor:
    """
    Re-identification risk assessment based on equivalence
    classes defined over quasi-identifiers (QIDs).
    """

    def __init__(self, df: pd.DataFrame, qids: List[str]) -> None:
        if not qids:
            raise ValueError("RiskAssessor requires at least one QID.")
        self.df = df
        self.qids = qids

    def equivalence_class_sizes(self) -> pd.Series:
        """
        Size of each equivalence class defined by QIDs.
        """
        return self.df.groupby(self.qids, observed=True).size()

    def uniqueness_ratio(self) -> float:
        """
        Fraction of records that are unique on QIDs.
        """
        sizes = self.equivalence_class_sizes()
        unique_count = (sizes == 1).sum()
        return unique_count / len(self.df)

    def risk_report(self) -> Dict[str, Any]:
        """
        Key risk metrics for reporting:
        - number of records
        - number of equivalence classes
        - uniqueness ratio
        - average / min / max equivalence class sizes
        """
        sizes = self.equivalence_class_sizes()
        return {
            "records": int(len(self.df)),
            "num_equivalence_classes": int(len(sizes)),
            "uniqueness_ratio": float((sizes == 1).sum() / len(self.df)),
            "avg_equiv_class_size": float(sizes.mean()),
            "min_equiv_class_size": int(sizes.min()),
            "max_equiv_class_size": int(sizes.max()),
        }
