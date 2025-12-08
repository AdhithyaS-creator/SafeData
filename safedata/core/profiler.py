# safedata/core/profiler.py

import pandas as pd
from typing import Dict, Any, List


class DataProfiler:
    """
    Data profiling module.
    Computes:
    - basic summary stats
    - missing values
    - unique counts
    - heuristic QID suggestions
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def basic_summary(self) -> pd.DataFrame:
        """Describe all columns (numeric + categorical)."""
        return self.df.describe(include="all").transpose()

    def missing_values(self) -> pd.Series:
        """Count of missing values per column."""
        return self.df.isna().sum()

    def unique_counts(self) -> pd.Series:
        """Number of unique values per column."""
        return self.df.nunique()

    def infer_qids(self, max_unique_ratio: float = 0.5) -> List[str]:
        """
        Heuristic QID detector:
        - ignore columns that are constant
        - ignore columns that are almost all-unique
        """
        n = len(self.df)
        qids: List[str] = []
        for col in self.df.columns:
            u = self.df[col].nunique()
            if 1 < u < max_unique_ratio * n:
                qids.append(col)
        return qids

    def summary_dict(self) -> Dict[str, Any]:
        """Pack key stats into a dict (good for CLI / UI)."""
        return {
            "rows": len(self.df),
            "cols": self.df.shape[1],
            "missing": self.missing_values().to_dict(),
            "unique": self.unique_counts().to_dict(),
            "suggested_qids": self.infer_qids(),
        }
