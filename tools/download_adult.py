import os
import pandas as pd
from sklearn.datasets import fetch_openml

def main() -> None:
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)

    # Download dataset
    adult = fetch_openml(name="adult", version=2, as_frame=True)
    df = adult.frame.copy()

    # Rename sensitive target column
    if "class" in df.columns:
        df = df.rename(columns={"class": "income"})

    # ✅ DROP UNNECESSARY COLUMN
    if "fnlwgt" in df.columns:
        df = df.drop(columns=["fnlwgt"])

    if "education-num" in df.columns:
        df = df.drop(columns=["education-num"])

    # Save cleaned dataset
    out_path = "C:\\Users\\USER\\Desktop\\SafeData\\data\\adult.csv"
    df.to_csv(out_path, index=False)

    print(f"✔ Saved dataset to {out_path}")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    print("Final columns:", list(df.columns))

if __name__ == "__main__":
    main()
