import numpy as np
import pandas as pd


def add_numerical_features(df: pd.DataFrame, current_year: int = 1405) -> pd.DataFrame:
    """
    Add AreaPerRoom and Age features.

    Domain rules:
    - Room == 0 is treated as 1 (studio assumption) to avoid division by zero.
      Row removal for invalid rooms is the responsibility of preprocessing, not here.
    - YearOfConstruction is dropped after Age is computed — it carries the same
      information in a less model-friendly form.
    """
    df = df.copy()

    n_zero_rooms = (df["Room"] == 0).sum()
    if n_zero_rooms > 0:
        print(f"Warning: {n_zero_rooms} rows had Room=0, treated as Room=1 for AreaPerRoom")

    df["AreaPerRoom"] = df["Area"] / df["Room"].replace(0, 1)
    df["Age"] = current_year - df["YearOfConstruction"]
    df = df.drop(columns=["YearOfConstruction"])

    return df


def add_transformation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add LogPrice as log1p transform of Price.

    Using log1p (instead of log) is safe for zero-price edge cases,
    and keeps the transform invertible via np.expm1.
    """
    df = df.copy()
    df["LogPrice"] = np.log1p(df["Price"])
    return df