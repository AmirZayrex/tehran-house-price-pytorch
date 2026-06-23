import pandas as pd
from sklearn.model_selection import KFold


def add_location_score(
    df: pd.DataFrame,
    k: int = 20,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """
    Add LocationScore using Bayesian (smoothed) target encoding with K-Fold
    out-of-fold computation.

    Why target encoding?
        Address is a high-cardinality categorical feature. One-hot encoding
        would produce hundreds of sparse columns. Instead, we replace each
        address with a single numeric score that reflects the average LogPrice
        in that neighborhood.

    Why smoothing (Bayesian averaging)?
        A neighborhood with only one listing would get a score equal to that
        one listing's price — highly unreliable. Smoothing pulls low-count
        neighborhoods toward the global mean:

            score = (count * mean + k * global_mean) / (count + k)

        where k controls the strength of the pull. Higher k = more shrinkage
        toward the global mean for small neighborhoods.

    Why K-Fold out-of-fold?
        Computing a row's LocationScore from a dataset that includes the row
        itself would constitute data leakage — the feature would encode the
        row's own target. K-Fold OOF prevents this: each row's score is
        computed exclusively from the other folds.

    Args:
        df:           DataFrame containing Address and LogPrice columns.
        k:            Smoothing strength. Default 20 is a reasonable starting
                      point; increase for noisier datasets.
        n_splits:     Number of KFold splits. Higher values reduce variance in
                      the score estimates at the cost of compute time.
        random_state: Seed for KFold shuffle.

    Returns:
        df:             Input DataFrame with LocationScore column added.
        location_dict:  Mapping of Address -> score computed on the full
                        training set. Used to encode the test set without
                        leakage (test set was never part of this computation).
    """
    df = df.copy()
    global_mean = df["LogPrice"].mean()
    df["LocationScore"] = global_mean  # fallback for unseen addresses

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for train_idx, val_idx in kf.split(df):
        fold_train = df.iloc[train_idx]
        fold_val_index = df.index[val_idx]

        stats = fold_train.groupby("Address")["LogPrice"].agg(["mean", "count"])
        stats["score"] = (
            (stats["mean"] * stats["count"] + global_mean * k)
            / (stats["count"] + k)
        )

        df.loc[fold_val_index, "LocationScore"] = (
            df.loc[fold_val_index, "Address"]
            .map(stats["score"])
            .fillna(global_mean)
        )

    # Build the final lookup dict from the full training set.
    # This is used to encode X_test and is leakage-free because X_test
    # was split off before this function is ever called.
    final_stats = df.groupby("Address")["LogPrice"].agg(["mean", "count"])
    final_stats["score"] = (
        (final_stats["mean"] * final_stats["count"] + global_mean * k)
        / (final_stats["count"] + k)
    )
    location_dict = final_stats["score"].to_dict()

    print(f"LocationScore added — {len(location_dict)} unique addresses encoded (n_splits={n_splits}, k={k})")

    return df, location_dict