import pandas as pd
import numpy as np

from AutoClean import AutoClean  # pip install py-AutoClean
from cleanlab.classification import CleanLearning
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer

def clean_model_aware_numeric(
    df,
    model_pipeline,
    label_col="label",
    numeric_cols=None,
    cv_folds=3
):
    df = df.copy()
    initial_count = len(df)

    if numeric_cols is None:
        raise ValueError("numeric_cols must be specified")

    # ensure labels start at 0
    y = df[label_col].values
    if y.min() != 0:
        y = y - y.min()

    try:
        X = df[numeric_cols]

        cl = CleanLearning(clf=model_pipeline, cv_n_folds=cv_folds, seed=42)
        cl.fit(X, y)

        label_issues = cl.get_label_issues()
        is_issue = label_issues["is_label_issue"].values

        # confidence stats
        probs = cl.predict_proba(X)
        max_conf = probs.max(axis=1)

        stats = {
            "n_issues": int(is_issue.sum()),
            "issue_rate": float(is_issue.sum() / initial_count),
            "avg_conf_clean": float(max_conf[~is_issue].mean()),
            "avg_conf_noisy": float(max_conf[is_issue].mean()),
            "class_noise": df.assign(is_issue=is_issue).groupby(label_col)["is_issue"].mean().to_dict()
        }

        df_clean = df.loc[~is_issue].copy()

        return df_clean, {
            "dropped": initial_count - len(df_clean),
            "name": "Model-Aware Numeric Cleaning (cleanlab)",
            "stats": stats
        }

    except Exception as e:
        return df, {
            "dropped": 0,
            "name": "Model-Aware Numeric Cleaning (Failed)",
            "stats": None,
            "error": str(e)
        }


def scale_robust(series):
    scaler = RobustScaler()
    return pd.Series(scaler.fit_transform(series.values.reshape(-1, 1)).ravel(), index=series.index)

def power_transform(series, method="yeo-johnson"):
    pt = PowerTransformer(method=method)
    return pd.Series(pt.fit_transform(series.values.reshape(-1, 1)).ravel(), index=series.index)

def outliers_isolation_forest(df, numeric_cols, contamination=0.01):
    model = IsolationForest(contamination=contamination, random_state=42)
    mask = model.fit_predict(df[numeric_cols]) == 1
    return df.loc[mask]


def clean_adult_dataset(df, model_pipeline, label_col="income", numeric_cols=None, use_model_cleaning=True):
    """
    Mixed cleaning for UCI Adult dataset:
    - AutoClean handles structure, missing values, encoding, basic outlier winsorization
    - Optional numeric transformations & isolation forest
    - Optional Cleanlab row-level label noise detection
    """
    df = df.copy()

    # ---------- Step 1: AutoClean structural + missing + encoding ----------
    cleaner = AutoClean(df, mode="auto")
    df = cleaner.output

    # ---------- Step 2: Optional numeric transformations ----------
    if numeric_cols is not None:
        for col in numeric_cols:
            # Robust scaling
            df[col] = scale_robust(df[col])
            # Power transform (Yeo-Johnson)
            df[col] = power_transform(df[col])

    # ---------- Step 3: Optional IsolationForest outlier removal ----------
    if numeric_cols is not None:
        df = outliers_isolation_forest(df, numeric_cols)

    # ---------- Step 4: Model-aware cleaning with cleanlab ----------
    if use_model_cleaning and numeric_cols is not None:
        df, model_stats = clean_model_aware_numeric(
            df, model_pipeline=model_pipeline,
            label_col=label_col,
            numeric_cols=numeric_cols
        )
    else:
        model_stats = None

    return df, model_stats
