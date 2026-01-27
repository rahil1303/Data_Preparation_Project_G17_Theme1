import re
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from cleanlab.classification import CleanLearning
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from AutoClean import AutoClean

def salvage_numeric_str(x) -> float:
    """Extract the first integer-like token from a value (string/others). Returns np.nan if none."""
    s = str(x).strip()
    s = re.sub(r"[a-zA-Z]", "", s)
    m = re.search(r"[+-]?\d+(\.\d+)?", s)
    if m:
        return abs(int(float(m.group(0)))) 
    return np.nan


def salvage_numeric_columns(
    df: pd.DataFrame,
    numeric_cols: list[str],
) -> pd.DataFrame:
    """
    Apply salvage_numeric_str to each entry of the specified numeric columns.

    - Operates column-by-column (pandas 2.x safe)
    - Preserves index and column order
    - Does NOT mutate the input DataFrame
    """
    df_out = df.copy()

    for col in numeric_cols:
        if col not in df_out.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")

        df_out[col] = df_out[col].apply(salvage_numeric_str)

    return df_out

def impute_columns(
    df: pd.DataFrame,
    numeric_cols: list,
) -> pd.DataFrame:

    df_out = df.copy()

    imputer = SimpleImputer()

    df_out[numeric_cols] = imputer.fit_transform(df_out[numeric_cols])

    return df_out

def run_cleanlab(df_cleaned, clf, label_col, threshold=0.98):
    y = df_cleaned[label_col]
    X = df_cleaned.drop(columns=[label_col])

    y_codes = pd.Categorical(y).codes
    cl = CleanLearning(clf, seed=42)
    cl.fit(X, y_codes)

    label_issues = cl.get_label_issues()
    is_flagged = label_issues['is_label_issue'].values

    probs = cl.predict_proba(X)
    max_conf = probs.max(axis=1)
    is_high_conf_suspect = is_flagged & (max_conf >= threshold)

    return X.loc[~is_high_conf_suspect].reset_index(drop=True), y.loc[~is_high_conf_suspect].reset_index(drop=True)

def run_num_clean(numeric_features, X, y, clf, use_cleanlab=False):
    numericified = salvage_numeric_columns(X, numeric_features)

    imputed=impute_columns(numericified, numeric_cols=numeric_features)
    scaler = StandardScaler()
    imputed[numeric_features] = scaler.fit_transform(imputed[numeric_features])

    df_combined_for_autoclean = pd.concat([imputed, y], axis=1) 
    label_col = 'income'  

    cleaner = AutoClean(
        df_combined_for_autoclean,
        mode='auto'
    )

    df = cleaner.output.reset_index(drop=True) 
    if use_cleanlab:
        return run_cleanlab(df, clf, label_col)
    
    return df.drop(columns=[label_col]), df[label_col]
