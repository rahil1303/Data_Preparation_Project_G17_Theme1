import pandas as pd
import numpy as np
import re
import ftfy
from cleanlab.classification import CleanLearning
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ABBREV_MAP = {
    r"tbh": "to be honest",
    r"w/": "with",
    r"u": "you",
    r"imo": "in my opinion",
    r"fyi": "for your information",
    r"pls": "please",
    r"thx": "thanks",
    r"plz": "please",
    r"btw": "by the way",
    r"cuz": "because",
    r"cos": "because",
    r"idk": "i do not know",
    r"asap": "as soon as possible",
    r"omg": "oh my god",
    r"lol": "laughing out loud",
    r"ppl": "people",
    r"hrs": "hours",
    r"mins": "minutes",
    r"wks": "weeks"
}

ABBREV_RE = re.compile(r'\b(' + '|'.join(re.escape(k) for k in ABBREV_MAP.keys()) + r')\b', flags=re.IGNORECASE)

NON_ALPHANUM = re.compile(r"[^a-zA-Z0-9\s']")
MULTI_SPACE = re.compile(r"\s+")

def salvage_numeric_str(x):
    s = str(x).strip()
    s = re.sub(r"[a-zA-Z]", "", s)  # Remove letters
    m = re.search(r"[+-]?\d+(\.\d+)?", s)
    if m:
        return abs(float(m.group(0)))
    return np.nan

def clean_basic(X, y, numeric_columns):
    initial_count = len(X)
    X = X.copy()
    if isinstance(y, pd.Series):
        y = y.copy()
    
    for col in numeric_columns:
        if col in X.columns:
            X[col] = X[col].apply(salvage_numeric_str)
    
    imputer = SimpleImputer(strategy='mean')
    X[numeric_columns] = imputer.fit_transform(X[numeric_columns])
    
    dropped = 0
    
    return X, y, {
        "dropped": dropped,
        "name": "Basic Cleaning (Salvage + Impute)",
        "columns_cleaned": numeric_columns
    }

def clean_heuristic(X, y, numeric_columns):
    initial_count = len(X)
    X = X.copy()
    y = y.copy()
    
    mask = pd.Series([True] * len(X), index=X.index)
    
    for col in numeric_columns:
        if col in X.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            col_mask = (X[col] >= lower_bound) & (X[col] <= upper_bound)
            mask = mask & col_mask
    
    X = X[mask]
    y = y[mask]
    
    scaler = StandardScaler()
    X[numeric_columns] = scaler.fit_transform(X[numeric_columns])
    
    dropped = initial_count - len(X)
    
    return X, y, {
        "dropped": dropped,
        "name": "Heuristic Cleaning (Outliers + Standardization)"
    }


def clean_domain(X, y, numeric_columns):
    initial_count = len(X)
    X = X.copy()
    y = y.copy()

    dropped = 0
    
    return X, y, {
        "dropped": dropped,
        "name": "Domain Cleaning (Light)"
    }

def clean_semantic(X, y, columns):
    X = X.copy()
    y = pd.Series(y).copy()

    initial_len = len(X)

    # get the columns that are numneric
    num_cols = [col for col in columns if col in X.columns and X[col].dtype in [np.float64, np.int64, np.float32, np.int32]]
    modified = 0
    for col in num_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        #replace no value with mean
        mean_val = X[col].mean()
        n_missing = X[col].isna().sum()

        if n_missing > 0:
            X[col].fillna(mean_val, inplace=True)
            modified += n_missing

    return X, y, {
        "dropped": initial_len - len(X),
        "name": "Semantic Cleaning (X, y, columns)",
        "modified": modified,
    }


def clean_model_aware(X, y, model_pipeline, numeric_columns, label_name='income', threshold=0.95):
    initial_count = len(X)
    X = X.copy()
    y = y.copy()
    
    try:
        X_numeric = X[numeric_columns].copy()
        
        y_codes = pd.Categorical(y).codes
        
        cl = CleanLearning(clf=model_pipeline, seed=42)
        cl.fit(X_numeric, y_codes)
        
        label_issues = cl.get_label_issues()
        is_flagged = label_issues['is_label_issue'].values
        
        probs = cl.predict_proba(X_numeric)
        max_conf = probs.max(axis=1)
        is_high_conf_suspect = is_flagged & (max_conf >= threshold)
        
        X_clean = X.loc[~is_high_conf_suspect].reset_index(drop=True)
        y_clean = y.loc[~is_high_conf_suspect].reset_index(drop=True)
        
        dropped = initial_count - len(X_clean)
        
        stats = {
            "n_issues": is_flagged.sum(),
            "n_removed": dropped,
            "issue_rate": is_flagged.sum() / initial_count,
            "avg_conf_clean": max_conf[~is_flagged].mean(),
            "avg_conf_noisy": max_conf[is_flagged].mean() if is_flagged.sum() > 0 else 0
        }
        
        return X_clean, y_clean, {
            "dropped": dropped,
            "name": "Model-Aware Cleaning (Cleanlab)",
            "stats": stats
        }
    
    except Exception as e:
        return X, y, {
            "dropped": 0,
            "name": "Model-Aware (Failed)",
            "stats": None,
            "error": str(e)
        }


def clean_progressive(X, y, numeric_columns, model_pipeline=None, levels=[1, 2, 3, 4]):
    X_clean = X.copy()
    y_clean = y.copy()
    metadata = {"levels_applied": [], "total_dropped": 0}
    
    if 1 in levels:
        X_clean, y_clean, meta = clean_basic(X_clean, y_clean, numeric_columns)
        metadata["levels_applied"].append("Basic")
        metadata["total_dropped"] += meta["dropped"]
        metadata["basic"] = meta
    
    if 2 in levels:
        X_clean, y_clean, meta = clean_heuristic(X_clean, y_clean, numeric_columns)
        metadata["levels_applied"].append("Heuristic")
        metadata["total_dropped"] += meta["dropped"]
        metadata["heuristic"] = meta
    
    if 3 in levels:
        X_clean, y_clean, meta = clean_semantic(X_clean, y_clean, numeric_columns)
        metadata["levels_applied"].append("Domain")
        metadata["total_dropped"] += meta["dropped"]
        metadata["domain"] = meta
    
    if 4 in levels and model_pipeline is not None:
        X_clean, y_clean, meta = clean_model_aware(X_clean, y_clean, model_pipeline, numeric_columns)
        metadata["levels_applied"].append("Model-Aware")
        metadata["total_dropped"] += meta["dropped"]
        metadata["model_aware"] = meta
    
    return X_clean, y_clean, metadata



#categorical cleaning

def clean_categorical_basic(X, y, categorical_columns):
    initial_count = len(X)
    X = X.copy()
    y = y.copy()
    mask = pd.Series([True] * len(X), index=X.index)
    
    for col in categorical_columns:
        if col in X.columns:
            value_counts = X[col].value_counts(normalize=True)
            valid_values = value_counts[value_counts >= 0.005].index
            
            col_mask = X[col].isin(valid_values)
            mask = mask & col_mask
    
    X = X[mask]
    y = y[mask]
    
    dropped = initial_count - len(X)
    
    return X, y, {
        "dropped": dropped,
        "name": "Categorical Basic Cleaning (Drop Corrupted)",
        "columns_cleaned": categorical_columns
    }


def clean_categorical_mode_imputation(X, y, categorical_columns):
    initial_count = len(X)
    X = X.copy()
    y = y.copy()
    
    replacements = {}
    
    for col in categorical_columns:
        if col in X.columns:
            # Get value counts
            value_counts = X[col].value_counts(normalize=True)
            mode_value = value_counts.index[0]  # Most common value
            
            # Replace rare values (< 0.5% frequency) with mode
            rare_mask = X[col].map(value_counts).fillna(0) < 0.005
            n_replaced = rare_mask.sum()
            
            if n_replaced > 0:
                X.loc[rare_mask, col] = mode_value
                replacements[col] = n_replaced
    
    dropped = 0
    
    return X, y, {
        "dropped": dropped,
        "name": "Categorical Mode Imputation",
        "replacements": replacements
    }


def clean_categorical_progressive(X, y, categorical_columns, levels=[1]):
    X_clean = X.copy()
    y_clean = y.copy()
    metadata = {"levels_applied": [], "total_dropped": 0}
    
    if 1 in levels:
        X_clean, y_clean, meta = clean_categorical_basic(X_clean, y_clean, categorical_columns)
        metadata["levels_applied"].append("Basic")
        metadata["total_dropped"] += meta["dropped"]
        metadata["basic"] = meta
    
    if 2 in levels:
        X_clean, y_clean, meta = clean_categorical_mode_imputation(X_clean, y_clean, categorical_columns)
        metadata["levels_applied"].append("Mode Imputation")
        metadata["mode_imputation"] = meta
    
    return X_clean, y_clean, metadata


def clean_all(df, model_pipeline=None, level="full"):
    level_map = {
        "basic": [1],
        "heuristic": [1, 2],
        "semantic": [1, 2, 3],
        "model-aware": [1, 2, 3, 4],
        "full": [1, 2, 3, 4]
    }
    
    levels = level_map.get(level, [1, 2, 3])
    df_clean, metadata = clean_progressive(df, model_pipeline, levels)
    return df_clean


def clean_all(X, y, numeric_columns, model_pipeline=None, level="full"):
    level_map = {
        "basic": [1],
        "heuristic": [1, 2],
        "domain": [1, 2, 3],
        "model-aware": [1, 2, 3, 4],
        "full": [1, 2, 3, 4]
    }
    
    levels = level_map.get(level, [1, 2, 3, 4])
    X_clean, y_clean, metadata = clean_progressive(X, y, numeric_columns, model_pipeline, levels)
    return X_clean, y_clean


def clean_simple(X, y, numeric_columns, model_pipeline, use_cleanlab=False):
    if use_cleanlab:
        levels = [1, 2, 4]
    else:
        levels = [1, 2]
    
    X_clean, y_clean, metadata = clean_progressive(
        X, y, numeric_columns, model_pipeline, levels=levels
    )
    
    return X_clean, y_clean