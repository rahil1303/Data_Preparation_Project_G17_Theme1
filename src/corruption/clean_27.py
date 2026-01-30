import pandas as pd
import numpy as np
import re
import ftfy
from cleanlab.classification import CleanLearning
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

"""
UNIFIED CLEANING LIBRARY
Combines:
- NEW functions: Text cleaning with 4-level hierarchy (basic → heuristic → semantic → model-aware)
- ADULT functions: Numeric cleaning with 4-level hierarchy (basic → heuristic → domain → model-aware)
"""

# ============================================================================
# NEW FUNCTIONS - Text Cleaning (4 Levels for Amazon)
# ============================================================================

# --- Level 1: Basic Cleaning ---

def NEW_clean_basic(df):
    """NEW Level 1: Basic - Drop Missing/Unicode Fix"""
    initial_count = len(df)
    df = df.copy()

    # 1. Count and Handle Missing Text
    df['text'] = df['text'].replace(r'^\s*$', np.nan, regex=True)
    missing_text_count = df['text'].isna().sum()

    # Drop rows where text is missing
    df = df.dropna(subset=['text'])

    # 2. Normalize and Unicode fix
    df['text'] = df['text'].astype(str).apply(lambda x: ftfy.fix_text(re.sub(r'\s+', ' ', x.strip().lower())))

    # 3. Label cleaning
    initial_labels = df['label'].isna().sum()
    df = df.dropna(subset=['label'])
    df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
    df = df[df['label'].between(1, 5)]

    dropped = initial_count - len(df)

    return df, {
        "dropped": dropped, 
        "name": "Basic Cleaning", 
        "missing_text": missing_text_count,
        "missing_labels": initial_labels
    }


# --- Level 2: Heuristic Cleaning ---

def NEW_clean_heuristic(df):
    """NEW Level 2: Heuristic - Statistical Outlier Detection"""
    initial_count = len(df)
    df = df.copy()
    text = df["text"].astype(str)

    # 1. Length filter
    char_len = text.str.len()
    len_mask = char_len > 3

    # 2. Alpha ratio
    alpha_ratio = text.str.count(r"[a-zA-Z]") / char_len.replace(0, 1)
    alpha_mask = alpha_ratio >= 0.1

    # 3. Vocabulary richness
    tokens = text.str.split()
    unique_ratio = tokens.apply(
        lambda t: len(set(t)) / len(t) if len(t) > 0 else 0
    )
    richness_mask = unique_ratio >= 0.05

    # 4. Repeated-character spam
    repeated_char_mask = ~text.str.contains(r"(.)\1{10,}", regex=True)

    # Combine
    final_mask = len_mask & alpha_mask & richness_mask & repeated_char_mask
    df_clean = df[final_mask]

    dropped = initial_count - len(df_clean)

    return df_clean, {
        "dropped": dropped,
        "name": "Heuristic Cleaning"
    }


# --- Level 3: Semantic Cleaning ---

# Abbreviations / slang
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

# Precompiled regexes
NON_ALPHANUM = re.compile(r"[^a-zA-Z0-9\s']")
MULTI_SPACE = re.compile(r"\s+")

def NEW_clean_semantic(df):
    """NEW Level 3: Semantic - Abbreviation expansion, negation preservation"""
    df = df.copy()
    initial_text = df['text'].copy()

    # --- Step 1: Lowercase & replace abbreviations ---
    def expand_abbrev(text):
        return ABBREV_RE.sub(lambda m: ABBREV_MAP[m.group(0).lower()], text.lower())

    df['text'] = df['text'].astype(str).map(expand_abbrev)

    # Squash 3+ chars to 2
    df['text'] = df['text'].str.replace(r'(.)\1{2,}', r'\1\1', regex=True)

    # --- Step 2: Remove unwanted characters but keep apostrophes ---
    df['text'] = df['text'].map(lambda x: NON_ALPHANUM.sub(" ", x))

    # --- Step 3: Normalize whitespace ---
    df['text'] = df['text'].map(lambda x: MULTI_SPACE.sub(" ", x).strip())

    # --- Step 4: Token-level filtering with negation preservation ---
    def join_negations(text):
        tokens = text.split()
        if not tokens: 
            return ""
        neg_words = {
            "not", "no", "never", "none",
            "dont", "didnt", "wasnt", "isnt", "arent", "werent", "cant", "couldnt",
            "don't", "didn't", "wasn't", "isn't", "aren't", "weren't", "can't", "couldn't"
        }
        cleaned = []
        i = 0
        while i < len(tokens):
            if tokens[i] in neg_words and i + 1 < len(tokens):
                cleaned.append(f"{tokens[i]}_{tokens[i+1]}")
                i += 2
            else:
                if len(tokens[i]) > 1:
                    cleaned.append(tokens[i])
                i += 1
        return " ".join(cleaned)

    df['text'] = df['text'].map(join_negations)

    # --- Step 5: Fallback for empty text ---
    df['text'] = df['text'].replace("", "neutral_content")

    num_modified = (initial_text != df['text']).sum()

    return df, {
        "dropped": 0, 
        "name": "Semantic Cleaning", 
        "modified": num_modified
    }


# --- Level 4: Model-Aware Cleaning (Cleanlab) ---

def NEW_clean_model_aware(df, model_pipeline):
    """NEW Level 4: Model-Aware - Uses Cleanlab to detect label noise"""
    initial_count = len(df)
    df = df.copy()
    df['label_indexed'] = df['label'] - 1

    try:
        X_data = df['text'].fillna("").astype(str).values
        y_data = df['label_indexed'].values
        cl = CleanLearning(clf=model_pipeline, cv_n_folds=3)
        cl.fit(X_data, y_data)

        label_issues = cl.get_label_issues()
        is_issue = label_issues['is_label_issue'].values
        n_issues = is_issue.sum()

        # Confidence and Noise logic
        probs = cl.predict_proba(X_data)
        max_conf = probs.max(axis=1)

        stats = {
            "n_issues": n_issues,
            "issue_rate": n_issues / initial_count,
            "avg_conf_clean": max_conf[~is_issue].mean(),
            "avg_conf_noisy": max_conf[is_issue].mean(),
            "class_noise": df.assign(is_issue=is_issue).groupby("label")["is_issue"].mean().to_dict()
        }

        df_clean = df[~is_issue].copy().drop(columns=['label_indexed'])
        return df_clean, {
            "dropped": initial_count - len(df_clean), 
            "name": "Model-Aware Cleaning", 
            "stats": stats
        }

    except Exception as e:
        return df, {
            "dropped": 0, 
            "name": "Model-Aware (Failed)", 
            "stats": None,
            "error": str(e)
        }


# --- NEW Master Pipeline: Progressive Cleaning ---

def NEW_clean_progressive(df, model_pipeline=None, levels=[1, 2, 3, 4]):
    """
    NEW Master Pipeline: Apply cleaning levels progressively
    
    Args:
        df: Input dataframe
        model_pipeline: sklearn pipeline for Level 4 (model-aware)
        levels: List of levels to apply [1, 2, 3, 4]
    
    Returns:
        df_clean: Cleaned dataframe
        metadata: Dict with info from each level
    """
    df_clean = df.copy()
    metadata = {"levels_applied": [], "total_dropped": 0}
    
    if 1 in levels:
        df_clean, meta = NEW_clean_basic(df_clean)
        metadata["levels_applied"].append("Basic")
        metadata["total_dropped"] += meta["dropped"]
        metadata["basic"] = meta
    
    if 2 in levels:
        df_clean, meta = NEW_clean_heuristic(df_clean)
        metadata["levels_applied"].append("Heuristic")
        metadata["total_dropped"] += meta["dropped"]
        metadata["heuristic"] = meta
    
    if 3 in levels:
        df_clean, meta = NEW_clean_semantic(df_clean)
        metadata["levels_applied"].append("Semantic")
        metadata["semantic"] = meta
    
    if 4 in levels and model_pipeline is not None:
        df_clean, meta = NEW_clean_model_aware(df_clean, model_pipeline)
        metadata["levels_applied"].append("Model-Aware")
        metadata["total_dropped"] += meta["dropped"]
        metadata["model_aware"] = meta
    
    return df_clean, metadata


# ============================================================================
# ADULT FUNCTIONS - Numeric Cleaning (4 Levels for Adult Income)
# ============================================================================

def ADULT_salvage_numeric_str(x):
    """Helper: Extract numbers from corrupted strings like '25.0x' -> 25.0"""
    s = str(x).strip()
    s = re.sub(r"[a-zA-Z]", "", s)  # Remove letters
    m = re.search(r"[+-]?\d+(\.\d+)?", s)
    if m:
        return abs(float(m.group(0)))
    return np.nan


# --- Level 1: Basic Cleaning ---

def ADULT_clean_basic(X, y, numeric_columns):
    """ADULT Level 1: Basic - Salvage corrupted numbers & impute missing"""
    initial_count = len(X)
    X = X.copy()
    if isinstance(y, pd.Series):
        y = y.copy()
    
    # Step 1: Salvage numeric strings ('25.0x' -> 25.0)
    for col in numeric_columns:
        if col in X.columns:
            X[col] = X[col].apply(ADULT_salvage_numeric_str)
    
    # Step 2: Impute missing values with mean
    imputer = SimpleImputer(strategy='mean')
    X[numeric_columns] = imputer.fit_transform(X[numeric_columns])
    
    dropped = 0  # No rows dropped at this level
    
    return X, y, {
        "dropped": dropped,
        "name": "Basic Cleaning (Salvage + Impute)",
        "columns_cleaned": numeric_columns
    }


# --- Level 2: Heuristic Cleaning ---

def ADULT_clean_heuristic(X, y, numeric_columns):
    """ADULT Level 2: Heuristic - Outlier removal & standardization"""
    initial_count = len(X)
    X = X.copy()
    y = y.copy()
    
    # Step 1: Remove outliers using IQR method
    mask = pd.Series([True] * len(X), index=X.index)
    
    for col in numeric_columns:
        if col in X.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR  # 3 IQR (less aggressive than 1.5)
            upper_bound = Q3 + 3 * IQR
            
            col_mask = (X[col] >= lower_bound) & (X[col] <= upper_bound)
            mask = mask & col_mask
    
    X = X[mask]
    y = y[mask]
    
    # Step 2: Standardize (Z-score normalization)
    scaler = StandardScaler()
    X[numeric_columns] = scaler.fit_transform(X[numeric_columns])
    
    dropped = initial_count - len(X)
    
    return X, y, {
        "dropped": dropped,
        "name": "Heuristic Cleaning (Outliers + Standardization)"
    }


# --- Level 3: Domain Cleaning ---

def ADULT_clean_domain(X, y, numeric_columns):
    """ADULT Level 3: Domain - Domain-specific rules (optional/light)"""
    initial_count = len(X)
    X = X.copy()
    y = y.copy()
    
    # Domain rules for Adult dataset
    # Note: These are applied AFTER standardization, so we're working with z-scores
    # This level is intentionally light - just catching extreme violations
    
    # No additional filtering needed since standardization already handles most issues
    # This level exists for structural consistency with NEW's 4-level approach
    
    dropped = 0
    
    return X, y, {
        "dropped": dropped,
        "name": "Domain Cleaning (Light)"
    }


# --- Level 4: Model-Aware Cleaning (Cleanlab) ---

def ADULT_clean_model_aware(X, y, model_pipeline, numeric_columns, label_name='income', threshold=0.95):
    """ADULT Level 4: Model-Aware - Cleanlab label noise detection"""
    initial_count = len(X)
    X = X.copy()
    y = y.copy()
    
    try:
        # ✅ FIX: Use only numeric columns for Cleanlab (model expects only these)
        X_numeric = X[numeric_columns].copy()
        
        # Combine X and y for Cleanlab
        y_codes = pd.Categorical(y).codes
        
        cl = CleanLearning(clf=model_pipeline, seed=42)
        cl.fit(X_numeric, y_codes)
        
        label_issues = cl.get_label_issues()
        is_flagged = label_issues['is_label_issue'].values
        
        # Confidence filtering
        probs = cl.predict_proba(X_numeric)
        max_conf = probs.max(axis=1)
        is_high_conf_suspect = is_flagged & (max_conf >= threshold)
        
        # Remove high-confidence mislabeled samples from FULL X (keeps all columns)
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


# --- ADULT Master Pipeline: Progressive Cleaning ---

def ADULT_clean_progressive(X, y, numeric_columns, model_pipeline=None, levels=[1, 2, 3, 4]):
    """
    ADULT Master Pipeline: Apply cleaning levels progressively
    
    Args:
        X: Feature dataframe
        y: Target series
        numeric_columns: List of numeric column names
        model_pipeline: sklearn pipeline for Level 4 (model-aware)
        levels: List of levels to apply [1, 2, 3, 4]
    
    Returns:
        X_clean: Cleaned features
        y_clean: Cleaned targets
        metadata: Dict with info from each level
    """
    X_clean = X.copy()
    y_clean = y.copy()
    metadata = {"levels_applied": [], "total_dropped": 0}
    
    if 1 in levels:
        X_clean, y_clean, meta = ADULT_clean_basic(X_clean, y_clean, numeric_columns)
        metadata["levels_applied"].append("Basic")
        metadata["total_dropped"] += meta["dropped"]
        metadata["basic"] = meta
    
    if 2 in levels:
        X_clean, y_clean, meta = ADULT_clean_heuristic(X_clean, y_clean, numeric_columns)
        metadata["levels_applied"].append("Heuristic")
        metadata["total_dropped"] += meta["dropped"]
        metadata["heuristic"] = meta
    
    if 3 in levels:
        X_clean, y_clean, meta = ADULT_clean_domain(X_clean, y_clean, numeric_columns)
        metadata["levels_applied"].append("Domain")
        metadata["total_dropped"] += meta["dropped"]
        metadata["domain"] = meta
    
    if 4 in levels and model_pipeline is not None:
        X_clean, y_clean, meta = ADULT_clean_model_aware(X_clean, y_clean, model_pipeline, numeric_columns)
        metadata["levels_applied"].append("Model-Aware")
        metadata["total_dropped"] += meta["dropped"]
        metadata["model_aware"] = meta
    
    return X_clean, y_clean, metadata


# ============================================================================
# CATEGORICAL CLEANING FUNCTIONS - For Adult Income Categorical Features (NEW!)
# ============================================================================

def ADULT_clean_categorical_basic(X, y, categorical_columns):
    """
    CATEGORICAL Level 1: Basic - Drop rows with unknown/corrupted categorical values
    
    Simple approach: Remove rows where categorical values don't match known categories
    This is appropriate for typos, defaults, and invalid categories
    """
    initial_count = len(X)
    X = X.copy()
    y = y.copy()
    
    # Track valid categories for each column (from original data before corruption)
    # In practice, we'll use a heuristic: drop rows with very rare values (< 1% frequency)
    mask = pd.Series([True] * len(X), index=X.index)
    
    for col in categorical_columns:
        if col in X.columns:
            # Get value counts
            value_counts = X[col].value_counts(normalize=True)
            
            # Keep only values that appear in at least 0.5% of rows
            # This catches "Default", typos, and other corrupted values
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


def ADULT_clean_categorical_mode_imputation(X, y, categorical_columns):
    """
    CATEGORICAL Level 2: Mode Imputation - Replace rare/corrupted values with mode
    
    Instead of dropping rows, replace corrupted values with the most common value
    """
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
    
    dropped = 0  # No rows dropped, just imputation
    
    return X, y, {
        "dropped": dropped,
        "name": "Categorical Mode Imputation",
        "replacements": replacements
    }


def ADULT_clean_categorical_progressive(X, y, categorical_columns, levels=[1]):
    """
    CATEGORICAL Master Pipeline: Apply cleaning levels progressively
    
    Args:
        X: Feature dataframe
        y: Target series
        categorical_columns: List of categorical column names
        levels: List of levels to apply [1, 2]
    
    Returns:
        X_clean: Cleaned features
        y_clean: Cleaned targets
        metadata: Dict with info from each level
    """
    X_clean = X.copy()
    y_clean = y.copy()
    metadata = {"levels_applied": [], "total_dropped": 0}
    
    if 1 in levels:
        X_clean, y_clean, meta = ADULT_clean_categorical_basic(X_clean, y_clean, categorical_columns)
        metadata["levels_applied"].append("Basic")
        metadata["total_dropped"] += meta["dropped"]
        metadata["basic"] = meta
    
    if 2 in levels:
        X_clean, y_clean, meta = ADULT_clean_categorical_mode_imputation(X_clean, y_clean, categorical_columns)
        metadata["levels_applied"].append("Mode Imputation")
        metadata["mode_imputation"] = meta
    
    return X_clean, y_clean, metadata


# ============================================================================
# CONVENIENCE WRAPPERS
# ============================================================================

def NEW_clean_all(df, model_pipeline=None, level="full"):
    """
    Convenience wrapper for NEW text cleaning
    
    Args:
        level: 'basic', 'heuristic', 'semantic', 'model-aware', 'full'
    """
    level_map = {
        "basic": [1],
        "heuristic": [1, 2],
        "semantic": [1, 2, 3],
        "model-aware": [1, 2, 3, 4],
        "full": [1, 2, 3, 4]
    }
    
    levels = level_map.get(level, [1, 2, 3])
    df_clean, metadata = NEW_clean_progressive(df, model_pipeline, levels)
    return df_clean


def ADULT_clean_all(X, y, numeric_columns, model_pipeline=None, level="full"):
    """
    Convenience wrapper for ADULT numeric cleaning
    
    Args:
        level: 'basic', 'heuristic', 'domain', 'model-aware', 'full'
    """
    level_map = {
        "basic": [1],
        "heuristic": [1, 2],
        "domain": [1, 2, 3],
        "model-aware": [1, 2, 3, 4],
        "full": [1, 2, 3, 4]
    }
    
    levels = level_map.get(level, [1, 2, 3, 4])
    X_clean, y_clean, metadata = ADULT_clean_progressive(X, y, numeric_columns, model_pipeline, levels)
    return X_clean, y_clean


def ADULT_clean_simple(X, y, numeric_columns, model_pipeline, use_cleanlab=False):
    """
    Simple 2-mode ADULT cleaning (matches Adult notebook usage)
    
    Mimics: clean_num.run_num_clean(numeric_features, X, y, clf, use_cleanlab)
    
    Args:
        X: Feature dataframe
        y: Target series
        numeric_columns: List of numeric column names
        model_pipeline: sklearn pipeline
        use_cleanlab: If False, runs Levels 1+2 only. If True, runs Levels 1+2+4 (with Cleanlab)
    
    Returns:
        X_clean: Cleaned features
        y_clean: Cleaned targets
    
    Usage:
        # Mode 1: Basic cleaning only
        X_clean, y_clean = ADULT_clean_simple(X, y, numeric_cols, clf, use_cleanlab=False)
        
        # Mode 2: Basic cleaning + Cleanlab
        X_clean, y_clean = ADULT_clean_simple(X, y, numeric_cols, clf, use_cleanlab=True)
    """
    if use_cleanlab:
        # Mode 2: Levels 1+2+4 (Basic + Heuristic + Model-Aware)
        levels = [1, 2, 4]
    else:
        # Mode 1: Levels 1+2 only (Basic + Heuristic)
        levels = [1, 2]
    
    X_clean, y_clean, metadata = ADULT_clean_progressive(
        X, y, numeric_columns, model_pipeline, levels=levels
    )
    
    return X_clean, y_clean


print("✅ Unified cleaning functions loaded (NEW text + ADULT numeric 4-level hierarchies)")