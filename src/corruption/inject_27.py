from jenga.corruptions.generic import MissingValues, SwappedValues, CategoricalShift
from jenga.corruptions.text import BrokenCharacters
import pandas as pd
import numpy as np
import random

"""
UNIFIED CORRUPTION LIBRARY
Combines:
- NEW functions: Text corruptions for Amazon (moderate 12-30% fractions)
- ADULT functions: Numeric corruptions for Adult Income dataset
- CATEGORICAL functions: Categorical corruptions for Adult Income dataset (NEW!)
"""

# ============================================================================
# NEW FUNCTIONS - Text Corruptions for Amazon (12-30%)
# ============================================================================

def NEW_apply_missing_values(df, fraction=0.30):
    """NEW Batch 01: Missing Values in text - MODERATE 30%"""
    df = df.copy()
    mv = MissingValues(column="text", fraction=fraction, missingness="MCAR")
    return mv.transform(df)

def NEW_apply_broken_characters(df, fraction=0.25):
    """NEW Batch 02: Broken Characters in text - MODERATE 25%"""
    df = df.copy()
    bc = BrokenCharacters(column="text", fraction=fraction)
    return bc.transform(df)

def NEW_apply_swapped_text(df, fraction=0.20):
    """NEW Batch 03: Swapped text values - MODERATE 20% (CUSTOM - fixes JENGA cross-column issue)"""
    df = df.copy()
    np.random.seed(42)

    n_rows = len(df)
    n_swap = int(fraction * n_rows)

    # Select rows to swap
    swap_idx = np.random.choice(df.index, size=n_swap, replace=False)
    shuffled_idx = np.random.permutation(swap_idx)

    # Swap 'text' values across rows (stays within text column)
    df.loc[swap_idx, 'text'] = df.loc[shuffled_idx, 'text'].values

    return df

def NEW_apply_missing_labels(df, fraction=0.15):
    """NEW Batch 04: Missing Labels - MODERATE 15%"""
    df = df.copy()
    mv = MissingValues(column="label", fraction=fraction, missingness="MCAR")
    return mv.transform(df)

def NEW_apply_swapped_labels(df, fraction=0.12):
    """NEW Batch 05: Swapped Labels - MODERATE 12% (CUSTOM - fixes JENGA cross-column issue)"""
    df = df.copy()
    np.random.seed(42)

    n_rows = len(df)
    n_swap = int(fraction * n_rows)

    # Select rows to swap
    swap_idx = np.random.choice(df.index, size=n_swap, replace=False)
    shuffled_idx = np.random.permutation(swap_idx)

    # Swap 'label' values across rows (stays within label column)
    df.loc[swap_idx, 'label'] = df.loc[shuffled_idx, 'label'].values

    return df

def NEW_apply_combined_text_corruption(df):
    """NEW Batch 06: Broken Chars (10%) + Missing (8%) - MODERATE"""
    df = df.copy()
    bc = BrokenCharacters(column="text", fraction=0.10)
    df = bc.transform(df)
    mv = MissingValues(column="text", fraction=0.08, missingness="MCAR")
    return mv.transform(df)

def NEW_apply_combined_text_labels(df):
    """NEW Batch 07: Swapped Text (15%) + Swapped Labels (8%) - MODERATE"""
    df = df.copy()

    # Swap 15% of text
    df = NEW_apply_swapped_text(df, fraction=0.15)

    # Swap 8% of labels
    df = NEW_apply_swapped_labels(df, fraction=0.08)

    return df

def NEW_apply_heavy_missing(df):
    """NEW Batch 08: Heavy Missing - Text (25%) + Labels (10%) - MODERATE"""
    df = df.copy()
    mv_text = MissingValues(column="text", fraction=0.25, missingness="MCAR")
    df = mv_text.transform(df)
    mv_label = MissingValues(column="label", fraction=0.10, missingness="MCAR")
    return mv_label.transform(df)

def NEW_apply_all_corruptions(df):
    """NEW Batch 09: All - Broken (8%) + Swapped (10%) + Missing (5%) - MODERATE"""
    df = df.copy()
    bc = BrokenCharacters(column="text", fraction=0.08)
    df = bc.transform(df)
    df = NEW_apply_swapped_text(df, fraction=0.10)
    mv_text = MissingValues(column="text", fraction=0.05, missingness="MCAR")
    df = mv_text.transform(df)
    df = NEW_apply_swapped_labels(df, fraction=0.05)
    return df


# ============================================================================
# ADULT FUNCTIONS - Numeric Corruptions for Adult Income Dataset
# ============================================================================

def ADULT_apply_all_numerical_corruptions(X, y, numeric_columns):
    """
    ADULT Batch 01: Comprehensive numeric corruption
    Combines negation, scaling, missing values, and character injection
    
    Based on Adult notebook manual corruption approach:
    - 10% negation (make values negative)
    - 10% scaling (multiply by random factor 0.5-10.0)
    - 10% missing (replace with '?')
    - 10% character injection (append random char to number)
    """
    X = X.copy()
    if isinstance(y, pd.Series):
        y = y.to_frame(name=y.name or 'target')
    
    # Combine X and y for processing
    df = pd.concat([X, y], axis=1)
    y_name = y.columns[0]
    
    # Corruption parameters
    negate_fraction = 0.10
    scale_fraction = 0.10
    missing_fraction = 0.10
    char_inject_fraction = 0.10
    chars_to_inject = ['#', '@', '!', 'x', 'a']
    
    # Apply corruptions to each numeric column
    for col in numeric_columns:
        if col not in df.columns:
            continue
            
        for idx in df.index:
            cell = df.at[idx, col]
            
            # Skip if already NaN
            if pd.isna(cell):
                continue
                
            try:
                val = float(cell)
            except:
                continue
            
            # 1. Negation (10% chance)
            if random.random() < negate_fraction:
                val = -abs(val)
            
            # 2. Scaling (10% chance)
            if random.random() < scale_fraction:
                factor = random.uniform(0.5, 10.0)
                val = val * factor
            
            # 3. Missing values (10% chance) - replaces with '?'
            if random.random() < missing_fraction:
                df.at[idx, col] = '?'
                continue
            
            # 4. Character injection (10% chance)
            if random.random() < char_inject_fraction:
                df.at[idx, col] = str(val) + random.choice(chars_to_inject)
            else:
                df.at[idx, col] = str(val)
    
    # Preserve original column order and index
    df = df[X.columns.tolist() + [y_name]]
    df.index = X.index
    
    # Split back into X and y
    X_corrupted = df.drop(columns=[y_name])
    y_corrupted = df[y_name]
    
    return X_corrupted, y_corrupted


def ADULT_apply_missing_values(X, y, numeric_columns, fraction=0.20):
    """ADULT Batch 02: Missing values only in numeric columns"""
    X = X.copy()
    if isinstance(y, pd.Series):
        y = y.to_frame(name=y.name or 'target')
    
    df = pd.concat([X, y], axis=1)
    y_name = y.columns[0]
    
    for col in numeric_columns:
        if col not in df.columns:
            continue
        
        n_corrupt = int(len(df) * fraction)
        corrupt_idx = np.random.choice(df.index, n_corrupt, replace=False)
        df.loc[corrupt_idx, col] = np.nan
    
    X_corrupted = df.drop(columns=[y_name])
    y_corrupted = df[y_name]
    
    return X_corrupted, y_corrupted


def ADULT_apply_scaling_corruption(X, y, numeric_columns, fraction=0.20):
    """ADULT Batch 03: Random scaling of numeric values"""
    X = X.copy()
    if isinstance(y, pd.Series):
        y = y.to_frame(name=y.name or 'target')
    
    df = pd.concat([X, y], axis=1)
    y_name = y.columns[0]
    
    for col in numeric_columns:
        if col not in df.columns:
            continue
        
        n_corrupt = int(len(df) * fraction)
        corrupt_idx = np.random.choice(df.index, n_corrupt, replace=False)
        
        for idx in corrupt_idx:
            if pd.notna(df.at[idx, col]):
                try:
                    val = float(df.at[idx, col])
                    factor = random.uniform(0.5, 10.0)
                    df.at[idx, col] = val * factor
                except:
                    continue
    
    X_corrupted = df.drop(columns=[y_name])
    y_corrupted = df[y_name]
    
    return X_corrupted, y_corrupted


def ADULT_apply_negation(X, y, numeric_columns, fraction=0.15):
    """ADULT Batch 04: Negate numeric values (make them negative)"""
    X = X.copy()
    if isinstance(y, pd.Series):
        y = y.to_frame(name=y.name or 'target')
    
    df = pd.concat([X, y], axis=1)
    y_name = y.columns[0]
    
    for col in numeric_columns:
        if col not in df.columns:
            continue
        
        n_corrupt = int(len(df) * fraction)
        corrupt_idx = np.random.choice(df.index, n_corrupt, replace=False)
        
        for idx in corrupt_idx:
            if pd.notna(df.at[idx, col]):
                try:
                    val = float(df.at[idx, col])
                    df.at[idx, col] = -abs(val)
                except:
                    continue
    
    X_corrupted = df.drop(columns=[y_name])
    y_corrupted = df[y_name]
    
    return X_corrupted, y_corrupted


def ADULT_apply_character_injection(X, y, numeric_columns, fraction=0.15):
    """ADULT Batch 05: Inject random characters into numeric values"""
    X = X.copy()
    if isinstance(y, pd.Series):
        y = y.to_frame(name=y.name or 'target')
    
    df = pd.concat([X, y], axis=1)
    y_name = y.columns[0]
    chars_to_inject = ['#', '@', '!', 'x', 'a']
    
    for col in numeric_columns:
        if col not in df.columns:
            continue
        
        n_corrupt = int(len(df) * fraction)
        corrupt_idx = np.random.choice(df.index, n_corrupt, replace=False)
        
        for idx in corrupt_idx:
            if pd.notna(df.at[idx, col]):
                df.at[idx, col] = str(df.at[idx, col]) + random.choice(chars_to_inject)
    
    X_corrupted = df.drop(columns=[y_name])
    y_corrupted = df[y_name]
    
    return X_corrupted, y_corrupted


def ADULT_apply_combined_missing_scaling(X, y, numeric_columns):
    """ADULT Batch 06: Combined Missing (15%) + Scaling (12%)"""
    X_corrupt, y_corrupt = ADULT_apply_missing_values(X, y, numeric_columns, 0.15)
    X_corrupt, y_corrupt = ADULT_apply_scaling_corruption(X_corrupt, y_corrupt, numeric_columns, 0.12)
    return X_corrupt, y_corrupt


def ADULT_apply_combined_negation_chars(X, y, numeric_columns):
    """ADULT Batch 07: Combined Negation (10%) + Char Injection (10%)"""
    X_corrupt, y_corrupt = ADULT_apply_negation(X, y, numeric_columns, 0.10)
    X_corrupt, y_corrupt = ADULT_apply_character_injection(X_corrupt, y_corrupt, numeric_columns, 0.10)
    return X_corrupt, y_corrupt


def ADULT_apply_heavy_missing(X, y, numeric_columns):
    """ADULT Batch 08: Heavy Missing - 30% across all numeric columns"""
    return ADULT_apply_missing_values(X, y, numeric_columns, 0.30)


def ADULT_apply_all_light_corruptions(X, y, numeric_columns):
    """ADULT Batch 09: Light Multi-Corruption - Scaling (8%) + Missing (5%) + Negation (5%) + Chars (5%)"""
    # Start with scaling
    X_corrupt, y_corrupt = ADULT_apply_scaling_corruption(X, y, numeric_columns, 0.08)
    # Add missing
    X_corrupt, y_corrupt = ADULT_apply_missing_values(X_corrupt, y_corrupt, numeric_columns, 0.05)
    # Add negation
    X_corrupt, y_corrupt = ADULT_apply_negation(X_corrupt, y_corrupt, numeric_columns, 0.05)
    # Add char injection
    X_corrupt, y_corrupt = ADULT_apply_character_injection(X_corrupt, y_corrupt, numeric_columns, 0.05)
    return X_corrupt, y_corrupt


# ============================================================================
# CATEGORICAL FUNCTIONS - For Adult Income Categorical Features (NEW!)
# ============================================================================

def ADULT_apply_category_shift(X, y, categorical_columns, fraction=0.30):
    """
    ADULT Categorical 01: Category Shift - Shifts categorical values between rows
    Uses JENGA's CategoricalShift corruption
    """
    X = X.copy()
    if isinstance(y, pd.Series):
        y = y.to_frame(name=y.name or 'target')
    
    df = pd.concat([X, y], axis=1)
    y_name = y.columns[0]
    
    for col in categorical_columns:
        if col in df.columns:
            cs = CategoricalShift(column=col, fraction=fraction)
            df = cs.transform(df)
    
    X_corrupted = df.drop(columns=[y_name])
    y_corrupted = df[y_name]
    
    return X_corrupted, y_corrupted


def ADULT_apply_category_typo(X, y, categorical_columns, fraction=0.30):
    """
    ADULT Categorical 02: Category Typo - Removes a random character from categorical values
    Example: "Married" → "Marred"
    """
    X = X.copy()
    if isinstance(y, pd.Series):
        y = y.to_frame(name=y.name or 'target')
    
    df = pd.concat([X, y], axis=1)
    y_name = y.columns[0]
    
    df_length = len(df)
    sample_size = int(df_length * fraction)
    corrupt_rows = df.sample(n=sample_size, random_state=42).index
    
    def typo(val):
        """Remove one random character from the string"""
        val = str(val)
        if len(val) <= 1:
            return val
        i = random.randrange(len(val))
        return val[:i] + val[i+1:]
    
    for col in categorical_columns:
        if col in df.columns:
            df.loc[corrupt_rows, col] = df.loc[corrupt_rows, col].apply(typo)
    
    X_corrupted = df.drop(columns=[y_name])
    y_corrupted = df[y_name]
    
    return X_corrupted, y_corrupted


def ADULT_apply_category_default(X, y, categorical_columns, fraction=0.30):
    """
    ADULT Categorical 03: Category Default - Replaces categorical values with "Default"
    Example: "Married" → "Default"
    """
    X = X.copy()
    if isinstance(y, pd.Series):
        y = y.to_frame(name=y.name or 'target')
    
    df = pd.concat([X, y], axis=1)
    y_name = y.columns[0]
    
    df_length = len(df)
    sample_size = int(df_length * fraction)
    corrupt_rows = df.sample(n=sample_size, random_state=42).index
    
    for col in categorical_columns:
        if col in df.columns:
            df.loc[corrupt_rows, col] = "Default"
    
    X_corrupted = df.drop(columns=[y_name])
    y_corrupted = df[y_name]
    
    return X_corrupted, y_corrupted


def ADULT_apply_combined_categorical(X, y, categorical_columns):
    """
    ADULT Categorical 04: Combined - Shift (15%) + Typo (10%)
    """
    X_corrupt, y_corrupt = ADULT_apply_category_shift(X, y, categorical_columns, 0.15)
    X_corrupt, y_corrupt = ADULT_apply_category_typo(X_corrupt, y_corrupt, categorical_columns, 0.10)
    return X_corrupt, y_corrupt


# ============================================================================
# CONVENIENCE MAPPINGS - Easy access to corruption functions
# ============================================================================

NEW_CORRUPTIONS = {
    "01_missing_values": NEW_apply_missing_values,
    "02_broken_characters": NEW_apply_broken_characters,
    "03_swapped_text": NEW_apply_swapped_text,
    "04_missing_labels": NEW_apply_missing_labels,
    "05_swapped_labels": NEW_apply_swapped_labels,
    "06_combined_text": NEW_apply_combined_text_corruption,
    "07_combined_both": NEW_apply_combined_text_labels,
    "08_heavy_missing": NEW_apply_heavy_missing,
    "09_all_corruptions": NEW_apply_all_corruptions,
}

ADULT_CORRUPTIONS = {
    "01_all_numerical": ADULT_apply_all_numerical_corruptions,
    "02_missing_values": ADULT_apply_missing_values,
    "03_scaling": ADULT_apply_scaling_corruption,
    "04_negation": ADULT_apply_negation,
    "05_char_injection": ADULT_apply_character_injection,
    "06_combined_missing_scaling": ADULT_apply_combined_missing_scaling,
    "07_combined_negation_chars": ADULT_apply_combined_negation_chars,
    "08_heavy_missing": ADULT_apply_heavy_missing,
    "09_all_light": ADULT_apply_all_light_corruptions,
}

ADULT_CATEGORICAL_CORRUPTIONS = {
    "10_category_shift": ADULT_apply_category_shift,
    "11_category_typo": ADULT_apply_category_typo,
    "12_category_default": ADULT_apply_category_default,
    "13_combined_categorical": ADULT_apply_combined_categorical,
}

print("✅ Unified corruption functions loaded (NEW text + ADULT numeric + ADULT categorical versions)")