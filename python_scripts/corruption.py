from jenga.corruptions.generic import MissingValues, SwappedValues, CategoricalShift
from jenga.corruptions.text import BrokenCharacters
import pandas as pd
import numpy as np
import random


#numeric functions

def apply_all_numerical_corruptions(X, y, numeric_columns):
    X = X.copy()
    if isinstance(y, pd.Series):
        y = y.to_frame(name=y.name or 'target')
    
    #combine X and y for processing
    df = pd.concat([X, y], axis=1)
    y_name = y.columns[0]
    
    #corruption parameters
    negate_fraction = 0.10
    scale_fraction = 0.10
    missing_fraction = 0.10
    char_inject_fraction = 0.10
    chars_to_inject = ['#', '@', '!', 'x', 'a']
    
    #apply corruptions to each numeric column
    for col in numeric_columns:
        if col not in df.columns:
            continue
            
        for idx in df.index:
            cell = df.at[idx, col]
            
            if pd.isna(cell):
                continue
                
            try:
                val = float(cell)
            except:
                continue
            
            #negation
            if random.random() < negate_fraction:
                val = -abs(val)
            
            #scaling
            if random.random() < scale_fraction:
                factor = random.uniform(0.5, 10.0)
                val = val * factor
            
            #missing value
            if random.random() < missing_fraction:
                df.at[idx, col] = '?'
                continue
            
            #char injection
            if random.random() < char_inject_fraction:
                df.at[idx, col] = str(val) + random.choice(chars_to_inject)
            else:
                df.at[idx, col] = str(val)
    
    df = df[X.columns.tolist() + [y_name]]
    df.index = X.index
    
    X_corrupted = df.drop(columns=[y_name])
    y_corrupted = df[y_name]
    
    return X_corrupted, y_corrupted


def apply_missing_values(X, y, numeric_columns, fraction=0.20):
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


def apply_scaling_corruption(X, y, numeric_columns, fraction=0.20):
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


def apply_negation(X, y, numeric_columns, fraction=0.15):
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


def apply_character_injection(X, y, numeric_columns, fraction=0.15):
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


def apply_combined_missing_scaling(X, y, numeric_columns):
    X_corrupt, y_corrupt = apply_missing_values(X, y, numeric_columns, 0.15)
    X_corrupt, y_corrupt = apply_scaling_corruption(X_corrupt, y_corrupt, numeric_columns, 0.12)
    return X_corrupt, y_corrupt


def apply_combined_negation_chars(X, y, numeric_columns):
    X_corrupt, y_corrupt = apply_negation(X, y, numeric_columns, 0.10)
    X_corrupt, y_corrupt = apply_character_injection(X_corrupt, y_corrupt, numeric_columns, 0.10)
    return X_corrupt, y_corrupt


def apply_heavy_missing(X, y, numeric_columns):
    return apply_missing_values(X, y, numeric_columns, 0.30)


def apply_all_light_corruptions(X, y, numeric_columns):
    X_corrupt, y_corrupt = apply_scaling_corruption(X, y, numeric_columns, 0.08)
    X_corrupt, y_corrupt = apply_missing_values(X_corrupt, y_corrupt, numeric_columns, 0.05)
    X_corrupt, y_corrupt = apply_negation(X_corrupt, y_corrupt, numeric_columns, 0.05)
    X_corrupt, y_corrupt = apply_character_injection(X_corrupt, y_corrupt, numeric_columns, 0.05)
    return X_corrupt, y_corrupt


#categorical functions

def apply_category_shift(X, y, categorical_columns, fraction=0.30):
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


def apply_category_typo(X, y, categorical_columns, fraction=0.30):
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


def apply_category_default(X, y, categorical_columns, fraction=0.30):
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


def apply_combined_categorical(X, y, categorical_columns):
    """
    ADULT Categorical 04: Combined - Shift (15%) + Typo (10%)
    """
    X_corrupt, y_corrupt = apply_category_shift(X, y, categorical_columns, 0.15)
    X_corrupt, y_corrupt = apply_category_typo(X_corrupt, y_corrupt, categorical_columns, 0.10)
    return X_corrupt, y_corrupt

NUMERICAL_CORRUPTIONS = {
    "01_all_numerical": apply_all_numerical_corruptions,
    "02_missing_values": apply_missing_values,
    "03_scaling": apply_scaling_corruption,
    "04_negation": apply_negation,
    "05_char_injection": apply_character_injection,
    "06_combined_missing_scaling": apply_combined_missing_scaling,
    "07_combined_negation_chars": apply_combined_negation_chars,
    "08_heavy_missing": apply_heavy_missing,
    "09_all_light": apply_all_light_corruptions,
}

CATEGORICAL_CORRUPTIONS = {
    "10_category_shift": apply_category_shift,
    "11_category_typo": apply_category_typo,
    "12_category_default": apply_category_default,
    "13_combined_categorical": apply_combined_categorical,
}