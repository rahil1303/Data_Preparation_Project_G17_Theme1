from jenga.corruptions.generic import MissingValues, SwappedValues, CategoricalShift
from jenga.corruptions.text import BrokenCharacters
from jenga.corruptions.numerical import GaussianNoise, Scaling
import numpy as np
import pandas as pd
import random


#generic
def missing_values(df, columns = [], fraction=0.30):
    """Batch 01: Missing Values in text"""
    if columns == []:
        columns = df.columns.tolist()
    df = df.copy()
    for col in columns:
        mv = MissingValues(column=col, fraction=fraction, missingness="MCAR")
        df = mv.transform(df)
    return df

def swapped_values(df, columns=None, fraction=0.01, random_state=None):
    df = df.copy()
    rng = np.random.default_rng(random_state)
    if columns is None:
        columns = df.columns.tolist()
    for col in columns:
        # categorical / object → use jenga
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype(object)
            sv = SwappedValues(column=col, fraction=fraction)
            df = sv.transform(df)

        # numeric → swap numeric values only with random permutation
        elif pd.api.types.is_numeric_dtype(df[col]):
            idx = df[col].dropna().sample(frac=fraction, random_state=random_state).index
            df.loc[idx, col] = rng.permutation(df.loc[idx, col].values)

        else:
            raise TypeError(f"Unsupported dtype for column {col}: {df[col].dtype}")

    return df

def duplicate_rows(df, fraction=0.10):
    """Batch 03: Duplicate Rows (10%)"""
    df = df.copy()
    n_duplicates = int(len(df) * fraction)
    duplicates = df.sample(n=n_duplicates, random_state=42)
    df_with_duplicates = pd.concat([df, duplicates], ignore_index=True)
    #print("Total samples before duplication: ", len(df))
    #print("Total samples after duplication: ", len(df_with_duplicates))
    return df_with_duplicates

#category
def category_shift(df, columns=[], fraction=0.30):
    """Batch 04: Category Shift"""
    df = df.copy()
    if columns == []:
        #check if the column is a categorical column
        columns = [col for col in df.columns if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col])]
    for col in columns:
        cs = CategoricalShift(column=col, fraction=fraction)
        df = cs.transform(df)
    return df

def category_typo(df, columns = [], fraction = 0.30):
    df = df.copy()
    df_length = len(df)
    sample = int(df_length * fraction)
    rows = df.sample(n=sample, random_state=42).index

    if columns == []:
        #check if the column is a categorical column
        columns = [col for col in df.columns if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col])]
    
    def typo(val):
        val = str(val)
        if len(val) <= 1:
            return val
        i = random.randrange(len(val))
        return val[:i] + val[i+1:]

    for col in columns:
        df.loc[rows, col] = df.loc[rows, col].apply(typo)
    
    return df

def category_default(df, columns = [], fraction = 0.30):
    df = df.copy()
    df_length = len(df)
    sample = int(df_length * fraction)
    rows = df.sample(n=sample, random_state=42).index

    if columns == []:
        #check if the column is a categorical column
        columns = [col for col in df.columns if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col])]

    for col in columns:
        df.loc[rows, col] = "Default"
    
    return df

#text-specific
def broken_characters(df, columns = [], fraction=0.25):
    """Batch 05: Broken Characters in text"""
    df = df.copy()
    if columns == []:
        columns = [col for col in df.columns if pd.api.types.is_string_dtype(df[col])]
    for col in columns:
        if pd.api.types.is_string_dtype(df[col]):
            bc = BrokenCharacters(column=col, fraction=fraction)
            df = bc.transform(df)
    return df

def text_length_limit(df, columns = [], max_length=150, fraction=1.0):
    """Batch 06: Max Length"""
    df = df.copy()
    if columns == []:
        columns = [col for col in df.columns if pd.api.types.is_string_dtype(df[col])]
    #print("the average review length before: ", df["text"].str.len().mean())
    #sample a fraction of the rows to apply the length limit
    n_rows = len(df)
    n_sample = int(n_rows * fraction)
    rows = df.sample(n=n_sample, random_state=42).index

    for col in columns:
        if pd.api.types.is_string_dtype(df[col]):
            df.loc[rows, col] = (
                df.loc[rows, col]
                .astype(str)
                .str.slice(0, max_length)
            )
    #print("the average review length after: ", df["text"].str.len().mean())
    return df

#numeric
def gaussian_noise(df, columns = [], fraction=0.10):
    """Batch 07: Gaussian Noise (10%)"""
    df = df.copy()
    for col in columns:
        cur = df[col]
        numeric_mask = cur.notna()

        if numeric_mask.sum() == 0:
            continue

        temp_df = pd.DataFrame({col: cur.copy()})  

        gn = GaussianNoise(column=col, fraction=fraction)
        temp_out = gn.transform(temp_df)

        df.loc[numeric_mask, col] = temp_out.loc[numeric_mask, col]

    return df

def scaling(df, columns = [], fraction=0.10):
    """Batch 08: Scaling (10%)"""
    df = df.copy()
    if columns == []:
        columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            sc = Scaling(column=col, fraction=fraction)
            df = sc.transform(df)
    return df

def constraint_violation(df, column = "age", lower_bound = 0, upper_bound = 100, fraction=0.10):
    """Batch 09: Constraint Violations (10%)"""
    df = df.copy()
    n_rows = len(df)
    n_sample = int(n_rows * fraction)
    rows = df.sample(n=n_sample, random_state=42).index

    if pd.api.types.is_numeric_dtype(df[column]):
        #replace every number with upper_bound + 1 or lower_bound - 1 randomly
        for row in rows:
            if np.random.rand() > 0.5:
                df.at[row, column] = upper_bound + np.random.rand()*np.absolute(upper_bound)
            else:
                df.at[row, column] = lower_bound - np.random.rand()*np.absolute(lower_bound)
    return df

def numeric_to_text(df, columns = [], fraction=0.10):
    """Batch 10: Numeric to Text (10%)"""
    df = df.copy()

    n_rows = len(df)
    n_sample = int(n_rows * fraction)
    rows = df.sample(n=n_sample, random_state=42).index

    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(object)
            df.loc[rows, col] = df.loc[rows, col].astype(str)
    return df

def negative_values(df, columns = [], fraction=0.10):
    """Batch 11: Negative Values (10%)"""
    df = df.copy()
    if columns == []:
        columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

    n_rows = len(df)
    n_sample = int(n_rows * fraction)
    rows = df.sample(n=n_sample, random_state=42).index

    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df.loc[rows, col] = -np.absolute(df.loc[rows, col])
    return df

#combinations
def all_text_corruptions(df, columns = []):
    """Batch 12: All Text - Missing (5%) + Swapped (5%) + Duplicate Rows (10%) + Broken Characters (8%) + Max Length (150)"""
    df = df.copy()
    if columns == []:
        columns = [col for col in df.columns if pd.api.types.is_string_dtype(df[col])]
    
    df = missing_values(df, columns=columns, fraction=0.05)
    df = swapped_values(df, columns=columns, fraction=0.05)
    df = duplicate_rows(df, fraction=0.10)

    df = broken_characters(df, columns=columns, fraction=0.08)
    df = text_length_limit(df, columns=columns, max_length=150)
    return df

def all_numerical_corruptions(df, columns = []):
    """Batch 13: All Numeric - Missing (5%) + Swapped (5%) + Duplicate Rows (10%) + Gaussian Noise (10%) + Scaling (10%)"""
    df = df.copy()
    if columns == []:
        columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

    df = swapped_values(df, columns=columns, fraction=0.5)
    df = missing_values(df, columns=columns, fraction=0.3)
    df = duplicate_rows(df, fraction=0.1)

    df = gaussian_noise(df, columns=columns, fraction=0.25)
    df = scaling(df, columns=columns, fraction=0.25)
    df = constraint_violation(df, column="age", lower_bound=0, upper_bound=100, fraction=0.25)
    df = numeric_to_text(df, columns=columns, fraction=0.25)
    df = negative_values(df, columns=columns, fraction=0.25)
    return df

def all_corruptions(df, text_columns = [], numerical_columns = []):
    """Batch 14: All Corruptions"""
    df = df.copy()
    df = all_text_corruptions(df, columns=text_columns)
    df = all_numerical_corruptions(df, columns=numerical_columns)
    df = category_shift(df, columns=[], fraction=0.10)
    return df

corruption_functions = [
    ("01_missing_values", missing_values, {}),
    ("02_swapped_values", swapped_values, {}),
    ("03_duplicate_rows", duplicate_rows, {}),
    ("04_category_shift", category_shift, {}),
    ("05_broken_characters", broken_characters, {}),
    ("06_text_length_limit", text_length_limit, {}),
    ("07_gaussian_noise", gaussian_noise, {}),
    ("08_scaling", scaling, {}),
    ("09_constraint_violation", constraint_violation, {}),
    ("10_numeric_to_text", numeric_to_text, {}),
    ("11_negative_values", negative_values, {}),
    ("12_all_text_corruptions", all_text_corruptions, {}),
    ("13_all_numerical_corruptions", all_numerical_corruptions, {}),
    ("14_all_corruptions", all_corruptions, {}),
]

test_functions = [
    ("CATEGORY_SHIFT", category_shift, {}),
    ("CATEGORY_TYPO", category_typo, {}),
    ("CATEGORY_DEFAULT", category_default, {}),
]
print("✅ Corruption functions defined")