from jenga.corruptions.generic import MissingValues, SwappedValues
from jenga.corruptions.text import BrokenCharacters
import pandas as pd

def apply_missing_values(df, fraction=0.30):
    """Batch 01: Missing Values in text"""
    df = df.copy()
    mv = MissingValues(column="text", fraction=fraction, missingness="MCAR")
    return mv.transform(df)

def apply_broken_characters(df, fraction=0.25):
    """Batch 02: Broken Characters in text"""
    df = df.copy()
    bc = BrokenCharacters(column="text", fraction=fraction)
    return bc.transform(df)

def apply_swapped_text(df, fraction=0.20):
    """Batch 03: Swapped text values"""
    df = df.copy()
    sv = SwappedValues(column="text", fraction=fraction)
    return sv.transform(df)

def apply_missing_labels(df, fraction=0.15):
    """Batch 04: Missing Labels"""
    df = df.copy()
    mv = MissingValues(column="label", fraction=fraction, missingness="MCAR")
    return mv.transform(df)

def apply_swapped_labels(df, fraction=0.12):
    """Batch 05: Swapped Labels"""
    df = df.copy()
    sv = SwappedValues(column="label", fraction=fraction)
    return sv.transform(df)

def apply_combined_text_corruption(df):
    """Batch 06: Broken Chars (10%) + Missing (8%)"""
    df = df.copy()
    bc = BrokenCharacters(column="text", fraction=0.10)
    df = bc.transform(df)
    mv = MissingValues(column="text", fraction=0.08, missingness="MCAR")
    return mv.transform(df)

def apply_combined_text_labels(df):
    """Batch 07: Swapped Text (15%) + Swapped Labels (8%)"""
    df = df.copy()
    sv_text = SwappedValues(column="text", fraction=0.15)
    df = sv_text.transform(df)
    sv_label = SwappedValues(column="label", fraction=0.08)
    return sv_label.transform(df)

def apply_heavy_missing(df):
    """Batch 08: Heavy Missing - Text (25%) + Labels (10%)"""
    df = df.copy()
    mv_text = MissingValues(column="text", fraction=0.25, missingness="MCAR")
    df = mv_text.transform(df)
    mv_label = MissingValues(column="label", fraction=0.10, missingness="MCAR")
    return mv_label.transform(df)

#corruption that cuts of the text after a specified character length 
def apply_max_length_text(df, max_length=150):
    """Batch 09: Max Length"""
    df = df.copy()
    #print("the average review length before: ", df["text"].str.len().mean())
    df["text"] = df["text"].astype(str).str.slice(0, max_length)
    #print("the average review length after: ", df["text"].str.len().mean())
    return df

def apply_duplicate_rows(df, fraction=0.10):
    """Batch 10: Duplicate Rows (10%)"""
    df = df.copy()
    n_duplicates = int(len(df) * fraction)
    duplicates = df.sample(n=n_duplicates, random_state=42)
    df_with_duplicates = pd.concat([df, duplicates], ignore_index=True)
    #print("Total samples before duplication: ", len(df))
    #print("Total samples after duplication: ", len(df_with_duplicates))
    return df_with_duplicates

def apply_all_corruptions(df):
    """Batch 11: All - Broken (8%) + Swapped (10%) + Missing (5%)"""
    df = df.copy()
    bc = BrokenCharacters(column="text", fraction=0.08)
    df = bc.transform(df)
    sv_text = SwappedValues(column="text", fraction=0.10)
    df = sv_text.transform(df)
    mv_text = MissingValues(column="text", fraction=0.05, missingness="MCAR")
    df = mv_text.transform(df)
    sv_label = SwappedValues(column="label", fraction=0.05)
    return sv_label.transform(df)

corruption_functions = [
    ("01_missing_text", apply_missing_values, {}),
    ("02_broken_characters", apply_broken_characters, {}),
    ("03_swapped_text", apply_swapped_text, {}),
    ("04_missing_labels", apply_missing_labels, {}),
    ("05_swapped_labels", apply_swapped_labels, {}),
    ("06_combined_text_corruption", apply_combined_text_corruption, {}),
    ("07_combined_text_labels", apply_combined_text_labels, {}),
    ("08_heavy_missing", apply_heavy_missing, {}),
    ("09_max_length", apply_max_length_text, {}),
    ("10_duplicate_rows", apply_duplicate_rows, {}),
    ("11_all_corruptions", apply_all_corruptions, {}),
]

test_functions = [
    ("09_max_length", apply_max_length_text, {}),
    ("10_duplicate_rows", apply_duplicate_rows, {}),
]
print("✅ Corruption functions defined")