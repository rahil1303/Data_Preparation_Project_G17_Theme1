    
import pandas as pd

# Step 1: Remove NaN rows
def remove_nans(df):
    df_clean = df.copy()
    df_clean = df.dropna(subset=["text", "label"])
    return df_clean

# Step 2: Convert text to string and remove empty strings
def clean_text(df):
    df_clean = df.copy()
    df_clean["text"] = df["text"].astype(str)
    df_clean = df_clean[df_clean["text"] != "nan"]  # Remove "nan" strings
    df_clean = df_clean[df_clean["text"].str.len() > 0]  # Remove empty strings
    return df_clean

# Step 3: Convert labels to numeric (invalid labels → NaN)
def convert_labels(df):
    df_clean = df.copy()
    df_clean["label"] = pd.to_numeric(df_clean["label"], errors="coerce")
    return df_clean

# Step 4: Remove rows with NaN labels
def remove_nan_labels(df):
    df_clean = df.copy()
    df_clean = df_clean.dropna(subset=["label"])
    return df_clean

# Step 5: Convert to integer and validate range [1-5]
def validate_labels(df):
    df_clean = df.copy()
    df_clean["label"] = df_clean["label"].astype(int)
    return df_clean

def clean_all(df):
    original_size = len(df)
    df_clean = df.copy()
    df_clean = remove_nans(df_clean)
    after_nan_removal = len(df_clean)
    df_clean = clean_text(df_clean)
    after_text_clean = len(df_clean)
    df_clean = convert_labels(df_clean)
    after_numeric = len(df_clean)
    df_clean = remove_nan_labels(df_clean)
    after_label_removal = len(df_clean)
    df_clean = validate_labels(df_clean)
    final_size = len(df_clean)

        # Print cleaning summary
    #print(f"\n   🧹 Data Cleaning Summary:")
    #print(f"      Original samples: {original_size}")
    #print(f"      After NaN removal: {after_nan_removal} (-{original_size - after_nan_removal})")
    #print(f"      After text cleaning: {after_text_clean} (-{after_nan_removal - after_text_clean})")
    #print(f"      After numeric conversion: {after_numeric}")
    #print(f"      After bad label removal: {after_label_removal} (-{after_numeric - after_label_removal} bad labels)")
    #print(f"      After range validation: {final_size} (-{after_label_removal - final_size})")
    #print(f"      ⚠️  Total removed: {original_size - final_size} ({(original_size - final_size) / original_size * 100:.1f}%)")
    return df_clean