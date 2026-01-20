from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import os
from src.corruption import inject, clean
from src.models import train

#load dataset
#print("Loading Amazon Reviews 2023 (All Beauty)...")

BASE_DIR =  Path.cwd().parent / "datasets"
AMAZON_DIR = os.path.join(BASE_DIR, "amazon_reviews_2023_all_beauty")
os.makedirs(AMAZON_DIR, exist_ok=True)

try:
    dataset = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_review_All_Beauty",
        split="full",
        streaming=False
    )
    
    df = dataset.to_pandas()
    
except Exception as e:
    print(f"Error loading from HF: {e}")


#print(f"✅ Loaded {len(df)} reviews")
#print(f"Columns: {df.columns.tolist()}")
#print(df.head())

# Sample 50k for faster local processing
df_sample = df.sample(n=min(50000, len(df)), random_state=42)
sample_path = os.path.join(AMAZON_DIR, "sample_50k.csv")
df_sample.to_csv(sample_path, index=False)
#print(f"✅ Saved sample to: {sample_path}")

CORRUPT_DIR = os.path.join(AMAZON_DIR, "corrupted_batches")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

os.makedirs(AMAZON_DIR, exist_ok=True)
os.makedirs(CORRUPT_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

dataset = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023",
    "raw_review_All_Beauty",
    split="full",
    streaming=False
)
df = dataset.to_pandas()
#print(f"✅ Loaded {len(df)} reviews")

# Sample 15k for faster processing (8k train + 2k test + 5k corrupt)
df = df.sample(n=min(15000, len(df)), random_state=42).reset_index(drop=True)

if "text" not in df.columns:
    df["text"] = (df.get("title", "").fillna("").astype(str) + " " + 
                  df.get("body", "").fillna("").astype(str)).str.strip()

# Clean data
df = df.dropna(subset=["rating", "text"])
df = df[df["text"].str.len() > 0]

df["label"] = df["rating"].astype(int)

#print(f"\n📊 Data Summary:")
#print(f"   Total samples: {len(df)}")
#print(f"   Label distribution: {df['label'].value_counts().to_dict()}")

X = df["text"].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,  
    random_state=42,
    stratify=y
)

#print(f"\n✅ Train/Test Split:")
#print(f"   Train: {len(X_train)} samples")
#print(f"   Test: {len(X_test)} samples")

#print("\n🔧 Training baseline model...")
baseline_model = train.build_model()
baseline_model.fit(X_train, y_train)

# Evaluate baseline
y_pred_baseline = baseline_model.predict(X_test)
baseline_acc = accuracy_score(y_test, y_pred_baseline)

#print(f"\n✅ Baseline Model Trained")
#print(f"📊 Accuracy: {baseline_acc:.4f}")
#print(f"\nClassification Report:")
#print(classification_report(y_test, y_pred_baseline))

# Save baseline accuracy
baseline_metrics = {
    "accuracy": baseline_acc,
    "predictions": y_pred_baseline
}

df_corrupt_source = df.iloc[len(X_train) + len(X_test):][["text", "label"]].copy()
if len(df_corrupt_source) < 5000:
    df_corrupt_source = df.sample(n=5000, random_state=42).reset_index(drop=True)
else:
    df_corrupt_source = df_corrupt_source.sample(n=5000, random_state=42).reset_index(drop=True)

batches_config = inject.corruption_functions

corrupted_batches = {}

for batch_name, corruption_fn, kwargs in batches_config:
    #print(f"\n🔧 Batch {batch_name.split('_')[0]}: {batch_name.replace('_', ' ').title()}")
    df_batch = corruption_fn(df_corrupt_source, **kwargs)
    batch_path = os.path.join(CORRUPT_DIR, f"batch_{batch_name}.csv")
    df_batch.to_csv(batch_path, index=False)
    corrupted_batches[batch_name] = df_batch
    #print(f"✅ {batch_name} generated")

corruption_results = {}
cleaning_results = {}

for batch_name, df_batch in corrupted_batches.items():
    
    #X_corrupt = df_batch["text"].values
    #y_corrupt = df_batch["label"].values

    df_batch_clean = clean.clean_all(df_batch)

    X_clean = df_batch_clean["text"].values
    y_clean = df_batch_clean["label"].values


    # Split corrupted data (80% train, 20% test)
    #X_corrupt_train, X_corrupt_test, y_corrupt_train, y_corrupt_test = train_test_split(
    #    X_corrupt, y_corrupt,
    #    test_size=0.2,
    #    random_state=42,
    #    stratify=y_corrupt
    #)
    #clean data
    X_clean_train, X_clean_test, y_clean_train, y_clean_test = train_test_split(
        X_clean, y_clean,
        test_size=0.2,
        random_state=42,
        stratify=y_clean
    )
    
    print(f"   corrupt Train samples: {len(X_corrupt_train)}")
    print(f"   corrut Test samples: {len(X_corrupt_test)}")
    
    # Train new model on corrupted data
    #corrupted_model = train.build_model()
    #corrupted_model.fit(X_corrupt_train, y_corrupt_train)

    clean_model = train.build_model()
    clean_model.fit(X_clean_train, y_clean_train)
    
    # Evaluate on corrupted test set
    #y_pred_corrupt = corrupted_model.predict(X_corrupt_test)
    #corrupt_acc = accuracy_score(y_corrupt_test, y_pred_corrupt)

    #evaluate on clean test set
    y_pred_clean = clean_model.predict(X_clean_test)
    clean_acc = accuracy_score(y_clean_test, y_pred_clean)

    print(f"\n   📊 Model Trained on Cleaned Data: {batch_name}")
    print(f"   Accuracy: {clean_acc:.4f}")
    print(f"\n   📊 Baseline:")
    print(f"   Accuracy: {baseline_acc:.4f}")

    # Store results
    #corruption_results[batch_name] = {
    #    "accuracy": corrupt_acc,
    #    "train_size": len(X_corrupt_train),
    #    "test_size": len(X_corrupt_test),
    #    "predictions": y_pred_corrupt,
    #    "true_labels": y_corrupt_test,
    #    "model": corrupted_model
    #}
    cleaning_results[batch_name] = {
        "accuracy": clean_acc,
        "train_size": len(X_clean_train),
        "test_size": len(X_clean_test),
        "predictions": y_pred_clean,
        "true_labels": y_clean_test,
        "model": clean_model
    }

    # Compare cleaned to baseline
    accuracy_drop = baseline_acc - clean_acc
    drop_percentage = (accuracy_drop / baseline_acc) * 100
    print(f"\n   📉 Corrupt Comparison to Baseline:")
    print(f"      Baseline Accuracy: {baseline_acc:.4f}")
    print(f"      Drop: {accuracy_drop:.4f} ({drop_percentage:.2f}%)")
    
    if drop_percentage > 10:
        print(f"      ⚠️  SIGNIFICANT DROP - Corruption heavily impacts model")
    elif drop_percentage > 5:
        print(f"      ⚡ MODERATE DROP - Corruption has noticeable impact")
    else:
        print(f"      ✅ MINIMAL DROP - Model is robust to this corruption")

#print(f"\n✅ All 9 batches generated in {CORRUPT_DIR}")