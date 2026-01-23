from multiprocessing.resource_sharer import stop
from datasets import load_dataset
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from src.corruption import inject, clean
from src.models import train
from ucimlrepo import fetch_ucirepo

BASE_DIR = Path(__file__).resolve().parent
INCOME_DIR = os.path.join(BASE_DIR, "income_dataset")
CORRUPT_DIR = os.path.join(INCOME_DIR, "corrupted_batches")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

os.makedirs(INCOME_DIR, exist_ok=True)
os.makedirs(CORRUPT_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

#load dataset
adult = fetch_ucirepo(id=2)
X = adult.data.features
y = adult.data.targets.copy()

y.iloc[:, 0] = y.iloc[:, 0].str.replace('.', '', regex=False).str.strip()

selected_features = ['age', 'workclass', 'education-num', 'marital-status',
                     'occupation', 'relationship', 'race', 'sex',
                     'capital-gain', 'capital-loss', 'hours-per-week']

X = X[selected_features].copy()

numeric_features = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
categorical_features = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex']

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.06,
        max_depth=5,
        random_state=42
    ))
])

y_test = y.values.ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

baseline_acc = accuracy_score(y_test, y_pred)

print(f"Baseline Accuracy: {baseline_acc:.4f}")

df = X.copy()
df["income"] = y.values.ravel()

df_corrupt_source = df.iloc[len(X_train) + len(X_test):].copy()
if len(df_corrupt_source) < 5000:
    df_corrupt_source = df.sample(n=5000, random_state=42).reset_index(drop=True)
else:
    df_corrupt_source = df_corrupt_source.sample(n=5000, random_state=42).reset_index(drop=True)

batches_config = inject.test_functions
corrupted_batches = {}

for batch_name, corruption_fn, kwargs in batches_config:
    print(f"\n🔧 Batch {batch_name.split('_')[0]}: {batch_name.replace('_', ' ').title()}")
    df_batch = corruption_fn(df_corrupt_source, **kwargs)
    batch_path = os.path.join(CORRUPT_DIR, f"batch_{batch_name}.csv")
    df_batch.to_csv(batch_path, index=False)
    corrupted_batches[batch_name] = df_batch
    print(f"✅ {batch_name} generated")

print(f"\n✅ All 9 batches generated in {CORRUPT_DIR}")

# Store results
corruption_results = {}

for batch_name, df_batch in corrupted_batches.items():
    print(f"\n{'='*70}")
    print(f"🚨 Evaluating Corrupted Batch: {batch_name}")

    X_corrupt = df_batch[selected_features]
    y_corrupt = df_batch["income"]

    # Split corrupted data (80% train, 20% test)
    X_corrupt_train, X_corrupt_test, y_corrupt_train, y_corrupt_test = train_test_split(
        X_corrupt, y_corrupt,
        test_size=0.2,
        random_state=42,
        stratify=y_corrupt
    )
    
    print(f"   Train samples: {len(X_corrupt_train)}")
    print(f"   Test samples: {len(X_corrupt_test)}")
    
    # Train new model on corrupted data
    corrupted_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.06,
            max_depth=5,
            random_state=42
        ))
        ]
    )
    
    corrupted_model.fit(X_corrupt_train, y_corrupt_train)
    
    # Evaluate on corrupted test set
    y_pred_corrupt = corrupted_model.predict(X_corrupt_test)
    corrupt_acc = accuracy_score(y_corrupt_test, y_pred_corrupt)
    
    print(f"\n   📊 Model Trained on Corrupted Data")
    print(f"   Accuracy: {corrupt_acc:.4f}")
    
    # Store results
    corruption_results[batch_name] = {
        "accuracy": corrupt_acc,
        "train_size": len(X_corrupt_train),
        "test_size": len(X_corrupt_test),
        "predictions": y_pred_corrupt,
        "true_labels": y_corrupt_test,
        "model": corrupted_model
    }
    
    # Compare to baseline
    accuracy_drop = baseline_acc - corrupt_acc
    drop_percentage = (accuracy_drop / baseline_acc) * 100
    print(f"\n   📉 Comparison to Baseline:")
    print(f"      Baseline Accuracy: {baseline_acc:.4f}")
    print(f"      Drop: {accuracy_drop:.4f} ({drop_percentage:.2f}%)")
    
    if drop_percentage > 10:
        print(f"      ⚠️  SIGNIFICANT DROP - Corruption heavily impacts model")
    elif drop_percentage > 5:
        print(f"      ⚡ MODERATE DROP - Corruption has noticeable impact")
    else:
        print(f"      ✅ MINIMAL DROP - Model is robust to this corruption")

print(f"\n{'='*70}")
print(f"✅ All corrupted batches evaluated")
print(f"{'='*70}")
