from jenga.corruptions.generic import PermutationCorruption
from jenga.corruptions.noise import NoiseCorruption
from jenga.corruptions.swap import SwapCorruption
import pandas as pd
from sklearn.metrics import accuracy_score



def jenga_numeric_corruption(
    model,
    X_test,
    y_test,
    column,
    corruption="permutation",
    severity=1.0,
    random_state=42
):
    """
    REAL JENGA numeric corruptions.

    corruption: 'permutation', 'noise', 'swap'
    severity: corruption strength (used for noise / swap)
    """

    clean_acc = accuracy_score(y_test, model.predict(X_test))

    if corruption == "permutation":
        corruptor = PermutationCorruption(
            features=[column],
            random_state=random_state
        )

    elif corruption == "noise":
        corruptor = NoiseCorruption(
            features=[column],
            scale=severity,
            random_state=random_state
        )

    elif corruption == "swap":
        corruptor = SwapCorruption(
            features=[column],
            fraction=severity,
            random_state=random_state
        )

    else:
        raise ValueError("Unknown corruption type")

    X_corrupt = corruptor.transform(X_test)

    corrupt_acc = accuracy_score(y_test, model.predict(X_corrupt))

    return {
        "feature": column,
        "corruption": corruption,
        "severity": severity,
        "clean_accuracy": clean_acc,
        "corrupted_accuracy": corrupt_acc,
        "accuracy_drop": clean_acc - corrupt_acc
    }

def run_jenga_numeric_suite(
    model,
    X_test,
    y_test,
    numeric_features,
    corruption="permutation",
    severity=1.0
):
    results = []

    for col in numeric_features:
        res = jenga_numeric_corruption(
            model=model,
            X_test=X_test,
            y_test=y_test,
            column=col,
            corruption=corruption,
            severity=severity
        )
        results.append(res)

    return pd.DataFrame(results).sort_values(
        "accuracy_drop", ascending=False
    )