
from pathlib import Path

import joblib
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


MODEL_DIR = Path(__file__).resolve().parent
DATA_FILE = MODEL_DIR / "benchmark_data.csv"
MODEL_FILE = MODEL_DIR / "measured_optimizer_model.joblib"

FEATURE_NAMES = [
    "source_lines",
    "source_characters",
    "function_count",
    "class_count",
    "loop_count",
    "conditional_count",
    "arithmetic_operations",
    "constant_operations",
    "function_calls",
]


def main():
    if not DATA_FILE.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Run python ml_model/collect_data.py first."
        )

    df = pd.read_csv(DATA_FILE)

    required = FEATURE_NAMES + ["beneficial"]
    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if df[required].isnull().any().any():
        raise ValueError("Features or labels contain missing values.")

    # Remove inconclusive measurements (label -1).
    df = df[df["beneficial"].isin([0, 1])].copy()

    print("=" * 55)
    print("UML2Optimize - Measured ML Model Evaluation")
    print("=" * 55)
    print(f"All collected programs: 100")
    print(f"Conclusive programs: {len(df)}")
    print("\nConclusive label counts:")
    print(df["beneficial"].value_counts().sort_index())

    if len(df) < 20 or df["beneficial"].nunique() < 2:
        raise ValueError(
            "Not enough conclusive examples from both classes. "
            "Collect more benchmark data before training."
        )

    X = df[FEATURE_NAMES]
    y = df["beneficial"].astype(int)

    # Ensure each class has enough examples for a stratified split.
    if y.value_counts().min() < 2:
        raise ValueError(
            "Each class needs at least two examples for this split."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)

    model = DecisionTreeClassifier(
        max_depth=2,
        min_samples_leaf=3,
        random_state=42,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"\nTraining examples: {len(X_train)}")
    print(f"Test examples: {len(X_test)}")

    print("\nBaseline accuracy:")
    print(f"{accuracy_score(y_test, baseline_predictions):.2%}")

    print("\nDecision-tree accuracy:")
    print(f"{accuracy_score(y_test, predictions):.2%}")

    print("\nDecision-tree balanced accuracy:")
    print(f"{balanced_accuracy_score(y_test, predictions):.2%}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Not beneficial", "Beneficial"],
            zero_division=0,
        )
    )

    joblib.dump(model, MODEL_FILE)
    print(f"\nModel saved to: {MODEL_FILE}")

    print(
        "\nResearch limitation: The dataset contains a narrow family "
        "of generated programs. The test set is small, and the results "
        "do not establish generalization to different program types."
    )


if __name__ == "__main__":
    main()
