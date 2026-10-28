
import sys
from pathlib import Path

import joblib
import pandas as pd

# Find the main UML2Optimize project folder.
PROJECT_DIR = Path(__file__).resolve().parents[1]

# Add the project folder to Python's import path.
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ml_model.feature_extractor import extract_features


# Path to the trained model using measured benchmark data.
MODEL_PATH = (
    PROJECT_DIR
    / "ml_model"
    / "measured_optimizer_model.joblib"
)


def load_model():
    """Load the trained machine learning model."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}\n"
            "Run this command first:\n"
            "python ml_model/train_measured_model.py"
        )

    return joblib.load(MODEL_PATH)


def predict_optimization(source_code):
    """
    Predict whether optimization may be beneficial.

    Returns the prediction, confidence when available,
    and extracted source-code features.
    """
    features = extract_features(source_code)
    model = load_model()

    # Arrange features in the same order used during training.
    feature_names = [
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

    input_data = pd.DataFrame(
        [[features[name] for name in feature_names]],
        columns=feature_names,
    )

    prediction = int(model.predict(input_data)[0])

    # Decision trees can provide class probabilities.
    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        confidence = float(max(probabilities))

    return {
        "prediction": prediction,
        "confidence": confidence,
        "features": features,
        "interpretation": (
            "Optimization may be beneficial."
            if prediction == 1
            else "Optimization may not be beneficial."
        ),
    }


def main():
    print("=" * 55)
    print("UML2Optimize - ML Optimization Predictor")
    print("=" * 55)

    print(f"Model: {MODEL_PATH.name}")

    # Example program to test the predictor.
    sample_code = """
def calculate():
    a = 10
    b = 20
    result = a * b + 5
    return result
"""

    try:
        result = predict_optimization(sample_code)

        print("\nPrediction:")
        print(result["interpretation"])

        if result["confidence"] is not None:
            print(
                "Model confidence: "
                f"{result['confidence']:.2%}"
            )

        print("\nExtracted features:")

        for name, value in result["features"].items():
            print(f"  {name}: {value}")

        print("\nNote:")
        print(
            "Confidence is the model's estimated class probability, "
            "not a guarantee that optimization will improve speed."
        )

    except (ValueError, FileNotFoundError) as error:
        print(f"\nError: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
