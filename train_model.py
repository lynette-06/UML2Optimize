
from pathlib import Path

import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

# PROTOTYPE DATA ONLY:
# These illustrative labels are not measured benchmark results.
# Replace them with labels derived from repeated real benchmarks.
data = [
    [10, 200, 1, 1, 1, 0, 5, 4, 1, 1],
    [12, 250, 1, 1, 2, 0, 8, 6, 2, 1],
    [15, 300, 2, 1, 2, 1, 10, 8, 3, 1],
    [8, 160, 1, 0, 1, 0, 4, 3, 1, 1],
    [20, 420, 2, 1, 3, 2, 12, 10, 4, 1],
    [7, 130, 1, 0, 0, 0, 1, 1, 0, 1],
    [14, 280, 2, 1, 1, 2, 6, 5, 2, 1],
    [9, 180, 1, 1, 0, 1, 3, 2, 2, 1],
    [10, 200, 1, 1, 1, 0, 5, 0, 1, 0],
    [12, 240, 1, 1, 2, 0, 7, 0, 2, 0],
    [15, 310, 2, 1, 2, 1, 9, 0, 3, 0],
    [8, 150, 1, 0, 1, 0, 4, 0, 1, 0],
    [18, 370, 2, 1, 3, 2, 11, 0, 4, 0],
    [6, 120, 1, 0, 0, 0, 1, 0, 0, 0],
    [13, 260, 2, 1, 1, 2, 6, 0, 2, 0],
    [9, 170, 1, 1, 0, 1, 3, 0, 2, 0],
    [16, 330, 2, 1, 2, 1, 8, 5, 3, 1],
    [11, 220, 1, 1, 1, 1, 6, 4, 2, 1],
    [17, 350, 2, 1, 2, 2, 9, 7, 3, 1],
    [8, 155, 1, 0, 1, 1, 4, 0, 1, 0],
]

df = pd.DataFrame(data, columns=FEATURE_NAMES + ["beneficial"])

X = df[FEATURE_NAMES]
y = df["beneficial"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

output_path = Path(__file__).resolve().parent / "optimizer_model.joblib"
joblib.dump(model, output_path)

print("ML optimizer model trained successfully!")
print(f"Training examples: {len(X_train)}")
print(f"Test examples: {len(X_test)}")
print(f"Prototype test accuracy: {accuracy:.2%}")
print(f"Model saved to: {output_path.name}")
print("\nWARNING: Accuracy is not meaningful evidence of real")
print("performance because the labels are illustrative.")
