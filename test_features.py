
from ml_model.feature_extractor import extract_features


sample_code = """
class Calculator:
    def calculate(self, value):
        total = 10 * 5
        for i in range(value):
            total = total + i
        if total > 100:
            return total
        return 0
"""

features = extract_features(sample_code)

print("Extracted Python Code Features")
print("-" * 35)

for feature, value in features.items():
    print(f"{feature}: {value}")

assert features["class_count"] == 1
assert features["function_count"] == 1
assert features["loop_count"] == 1
assert features["conditional_count"] == 1
assert features["arithmetic_operations"] == 2
assert features["constant_operations"] == 1
assert features["function_calls"] == 1

print("\nAll feature extraction tests passed!")
