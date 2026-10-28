
from pathlib import Path

from compiler.parser import parse_model
from compiler.code_generator import generate_code
from optimizer.optimizer import optimize_code
from ml_model.predict import predict_optimization


def main():
    project_dir = Path(__file__).resolve().parent

    input_file = project_dir / "input" / "student_model.json"
    original_file = project_dir / "generated_code.py"
    optimized_file = project_dir / "optimized_code.py"

    print("=" * 50)
    print("          UML2Optimize Compiler")
    print("=" * 50)

    try:
        # Step 1: Parse the UML-derived JSON model.
        print("\n[1/5] Reading UML model...")
        model = parse_model(input_file)
        print("UML model loaded successfully.")

        # Step 2: Generate Python source code.
        print("\n[2/5] Generating Python code...")
        original_code = generate_code(model)
        original_file.write_text(original_code, encoding="utf-8")
        print(f"Original code saved: {original_file.name}")

        # Step 3: Extract features and predict optimization benefit.
        print("\n[3/5] Running ML prediction...")
        prediction_result = predict_optimization(original_code)

        prediction = prediction_result["prediction"]
        confidence = prediction_result["confidence"]

        print(
            "Prediction:",
            "Optimization may be beneficial"
            if prediction == 1
            else "Optimization may not be beneficial",
        )
        print(
            f"Model confidence: {confidence:.2%} "
            "(prototype model; not validated)"
        )

        print("\nExtracted features:")
        for name, value in prediction_result["features"].items():
            print(f"  {name}: {value}")

        # Step 4: Run the optimizer regardless of the prediction.
        # Keep the prediction advisory until validated with real data.
        print("\n[4/5] Optimizing generated code...")
        optimized_code = optimize_code(original_code)
        optimized_file.write_text(optimized_code, encoding="utf-8")
        print(f"Optimized code saved: {optimized_file.name}")

        # Step 5: Verify syntax and report completion.
        print("\n[5/5] Checking generated Python syntax...")
        compile(original_code, str(original_file), "exec")
        compile(optimized_code, str(optimized_file), "exec")
        print("Both generated files have valid Python syntax.")

        print("\n--- ORIGINAL CODE ---")
        print(original_code)

        print("\n--- OPTIMIZED CODE ---")
        print(optimized_code)

        print("\nUML2Optimize completed successfully!")

    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"\nCompilation failed: {error}")


if __name__ == "__main__":
    main()
