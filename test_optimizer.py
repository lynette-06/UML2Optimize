

from pathlib import Path

from optimizer.optimizer import optimize_code

PROJECT_DIR = Path(__file__).resolve().parent


def test_generated_files():
    """Check that both generated files contain valid, runnable Python."""
    for filename in ("generated_code.py", "optimized_code.py"):
        file_path = PROJECT_DIR / filename
        source = file_path.read_text(encoding="utf-8")

        # Check Python syntax.
        compile(source, filename, "exec")

        # Run the generated code in an isolated namespace.
        namespace = {}
        exec(compile(source, filename, "exec"), namespace)

        # Check that the generated classes can be instantiated.
        student = namespace["Student"]()
        course = namespace["Course"]()

        assert student.student_id is None
        assert student.name is None
        assert course.course_id is None
        assert course.course_name is None

        print(f"PASS: {filename} is valid and its classes work.")


def test_optimization_preserves_results():
    """Check that optimization doesn't change program results."""
    original_code = """
def calculate(a, b):
    total = 10 * 5
    result = (a + b) * 2
    return total + result
"""

    optimized_code = optimize_code(original_code)

    original_namespace = {}
    optimized_namespace = {}

    exec(compile(original_code, "<original>", "exec"), original_namespace)
    exec(compile(optimized_code, "<optimized>", "exec"), optimized_namespace)

    test_inputs = [(1, 2), (5, 7), (0, 0), (-3, 4)]

    for a, b in test_inputs:
        original_result = original_namespace["calculate"](a, b)
        optimized_result = optimized_namespace["calculate"](a, b)

        assert original_result == optimized_result, (
            f"Mismatch for inputs ({a}, {b})"
        )

    print("PASS: Original and optimized functions return identical results.")


if __name__ == "__main__":
    test_generated_files()
    test_optimization_preserves_results()
    print("\nAll correctness tests passed!")
