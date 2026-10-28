
from optimizer.optimizer import optimize_code


TEST_CASES = [
    ("x + 0", "x"),
    ("x - 0", "x"),
    ("x * 1", "x"),
    ("1 * x", "x"),
    ("x / 1", "x"),
    ("(10 * 20) + 5", "205"),
]


def main():
    print("=" * 60)
    print("UML2Optimize - Algebraic Simplification Tests")
    print("=" * 60)

    passed = 0

    for expression, expected in TEST_CASES:
        source = f"def calculate(x):\n    return {expression}\n"
        optimized = optimize_code(source)

        print(f"\nOriginal:  {expression}")
        print(f"Optimized code:\n{optimized}")

        # Check the result for several values.
        namespace = {}
        exec(compile(optimized, "<optimized>", "exec"), namespace)
        calculate = namespace["calculate"]

        values = [0, 1, 5, 10]

        # For the constant-only expression, x is not used.
        try:
            results = [calculate(value) for value in values]

            if expression == "(10 * 20) + 5":
                correct = all(result == 205 for result in results)
            else:
                correct = all(
                    result == eval(expected, {"x": value})
                    for value, result in zip(values, results)
                )

        except Exception as error:
            print(f"FAIL: {error}")
            continue

        if correct:
            print("Result: PASS")
            passed += 1
        else:
            print("Result: FAIL")

    print("\n" + "=" * 60)
    print(f"Passed: {passed}/{len(TEST_CASES)}")
    print("=" * 60)


if __name__ == "__main__":
    main()