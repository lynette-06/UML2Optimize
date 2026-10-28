
from optimizer.optimizer import optimize_code


TEST_CASES = [
    {
        "name": "Multiplication and addition",
        "expression": "(10 * 20) + 5",
        "expected": 205,
    },
    {
        "name": "Operator precedence",
        "expression": "8 + 4 * 3",
        "expected": 20,
    },
    {
        "name": "Subtraction",
        "expression": "100 - 25",
        "expected": 75,
    },
    {
        "name": "Division",
        "expression": "7 / 2",
        "expected": 3.5,
    },
    {
        "name": "Negative result",
        "expression": "5 - 12",
        "expected": -7,
    },
    {
        "name": "Division by zero",
        "expression": "10 / 0",
        "expected": None,
    },
]


def build_program(expression):
    """Build a Python program containing one arithmetic expression."""
    return (
        "def calculate():\n"
        f"    result = {expression}\n"
        "    return result\n"
    )


def run_program(source_code):
    """Compile and execute a test program."""
    namespace = {}
    compiled = compile(source_code, "<constant-folding-demo>", "exec")
    exec(compiled, namespace)
    return namespace["calculate"]()


def main():
    print("=" * 68)
    print("UML2Optimize - Extended Constant Folding Demonstration")
    print("=" * 68)

    passed = 0
    failed = 0

    for index, test in enumerate(TEST_CASES, start=1):
        original_code = build_program(test["expression"])
        optimized_code = optimize_code(original_code)

        print(f"\nTest {index}: {test['name']}")
        print(f"Original expression:  {test['expression']}")

        # Show the optimized assignment.
        optimized_assignment = next(
            (
                line.strip()
                for line in optimized_code.splitlines()
                if "result =" in line
            ),
            "(assignment not found)",
        )
        print(f"Optimized expression: {optimized_assignment}")

        try:
            original_result = run_program(original_code)
        except ArithmeticError as error:
            original_result = None
            original_error = type(error).__name__
        else:
            original_error = None

        try:
            optimized_result = run_program(optimized_code)
        except ArithmeticError as error:
            optimized_result = None
            optimized_error = type(error).__name__
        else:
            optimized_error = None

        # Compare both values and exception behavior.
        same_behavior = (
            original_error == optimized_error
            and (
                original_error is not None
                or original_result == optimized_result
            )
        )

        print(f"Original result:  {original_result}")
        print(f"Optimized result: {optimized_result}")

        if original_error is not None:
            print(f"Original error:   {original_error}")

        if optimized_error is not None:
            print(f"Optimized error:  {optimized_error}")

        if same_behavior:
            print("Correctness: PASS")
            passed += 1
        else:
            print("Correctness: FAIL")
            failed += 1

    print("\n" + "=" * 68)
    print("DEMONSTRATION SUMMARY")
    print("=" * 68)
    print(f"Total tests: {len(TEST_CASES)}")
    print(f"Passed:      {passed}")
    print(f"Failed:      {failed}")

    if failed == 0:
        print("\nAll demonstration tests passed!")
    else:
        print("\nSome tests failed. Review the results above.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
